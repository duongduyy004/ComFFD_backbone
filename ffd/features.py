"""Cache đặc trưng CLS của CLIP ViT đóng băng (spec v6, mục 2.1). Chưa chạy thử trên dữ liệu thật.

Mỗi frame lưu L + 1 slot (ViT-L: 25 slot x 1024): slot l-1 = CLS của hidden_states[l] (l = 1..L, residual stream,
chưa LayerNorm), slot L = pooler_output (CLS tầng cuối sau post_layernorm, đúng đầu vào head của arm `raw`).
Tính fp32 (không autocast; CUDA mặc định bật TF32, tắt bằng --no-tf32; `precision` nằm trong key), lưu fp16.

Bố cục `<feat_root>/<commit>/<dataset>/` (`feat_root` mặc định `<data_root>/features/clip`) (dataset ∈ train, val, ffpp_test, celebdf):
- `cls.npy`: memmap fp16 [N, L+1, D]; ViT-L cả 4 dataset ≈ 9.7 GB.
- `index.parquet`: video_id, frame_idx, label, method, path; cùng thứ tự `cls.npy` và `ffpp.frames()`.
- `state.json`: {"key": ..., "n_done": int}; cập nhật sau mỗi khối đã flush (chạy tiếp từ n_done).
- `meta.json`: ghi SAU CÙNG; có file này = cache hoàn tất. Gồm key + n_videos_dropped, phiên bản, seconds, verify.
key = commit, sha256 manifest (raw hoặc celebdf), sha256 params.json, transform_id, N, num_layers, dim. Thư mục đã có
mà key khác -> ValueError trước khi ghi gì.
Train = raw train (như arm `raw`), không tăng cường; mọi dataset qua `eval_transform`.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import progress
from .backbone import resolve_commit
from .data import transforms as T
from .data.ffpp import data_hashes, dropped_videos, frames
from .guards import allow_read
from .io_utils import write_json_atomic, write_parquet_atomic
from .train import EvalImages, loader_kwargs

FEAT_DATASETS = ("train", "val", "ffpp_test", "celebdf")
# dataset -> (nguồn, split, giai đoạn guard, tên guard)
_SRC = {"train": ("ffpp", "train", "train", "ffpp_train"), "val": ("ffpp", "val", "eval", "ffpp_val"),
        "ffpp_test": ("ffpp", "test", "eval", "ffpp_test"), "celebdf": ("celebdf", None, "eval", "celebdf")}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_INDEX_COLS = ["video_id", "frame_idx", "label", "method", "path"]
VERIFY_N = 16
VERIFY_TOL = 1e-2


def _build_model(model_name, model_revision, vision_config):
    """(CLIPVisionModel đóng băng fp32 ở eval, commit). vision_config (test) -> model ngẫu nhiên seed 0, commit "local"."""
    from transformers import CLIPVisionModel

    if vision_config is not None:
        torch.manual_seed(0)
        vit, commit = CLIPVisionModel(vision_config), "local"
    else:
        vit = CLIPVisionModel.from_pretrained(model_name, revision=model_revision)
        commit = resolve_commit(model_name, model_revision)
    vit.eval().requires_grad_(False)
    return vit.float(), commit


def cls_slots(vit, x: torch.Tensor) -> torch.Tensor:
    """[B, L+1, D] float32: CLS của hidden_states[1..L] rồi pooler_output."""
    o = vit(pixel_values=x, output_hidden_states=True)
    n = vit.config.num_hidden_layers
    return torch.stack([o.hidden_states[l][:, 0] for l in range(1, n + 1)] + [o.pooler_output], dim=1).float()


def transform_id() -> str:
    s = inspect.getsource(T.eval_transform) + inspect.getsource(T._to_tensor) + repr((T.CLIP_MEAN, T.CLIP_STD))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def default_feat_root(cfg) -> Path:
    """Thư mục cache mặc định: <data_root>/features/clip (cạnh dữ liệu; không ảnh hưởng data_hashes)."""
    return Path(cfg.data_root) / "features" / "clip"


def feat_dir_of(cfg, feat_root) -> Path:
    """<feat_root>/<commit>; commit = model_revision nếu là hex 40 ký tự, ngược lại resolve từ cache HF."""
    rev = cfg.model_revision
    commit = rev if rev and _HEX40.match(rev) else resolve_commit(cfg.model_name, rev)
    return Path(feat_root) / commit


def _frames(data_root, ds: str):
    src, split, stage, guard = _SRC[ds]
    allow_read(stage, guard)  # ngay trước khi mở
    return frames(data_root, src, split), dropped_videos(data_root, src, split)


def _check_key(out: Path, key: dict) -> None:
    for name in ("meta.json", "state.json"):
        p = out / name
        if not p.exists():
            continue
        got = json.loads(p.read_text(encoding="utf-8"))
        got = got.get("key", {}) if name == "state.json" else got
        diff = sorted(k for k in key if got.get(k) != key[k])
        if diff:
            raise ValueError(f"cache {out} ({name}) có khóa khác hiện tại: {diff}; xóa thư mục hoặc đổi --feat-root")


def _loader(paths, batch_size, num_workers, device):
    return torch.utils.data.DataLoader(EvalImages(paths), batch_size=batch_size, shuffle=False,
                                       **loader_kwargs(num_workers, device))


@torch.no_grad()
def _forward(vit, paths, batch_size, num_workers, device) -> np.ndarray:
    return np.concatenate([cls_slots(vit, x.to(device, non_blocking=True)).cpu().numpy()
                           for x in _loader(paths, batch_size, num_workers, device)])


def _precision(device, tf32: bool) -> str:
    """CUDA + tf32: bật TF32 cho matmul/cudnn (mantissa 10 bit, bằng fp16 lúc lưu; nhanh hơn fp32 2-3 lần trên Ampere+)."""
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    return "tf32" if cuda and tf32 else "fp32"


def _check_against(run, commit: str, ds: str, idx: pd.DataFrame) -> None:
    """Trước mọi forward: commit model của run phải trùng; thứ tự frame phải khớp scores_<ds>.parquet của run."""
    info = Path(run) / "run_info.json"
    if info.exists():
        mc = json.loads(info.read_text(encoding="utf-8")).get("model_commit")
        if mc is not None and mc != commit:
            raise ValueError(f"model_commit của {run} ({mc}) khác commit cache ({commit})")
    if ds in ("val", "ffpp_test", "celebdf"):
        check_order(idx, Path(run) / f"scores_{ds}.parquet")
        print(f"[features] {ds}: thứ tự frame khớp {run}", flush=True)


def _verify(vit, paths, arr, device) -> dict:
    n = min(VERIFY_N, len(paths))
    ref = _forward(vit, paths[:n], n, 0, device)
    got = np.asarray(arr[:n], dtype=np.float32)
    return {"n": n, **_verify_close(got, ref)}


def _verify_close(got: np.ndarray, ref: np.ndarray) -> dict:
    """So slot pooler cache (fp16) với tính lại (fp32): lệch <= VERIFY_TOL * max(1, |ref|) (tương đối, vì sai số làm
    tròn fp16 tăng theo |x|). Lệch hơn -> ValueError."""
    d = np.abs(got[:, -1] - ref[:, -1])
    if (d > VERIFY_TOL * np.maximum(1.0, np.abs(ref[:, -1]))).any():
        raise ValueError(f"kiểm tra cache lệch: max |pooler cache - tính lại| = {float(d.max())} vượt "
                         f"{VERIFY_TOL} x max(1, |giá trị|)")
    hid = np.abs(got[:, :-1] - ref[:, :-1]).max() / max(float(np.abs(ref[:, :-1]).max()), 1e-6)
    return {"max_abs_pooler": float(d.max()), "max_abs_ref": float(np.abs(ref[:, -1]).max()),
            "max_rel_hidden": float(hid)}


def _extract_one(vit, commit, data_root, out_dir: Path, ds, batch_size, num_workers, chunk, device, max_chunks,
                 precision, check_against):
    df, dropped = _frames(data_root, ds)
    h = data_hashes(data_root)
    n_layers, dim = vit.config.num_hidden_layers, vit.config.hidden_size
    key = {"commit": commit, "manifest_sha256": h["manifest_sha256"]["celebdf" if ds == "celebdf" else "raw"],
           "params_json_sha256": h["params_json_sha256"], "transform_id": transform_id(), "N": int(len(df)),
           "num_layers": int(n_layers), "dim": int(dim), "precision": precision}
    idx = pd.DataFrame({"video_id": df.video_id.astype(str), "frame_idx": df.frame_idx.astype(int),
                        "label": df.label.astype(int), "method": df.method.astype(str),
                        "path": df.path.map(str)})[_INDEX_COLS]
    if check_against is not None:
        _check_against(check_against, commit, ds, idx)
    out = out_dir / ds
    _check_key(out, key)
    if (out / "meta.json").exists():
        print(f"[features] {ds}: đã có, bỏ qua", flush=True)
        return
    st_path, npy = out / "state.json", out / "cls.npy"
    if st_path.exists():
        old = pd.read_parquet(out / "index.parquet")
        if not old.reset_index(drop=True).astype(str).equals(idx.astype(str)):
            raise ValueError(f"{out / 'index.parquet'} khác danh sách frame hiện tại; xóa thư mục cache")
        st = json.loads(st_path.read_text(encoding="utf-8"))
        n_done = int(st["n_done"])
        if st.get("batch_size") != batch_size:
            print(f"[features] {ds}: CẢNH BÁO chạy tiếp với batch {batch_size} khác lần trước ({st.get('batch_size')}); "
                  "phần còn lại có thể lệch ~1e-6 so với chạy liền", flush=True)
        arr = np.load(npy, mmap_mode="r+")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(idx, out / "index.parquet")
        arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(len(df), n_layers + 1, dim))
        n_done = 0
        write_json_atomic({"key": key, "n_done": 0, "batch_size": batch_size}, st_path)
    t0, k, n_start = time.time(), 0, n_done
    paths, N = list(idx.path), len(df)
    try:
        if n_done < N:
            # Một DataLoader cho phần còn lại; ghi khối (flush rồi state.json) ở ranh giới batch khi đã đủ >= chunk
            # dòng, nên chạy tiếp bắt đầu đúng ranh giới batch cũ (cùng cách chia batch với chạy liền).
            pos = n_done
            with torch.no_grad():
                for x in progress.bar(_loader(paths[n_done:], batch_size, num_workers, device),
                                      desc=f"features {ds}", unit="batch", leave=False):
                    f = cls_slots(vit, x.to(device, non_blocking=True)).cpu().numpy()
                    f16 = f.astype(np.float16)
                    if not (np.isfinite(f).all() and np.isfinite(f16).all()):  # NaN/inf hoặc tràn fp16 (> 65504)
                        raise ValueError(f"{ds}: đặc trưng không hữu hạn (hoặc tràn fp16) ở dòng "
                                         f"{pos}..{pos + len(f) - 1}")
                    arr[pos:pos + len(f)] = f16
                    pos += len(f)
                    if pos - n_done < chunk and pos < N:
                        continue
                    arr.flush()
                    n_done, k = pos, k + 1
                    write_json_atomic({"key": key, "n_done": n_done, "batch_size": batch_size}, st_path)
                    if k == 1:
                        rate = (n_done - n_start) / max(time.time() - t0, 1e-9)
                        print(f"[features] {ds}: {rate:.1f} frame/s, còn ~{(N - n_done) / rate / 60:.1f} phút "
                              f"({precision})", flush=True)
                    if max_chunks is not None and k >= max_chunks and n_done < N:
                        return
        verify = _verify(vit, paths, arr, device)
    finally:
        del arr  # nhả file trên Windows
    import transformers

    meta = {**key, "dataset": ds, "n_videos_dropped": int(dropped), "torch": torch.__version__,
            "transformers": transformers.__version__, "seconds": round(time.time() - t0, 1), "verify": verify}
    write_json_atomic(meta, out / "meta.json")
    print(f"[features] {ds}: {len(df)} frame xong ({meta['seconds']} s)", flush=True)


def extract_features(data_root, feat_root, *, datasets=FEAT_DATASETS, model_name, model_revision, batch_size=64,
                     num_workers=4, chunk=4096, device=None, vision_config=None, max_chunks=None, tf32: bool = True,
                     check_against=None) -> Path:
    """Cache CLS cho `datasets`; trả <feat_root>/<commit>. `max_chunks` chỉ để test mô phỏng ngắt.
    `check_against` (thư mục run, vd checkpoints/clip/raw): kiểm commit + thứ tự frame TRƯỚC khi forward."""
    datasets = tuple(datasets)
    bad = [d for d in datasets if d not in FEAT_DATASETS]
    if bad or not datasets:
        raise ValueError(f"datasets không hợp lệ {bad}; chỉ {list(FEAT_DATASETS)} (không rỗng)")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vit, commit = _build_model(model_name, model_revision, vision_config)
    vit.to(device)
    precision = _precision(device, tf32)
    out_dir = Path(feat_root) / commit
    for ds in datasets:
        _extract_one(vit, commit, data_root, out_dir, ds, batch_size, num_workers, chunk, device, max_chunks,
                     precision, check_against)
    return out_dir


def load_features(feat_dir, ds: str):
    """(cls memmap chỉ đọc [N, L+1, D], index DataFrame, meta dict). Thiếu meta.json -> FileNotFoundError."""
    out = Path(feat_dir) / ds
    if not (out / "meta.json").exists():
        raise FileNotFoundError(f"cache chưa hoàn tất: thiếu {out / 'meta.json'}")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return np.load(out / "cls.npy", mmap_mode="r"), pd.read_parquet(out / "index.parquet"), meta


def check_order(index: pd.DataFrame, scores_parquet) -> None:
    """Thứ tự (video_id, frame_idx, label) của cache phải khớp từng dòng với scores_<ds>.parquet của một run."""
    s = pd.read_parquet(scores_parquet)
    cols = ["video_id", "frame_idx", "label"]
    a = index[cols].reset_index(drop=True).astype(str)
    b = s[cols].reset_index(drop=True).astype(str)
    if len(a) != len(b) or not a.equals(b):
        raise ValueError(f"thứ tự frame của cache khác {scores_parquet} ({len(a)} vs {len(b)} dòng)")


def main(argv=None) -> int:
    from .config import load_train_config

    ap = argparse.ArgumentParser(prog="python -m ffd.features", description="Cache CLS CLIP đóng băng (spec v6)")
    ap.add_argument("--config", required=True, help="config.yaml của train (data_root, model, num_workers)")
    ap.add_argument("--feat-root", default=None, help="mặc định <data_root>/features/clip")
    ap.add_argument("--datasets", default=",".join(FEAT_DATASETS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--check-against", default=None,
                    help="thư mục run (vd checkpoints/clip/raw): kiểm commit và thứ tự frame trước khi trích")
    ap.add_argument("--no-tf32", action="store_true", help="tính fp32 thuần trên CUDA (chậm hơn 2-3 lần)")
    a = ap.parse_args(argv)
    cfg = load_train_config(a.config)
    datasets = tuple(d for d in a.datasets.split(",") if d)
    d = extract_features(cfg.data_root, a.feat_root or default_feat_root(cfg), datasets=datasets, model_name=cfg.model_name,
                         model_revision=cfg.model_revision, batch_size=a.batch_size, num_workers=cfg.num_workers,
                         device=a.device, tf32=not a.no_tf32, check_against=a.check_against)
    print(f"[features] xong: {d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
