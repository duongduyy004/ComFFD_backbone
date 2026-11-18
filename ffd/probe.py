"""Head probe trên cache CLS của CLIP đóng băng (spec v6, mục 2.2). Chưa chạy thử trên dữ liệu thật.

Arm (slot theo ffd.features: slot l-1 = CLS hidden_states[l], slot L = pooler_output):
- probe_last: pooler_output, tuyến tính.
- probe_multi: CLS tầng MULTI_LAYERS (8, 12, 16, 20, 24; residual, không phải pooler), chuẩn hóa từng chiều, nối, tuyến tính.
- probe_gate: CLS cả L tầng, trọng số softmax alpha theo tầng (khởi tạo đều) -> tổng có trọng số -> tuyến tính.
- probe_layer: một tầng `layer` (bảng từng tầng, CHỈ MÔ TẢ, không dùng để chọn tầng).
Huấn luyện: BCE, tỉ lệ lớp tự nhiên, AdamW lr 1e-3, batch 4096, tối đa 50 epoch, patience 5 theo BCE val; weight decay
chọn trong WD_GRID theo BCE val nhỏ nhất (hòa -> wd nhỏ hơn). Chỉ train + val được mở trong giai đoạn chọn; FF++ test
và Celeb-DF chỉ mở sau `_select_done_hook()`.
Run: <ckpt_root>/<arm>_s<seed> (probe_layer: probe_layer<l>_s<seed>): run_info.json, train_log.csv, head.pt,
scores_{val,ffpp_test,celebdf}.parquet (cột như ffd.evaluate), result.json.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import stats
from .evaluate import _SCORE_COLS
from .features import feat_dir_of, load_features
from .guards import allow_read
from .io_utils import write_json_atomic, write_parquet_atomic
from .metrics import frame_block
from .tables import write_text_atomic
from .train import _bce

PROBE_ARMS = ("probe_last", "probe_multi", "probe_gate", "probe_layer")
MULTI_LAYERS = (8, 12, 16, 20, 24)
WD_GRID = (1e-4, 1e-3, 1e-2)
LR = 1e-3
BATCH = 4096
MAX_EPOCHS = 50
PATIENCE = 5
_EVAL_GUARD = {"ffpp_test": "ffpp_test", "celebdf": "celebdf"}
SLOT_RAM_MAX = 2 * 1024**3  # byte fp16 tối đa nạp slot vào RAM (probe_multi train ≈ 1.3 GB; probe_gate dùng memmap)


def _select_done_hook() -> None:
    """Gọi ngay sau khi chốt weight decay (điểm mốc cho test chống leakage)."""


def slots_for(arm: str, num_layers: int, layer: int | None) -> tuple[int, ...]:
    if arm == "probe_last":
        return (num_layers,)
    if arm == "probe_multi":
        bad = [l for l in MULTI_LAYERS if l > num_layers]
        if bad:
            raise ValueError(f"MULTI_LAYERS {bad} vượt số tầng {num_layers}")
        return tuple(l - 1 for l in MULTI_LAYERS)
    if arm == "probe_gate":
        return tuple(range(num_layers))
    if arm == "probe_layer":
        if layer is None or not 1 <= int(layer) <= num_layers:
            raise ValueError(f"probe_layer cần layer trong [1, {num_layers}], nhận {layer!r}")
        return (int(layer) - 1,)
    raise ValueError(f"arm phải thuộc {PROBE_ARMS}, nhận {arm!r}")


class ProbeHead(nn.Module):
    def __init__(self, n_slots: int, dim: int, gate: bool):
        super().__init__()
        self.gate = bool(gate)
        if self.gate:
            self.w = nn.Parameter(torch.zeros(n_slots))
            self.fc = nn.Linear(dim, 1)
        else:
            self.fc = nn.Linear(n_slots * dim, 1)
        nn.init.zeros_(self.fc.weight)  # logistic regression lồi: khởi tạo 0, không phụ thuộc init ngẫu nhiên
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (torch.softmax(self.w, 0)[None, :, None] * x).sum(1) if self.gate else x.flatten(1)
        return self.fc(z).squeeze(-1)

    @property
    def alpha(self):
        return torch.softmax(self.w, 0).detach().cpu().numpy() if self.gate else None


def _check_cache_consistent(feat_dir) -> None:
    """Đọc meta.json (không đọc dữ liệu) của 4 dataset: commit, transform, params.json, số tầng, dim phải trùng; manifest
    FF++ trùng giữa train/val/ffpp_test. Lệch (vd chạy --datasets từng phần sau khi đổi transform) -> ValueError."""
    metas = {}
    for ds in ("train", "val", "ffpp_test", "celebdf"):
        p = Path(feat_dir) / ds / "meta.json"
        if not p.exists():
            raise FileNotFoundError(f"cache chưa hoàn tất: thiếu {p}")
        metas[ds] = json.loads(p.read_text(encoding="utf-8"))
    bad = [k for k in ("commit", "transform_id", "params_json_sha256", "num_layers", "dim")
           if len({json.dumps(m.get(k)) for m in metas.values()}) > 1]
    if len({metas[d].get("manifest_sha256") for d in ("train", "val", "ffpp_test")}) > 1:
        bad.append("manifest_sha256 (FF++)")
    if bad:
        raise ValueError(f"cache không nhất quán giữa các dataset ở {bad}; chạy lại ffd.features cho đủ 4 dataset")


def _slot_source(cls, slots, max_bytes: int = SLOT_RAM_MAX, chunk: int = 4096):
    """(nguồn, sl). Slot chọn nhỏ hơn max_bytes -> nạp fp16 [N, n_slots, D] vào RAM một lần, sl = None (tránh đọc lại
    cả 25 slot từ đĩa mỗi epoch). Lớn hơn (vd probe_gate) -> giữ memmap, sl = list(slots)."""
    s = list(slots)
    if len(cls) * len(s) * cls.shape[2] * 2 > max_bytes:
        return cls, s
    return np.concatenate([np.asarray(cls[i:i + chunk][:, s]) for i in range(0, len(cls), chunk)]), None


def _rows(src, ii, sl) -> np.ndarray:
    x = src[ii]
    return x if sl is None else x[:, sl]


def _norm_stats(src, sl, chunk: int = 1024):
    """mean/std [n_slots, D] trên train, hai lượt theo khối float64 (tránh triệt tiêu E[x²] − mean²);
    std < 1e-6 · max(1, |mean|) -> 1. Không hữu hạn -> ValueError."""
    n = len(src)
    tot = sum(_rows(src, slice(i, i + chunk), sl).astype(np.float64).sum(0) for i in range(0, n, chunk))
    mean = tot / n
    if not np.isfinite(mean).all():
        raise ValueError("đặc trưng train không hữu hạn (NaN/inf trong cache)")
    sq = sum(np.square(_rows(src, slice(i, i + chunk), sl).astype(np.float64) - mean).sum(0)
             for i in range(0, n, chunk))
    std = np.sqrt(sq / n)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("đặc trưng train không hữu hạn (NaN/inf trong cache)")
    std[std < 1e-6 * np.maximum(1.0, np.abs(mean))] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _batch(src, ii, sl, mean, std, device) -> torch.Tensor:
    x = _rows(src, ii, sl).astype(np.float32)
    x -= mean
    x /= std
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def _predict(head, src, sl, mean, std, device) -> np.ndarray:
    head.eval()
    out = [head(_batch(src, np.arange(i, min(i + BATCH, len(src))), sl, mean, std, device)).float().cpu().numpy()
           for i in range(0, len(src), BATCH)]
    return np.concatenate(out).astype(np.float64)


def _fit(tr, tr_sl, y_tr, va, va_sl, y_va, mean, std, gate, wd, seed, device, log):
    torch.manual_seed(seed)
    head = ProbeHead(mean.shape[0], mean.shape[1], gate).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    yt = torch.from_numpy(y_tr.astype(np.float32))
    best, best_state, best_epoch, bad = np.inf, None, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        perm, tot = torch.randperm(len(tr), generator=g).numpy(), 0.0
        for b in range(0, len(tr), BATCH):
            ii = np.sort(perm[b:b + BATCH])
            loss = nn.functional.binary_cross_entropy_with_logits(head(_batch(tr, ii, tr_sl, mean, std, device)),
                                                                  yt[ii].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ii)
        z = _predict(head, va, va_sl, mean, std, device)
        bce = _bce(z, y_va)
        if not np.isfinite(bce):
            raise ValueError(f"BCE val không hữu hạn (wd {wd}, epoch {epoch}); kiểm cache")
        log.append({"wd": wd, "epoch": epoch, "train_loss": tot / len(tr), "val_bce": bce,
                    "val_auc_frame": stats.auc(y_va, z)})
        if bce < best:
            best, best_state, best_epoch, bad = bce, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    head.load_state_dict(best_state)
    return head, best, best_epoch


def _run_name(arm: str, seed: int, layer) -> str:
    return f"probe_layer{int(layer)}_s{seed}" if arm == "probe_layer" else f"{arm}_s{seed}"


def run_probe(feat_dir, ckpt_root, arm: str, *, seed: int = 0, layer: int | None = None, device="cpu",
              force: bool = False) -> Path:
    device = torch.device(device)
    _check_cache_consistent(feat_dir)
    allow_read("train", "ffpp_train")
    tr, tr_idx, tr_meta = load_features(feat_dir, "train")
    allow_read("train", "ffpp_val")
    va, va_idx, _ = load_features(feat_dir, "val")
    slots = slots_for(arm, int(tr_meta["num_layers"]), layer)
    run = Path(ckpt_root) / _run_name(arm, seed, layer)
    constants = {"lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "wd_grid": list(WD_GRID),
                 "multi_layers": list(MULTI_LAYERS)}
    if (run / "run_info.json").exists():
        old = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        if old.get("constants") != constants or old.get("slots") != list(slots) or old.get("feat_meta") != tr_meta:
            raise ValueError(f"hằng số arm / slot / cache khác run cũ {run}; xóa thư mục run hoặc đổi ckpt_root")
    if (run / "result.json").exists() and not force:
        print(f"[probe] {run.name}: đã có result.json, bỏ qua (--force để chạy lại)", flush=True)
        return run

    y_tr, y_va = tr_idx.label.to_numpy(int), va_idx.label.to_numpy(int)
    if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
        raise ValueError("train hoặc val chỉ có một lớp")
    tr_src, tr_sl = _slot_source(tr, slots)
    va_src, va_sl = _slot_source(va, slots)
    mean, std = _norm_stats(tr_src, tr_sl)
    gate = arm == "probe_gate"
    log, best = [], None
    for wd in WD_GRID:  # tăng dần; chỉ thay khi BCE val nhỏ hơn hẳn -> hòa giữ wd nhỏ hơn
        head, bce, ep = _fit(tr_src, tr_sl, y_tr, va_src, va_sl, y_va, mean, std, gate, wd, seed, device, log)
        if best is None or bce < best[1]:
            best = (wd, bce, ep, head)
    wd_sel, bce_sel, ep_sel, head = best
    _select_done_hook()

    run.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mean": mean, "std": std, "slots": list(slots)}, run / "head.pt")
    pd.DataFrame(log).to_csv(run / "train_log.csv", index=False)
    res = {"arm": arm, "seed": seed, "wd_selected": wd_sel, "best_epoch": ep_sel, "val_bce": bce_sel}
    for ds in ("val", "ffpp_test", "celebdf"):
        if ds == "val":
            cls, idx, meta = va, va_idx, load_features(feat_dir, "val")[2]
        else:
            allow_read("eval", _EVAL_GUARD[ds])  # chỉ sau khi đã chốt wd
            cls, idx, meta = load_features(feat_dir, ds)
        df = idx.copy()
        src, sl = (va_src, va_sl) if ds == "val" else _slot_source(cls, slots)
        df["logit"] = _predict(head, src, sl, mean, std, device)
        write_parquet_atomic(df[_SCORE_COLS].reset_index(drop=True), run / f"scores_{ds}.parquet")
        res[ds] = frame_block(df, int(meta["n_videos_dropped"]))
    info = {"arm": arm, "seed": seed, "layer": layer, "slots": list(slots), "constants": constants,
            "feat_meta": tr_meta, "feat_dir": str(feat_dir), "wd_selected": wd_sel, "best_epoch": ep_sel}
    if gate:
        info["alpha"] = [float(a) for a in head.alpha]
    write_json_atomic(info, run / "run_info.json")
    write_json_atomic(res, run / "result.json")
    print(f"[probe] {run.name}: wd {wd_sel}, epoch {ep_sel}, AUC video val {res['val']['auc_video']:.4f} "
          f"ffpp_test {res['ffpp_test']['auc_video']:.4f} celebdf {res['celebdf']['auc_video']:.4f}", flush=True)
    return run


def layer_table(ckpt_root, num_layers: int, seed: int = 0) -> Path:
    """probe_layers.md: AUC video theo từng tầng (val, ffpp_test, celebdf). Chỉ mô tả."""
    L = ["# Probe từng tầng (AUC video)", "",
         f"Seed {seed}. **Bảng chỉ mô tả, không dùng để chọn tầng** (spec v6 mục 2.2).", "",
         "| tầng | val | ffpp_test | celebdf |", "|---|---|---|---|"]
    for l in range(1, num_layers + 1):
        run = Path(ckpt_root) / _run_name("probe_layer", seed, l)
        v = []
        for ds in ("val", "ffpp_test", "celebdf"):
            vs = stats.video_scores(pd.read_parquet(run / f"scores_{ds}.parquet"))
            v.append(stats.auc(vs.label, vs.score))
        L.append(f"| {l} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |")
    out = Path(ckpt_root) / "probe_layers.md"
    write_text_atomic("\n".join(L) + "\n", out)
    return out


def main(argv=None) -> int:
    from .config import load_train_config

    ap = argparse.ArgumentParser(prog="python -m ffd.probe", description="Probe trên cache CLS (spec v6)")
    ap.add_argument("--config", required=True, help="config.yaml của train (ckpt_root, model_revision)")
    ap.add_argument("--feat-root", default="features/clip")
    ap.add_argument("--arm", required=True, choices=PROBE_ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", default=None, help="số tầng hoặc 'all' (chỉ với probe_layer)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_train_config(a.config)
    fd = feat_dir_of(cfg, a.feat_root)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.arm != "probe_layer":
        if a.layer is not None:
            ap.error("--layer chỉ dùng với --arm probe_layer")
        run_probe(fd, cfg.ckpt_root, a.arm, seed=a.seed, device=device, force=a.force)
        return 0
    if a.layer == "all":
        n = int(load_features(fd, "train")[2]["num_layers"])
        for l in range(1, n + 1):
            run_probe(fd, cfg.ckpt_root, "probe_layer", seed=a.seed, layer=l, device=device, force=a.force)
        print(f"[probe] {layer_table(cfg.ckpt_root, n, a.seed)}", flush=True)
        return 0
    run_probe(fd, cfg.ckpt_root, "probe_layer", seed=a.seed, layer=int(a.layer), device=device, force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
