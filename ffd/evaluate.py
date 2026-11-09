"""Chấm điểm best.pt của một run trên tập test, ghi `scores_{dataset}.parquet` mỗi dataset (mức frame, kèm cờ).

Cột: run, mode, seed, split, dataset, video_id, frame_idx, label, logit, cell, flag.
Quy ước:
- `datasets` ⊆ {ffpp_test, compsynth_test, cdf2}; `allow_read("eval", ...)` gọi ngay trước khi mở từng nguồn.
- Chỉ nạp `best.pt` (qua `Detector.load_trainable`); cần `done.flag`; nếu run có prereg thì hash trong
  run_info.json phải bằng `file_hash` của file prereg HIỆN TẠI (sai/mất file -> ValueError).
- Detector dựng lại từ `run_info.json["config"]` (cấu hình đầy đủ do `rundir.write_run_info` ghi; run cũ không có
  khóa này -> ValueError, không đoán).
- ffpp_test: FFppRawCore(test) (tập video kỳ vọng từ frames.parquet subset test, hợp thêm discover_targets nếu data_root tồn tại;
  không có data_root -> expected_source="artifacts_only" ghi vào run/eval_info.json), video_id = "{tid}_{src}", cell = "". Video kỳ vọng (discover_targets, subset test) không còn frame keep
  -> một dòng frame_idx=-1, logit=0.0, flag="no_usable_frame"; video chấm được mà không thuộc tập kỳ vọng -> ValueError.
- compsynth_test: EvalCore(test), toàn bộ dòng manifest; video_id = "{tid}|real" | "{tid}|a0ff|{region}|{borrowed or method}"
  | "{tid}|{region}|{method}"; cell = "{region}|{method}" (fake) | "a0ff" | "real"; label = L0.
- cdf2: manifest của `prep_cross` (cross_roots[name] là thư mục chứa manifest.parquet + crops/, hoặc cha của nó
  chứa thư mục `name/`). Chỉ chấm frame keep=True. Video chỉ có dòng frame_idx == -1 & drop_reason == "no_usable_frame"
  -> một dòng frame_idx=-1, logit=0.0, flag="no_usable_frame"; drop_reason == "decode_error" -> flag="decode_error"
  (không bỏ video nào im lặng). Logit không hữu hạn -> ValueError.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from compsynth_ff.config import load_config
from compsynth_ff.io_utils import read_png, write_json_atomic, write_parquet_atomic

from .config import TrainConfig, config_hash, file_hash
from .data.transforms import eval_transform
from .guards import allow_read
from .model import Detector
from .train import _autocast

DATASETS = ("ffpp_test", "compsynth_test", "cdf2")
COLUMNS = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag"]


def _cfg_from_info(info: dict) -> TrainConfig:
    d = info.get("config")
    if not d:
        raise ValueError("run_info.json không có khóa 'config' (run cũ); không dựng lại được Detector")
    d = dict(d)
    for k in ("ffpp_config", "runs_root", "prereg"):
        if d.get(k) is not None:
            d[k] = Path(d[k])
    for k in ("lora_targets", "tap_layers"):
        d[k] = tuple(d.get(k, ()))
    return TrainConfig(**d)


def _check_run(run: Path) -> tuple[TrainConfig, dict]:
    if not (run / "done.flag").exists():
        raise ValueError(f"Run chưa xong (thiếu done.flag): {run}")
    if not (run / "best.pt").exists():
        raise ValueError(f"Thiếu best.pt: {run}")
    info_p = run / "run_info.json"
    if not info_p.exists():
        raise ValueError(f"Thiếu run_info.json: {run}")
    info = json.loads(info_p.read_text(encoding="utf-8"))
    cfg = _cfg_from_info(info)
    if info.get("prereg_hash"):
        if cfg.prereg is None or not Path(cfg.prereg).exists():
            raise ValueError(f"Run có prereg_hash nhưng không thấy file prereg {cfg.prereg}")
        now = file_hash(cfg.prereg)
        if now != info["prereg_hash"]:
            raise ValueError(f"Hash prereg khác lúc train ({info['prereg_hash']} != {now}); không chấm điểm")
    if config_hash(cfg) != info.get("config_hash"):
        raise ValueError(f"config dựng lại có hash {config_hash(cfg)} != config_hash trong run_info {info.get('config_hash')}")
    return cfg, info


@torch.no_grad()
def _score(model, n: int, get_img, bs: int, device) -> np.ndarray:
    out = []
    for s in range(0, n, bs):
        x = torch.stack([eval_transform(get_img(i)) for i in range(s, min(n, s + bs))]).to(device)
        with _autocast(device):
            out.append(model(x)["logit"].float().cpu())
    lg = torch.cat(out).numpy().astype(np.float64) if out else np.zeros(0)
    if not np.isfinite(lg).all():
        raise ValueError("Logit không hữu hạn khi chấm điểm")
    return lg


def _frame(df: pd.DataFrame, **const) -> pd.DataFrame:
    for k, v in const.items():
        df[k] = v
    return df


def _ffpp_test(model, ff_cfg, bs, device):
    """Trả (df, expected_source)."""
    from compsynth_ff.dataset import FFppRawCore
    from compsynth_ff.pairs import discover_targets
    from compsynth_ff.select_cells import load_frames

    allow_read("eval", "ffpp_test")
    core = FFppRawCore(ff_cfg, "test")
    tg = [core[i]["targets"] for i in range(len(core))]  # đọc ảnh hai lần: chấp nhận cho đơn giản
    lg = _score(model, len(core), lambda i: core[i]["image"], bs, device)
    df = pd.DataFrame({"video_id": [t["video_id"] for t in tg], "frame_idx": [int(t["frame_idx"]) for t in tg],
                       "label": [int(t["L0"]) for t in tg], "logit": lg, "cell": "", "flag": ""})
    # Đủ video: tập kỳ vọng dựng từ artifact tiền xử lý (frames.parquet, KHÔNG lọc keep) nên gồm cả video không còn
    # frame keep; nếu data_root còn tồn tại thì hợp thêm discover_targets (bắt target extract hỏng hoàn toàn).
    fr = load_frames(ff_cfg)
    fr = fr[(fr.subset == "test") & fr.src.isin(("real",) + tuple(ff_cfg.methods))]
    expected = {f"{t}_{s_}": (0 if s_ == "real" else 1) for t, s_ in fr[["tid", "src"]].drop_duplicates().itertuples(index=False)}
    if Path(ff_cfg.data_root).exists():
        source = "artifacts+data_root"
        for t in discover_targets(ff_cfg):
            if t.subset == "test":
                expected[f"{t.tid}_real"] = 0
                for m in t.fakes:
                    expected[f"{t.tid}_{m}"] = 1
    else:
        source = "artifacts_only"
    extra = set(df.video_id) - set(expected)
    if extra:
        raise ValueError(f"{len(extra)} video được chấm không thuộc tập test kỳ vọng (ví dụ {sorted(extra)[0]!r})")
    miss = sorted(set(expected) - set(df.video_id))
    if miss:
        df = pd.concat([df, pd.DataFrame({"video_id": miss, "frame_idx": -1, "label": [expected[v] for v in miss],
                                          "logit": 0.0, "cell": "", "flag": "no_usable_frame"})], ignore_index=True)
    return df, source


def _compsynth_test(model, ff_cfg, bs, device) -> pd.DataFrame:
    from compsynth_ff.dataset import L0_CODE, EvalCore

    allow_read("eval", "compsynth_test")
    core = EvalCore(ff_cfg, "test")
    vids, cells, labels, fidx = [], [], [], []
    for i in range(len(core)):
        m = core.meta(i)
        k, tid, r, me = m["kind"], m["tid"], m["region"], m["method"]
        if k == "real":
            v, c = f"{tid}|real", "real"
        elif k == "a0ff":
            bm = core.man["borrowed_method"].iloc[i]
            v, c = f"{tid}|a0ff|{r}|{bm if isinstance(bm, str) and bm else me}", "a0ff"
        else:
            v, c = f"{tid}|{r}|{me}", f"{r}|{me}"
        lab = L0_CODE[k]  # KeyError nếu kind lạ
        vids.append(v); cells.append(c); labels.append(lab); fidx.append(m["frame_idx"])  # noqa: E702
    lg = _score(model, len(core), lambda i: core[i]["image"], bs, device)
    return pd.DataFrame({"video_id": vids, "frame_idx": fidx, "label": labels, "logit": lg,
                         "cell": cells, "flag": ""})


def _cross(model, name, root, bs, device) -> pd.DataFrame:
    allow_read("eval", name)
    root = Path(root)
    base = root if (root / "manifest.parquet").exists() else root / name
    man = pd.read_parquet(base / "manifest.parquet")
    if (man.dataset != name).any():
        raise ValueError(f"manifest {base} chứa dataset khác {name!r}")
    keep = man[man.keep].sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    paths = [base / "crops" / r.video_id / f"{int(r.frame_idx):05d}.png" for r in keep.itertuples()]
    lg = _score(model, len(keep), lambda i: read_png(paths[i], cv2.IMREAD_COLOR), bs, device)
    parts = [pd.DataFrame({"video_id": keep.video_id, "frame_idx": keep.frame_idx.astype("int64"),
                           "label": keep.label.astype("int64"), "logit": lg, "cell": "", "flag": ""})]
    has_keep = set(keep.video_id)
    for flag in ("no_usable_frame", "decode_error"):
        f = man[(man.frame_idx == -1) & (man.drop_reason == flag)]
        if len(f) != f.video_id.nunique():
            raise ValueError(f"manifest có video trùng dòng {flag}")
        if set(f.video_id) & has_keep:
            raise ValueError(f"video vừa có frame keep vừa bị {flag}: manifest sai hợp đồng")
        parts.append(pd.DataFrame({"video_id": f.video_id, "frame_idx": -1, "label": f.label.astype("int64"),
                                   "logit": 0.0, "cell": "", "flag": flag}))
    out = pd.concat(parts, ignore_index=True)
    lost = set(man.video_id) - set(out.video_id)
    if lost:  # video không có frame keep và không có dòng cờ: manifest vi phạm hợp đồng, không bỏ im lặng
        raise ValueError(f"{len(lost)} video không có frame keep cũng không có dòng cờ (ví dụ {sorted(lost)[0]!r})")
    return out


def evaluate_run(run, datasets: list[str], cross_roots: dict, *, ff_cfg=None, vision_config=None,
                 device=None, batch_size: int = 16) -> Path:
    """Chấm điểm best.pt của `run` trên `datasets`, ghi `run/scores_{dataset}.parquet` mỗi dataset; trả thư mục run."""
    run = Path(run)
    bad = [d for d in datasets if d not in DATASETS]
    if bad:
        raise ValueError(f"datasets không hợp lệ {bad}; chỉ {list(DATASETS)}")
    if not datasets:
        raise ValueError("datasets rỗng")
    cfg, _ = _check_run(run)
    if "cdf2" in datasets and "cdf2" not in cross_roots:
        raise ValueError("Thiếu cross_roots['cdf2']")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if any(d != "cdf2" for d in datasets):
        ff_cfg = ff_cfg if ff_cfg is not None else load_config(cfg.ffpp_config)
    # Seed như train() trước khi dựng model: với `vision_config` ngẫu nhiên (test) phần đóng băng không nằm trong
    # best.pt nên phải tái tạo y hệt; với CLIP thật (from_pretrained) không ảnh hưởng.
    torch.manual_seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_trainable(ck["trainable"])
    model.eval()
    frames, eval_info = [], {}
    for ds in datasets:
        if ds == "ffpp_test":
            df, src = _ffpp_test(model, ff_cfg, batch_size, device)
            eval_info["ffpp_test"] = {"expected_source": src}
        elif ds == "compsynth_test":
            df = _compsynth_test(model, ff_cfg, batch_size, device)
        else:
            df = _cross(model, ds, cross_roots[ds], batch_size, device)
        frames.append(_frame(df, dataset=ds))
    df = pd.concat(frames, ignore_index=True)
    rid = f"compffd/{cfg.split}/seed{cfg.seed}" if cfg.mode == "compffd" else f"{cfg.mode}/seed{cfg.seed}"
    df = _frame(df, run=rid, mode=cfg.mode, seed=int(cfg.seed), split=cfg.split or "")
    df = df[COLUMNS].astype({"frame_idx": "int64", "label": "int64", "logit": "float64"})
    for ds, g in df.groupby("dataset", sort=False):  # một file mỗi dataset: chấm lại dataset nào chỉ thay file đó
        write_parquet_atomic(g.reset_index(drop=True), run / f"scores_{ds}.parquet")
    if eval_info:  # nguồn tập video kỳ vọng (artifacts_only = máy không có data_root FF++)
        path = run / "eval_info.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        write_json_atomic({**old, **eval_info}, path)
    return run
