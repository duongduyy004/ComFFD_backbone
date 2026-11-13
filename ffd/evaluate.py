"""Chấm điểm best.pt của một run trên tập test, ghi `scores_{dataset}.parquet` mỗi dataset (mức frame, kèm cờ).

Cột: run, mode, seed, split, dataset, video_id, frame_idx, label, logit, cell, flag, best_step (step của best.pt đã chấm).
Quy ước:
- `datasets` ⊆ {ffpp_test}; `allow_read("eval", ...)` gọi ngay trước khi mở từng nguồn.
- Chỉ nạp `best.pt` (qua `Detector.load_trainable`); cần `done.flag`; nếu run có prereg thì hash trong
  run_info.json phải bằng `file_hash` của file prereg HIỆN TẠI (sai/mất file -> ValueError).
- Artifact tiền xử lý phải trùng lúc train: `ffpp_raw_hash`, `ffpp_aug_hash` (config_hash trong hai manifest dưới
  ffpp_root dùng khi eval) bằng giá trị trong run_info; lệch -> ValueError. Commit HF của trunk dựng lại
  phải bằng `model_commit` trong run_info.
- Detector dựng lại từ `run_info.json["config"]` (cấu hình đầy đủ do `rundir.write_run_info` ghi; run cũ không có
  khóa này -> ValueError, không đoán).
- ffpp_test: FFppRaw(test) đọc ffpp_raw/test; tập video kỳ vọng = `expected_videos(ffpp_root, "test")` từ
  ffpp_raw/manifest.parquet (expected_source="ffpp_raw_manifest" ghi vào run/eval_info.json), video_id = "{tid}_{src}",
  cell = "". Video kỳ vọng không còn frame keep -> một dòng frame_idx=-1, logit=0.0, flag="no_usable_frame";
  video chấm được mà không thuộc tập kỳ vọng -> ValueError.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig, config_hash, file_hash
from .guards import allow_read
from .io_utils import write_json_atomic, write_parquet_atomic
from .model import Detector
from .rundir import ffpp_hashes
from .train import score_images

DATASETS = ("ffpp_test",)
COLUMNS = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag",
           "best_step"]


def _cfg_from_info(info: dict) -> TrainConfig:
    d = info.get("config")
    if not d:
        raise ValueError("run_info.json không có khóa 'config' (run cũ); không dựng lại được Detector")
    d = dict(d)
    for k in ("ffpp_root", "runs_root", "prereg"):
        if d.get(k) is not None:
            d[k] = Path(d[k])
    for k in ("lora_targets", "tap_layers"):
        d[k] = tuple(d.get(k, ()))
    if "jpeg_q" in d:
        d["jpeg_q"] = tuple(d["jpeg_q"])
    return TrainConfig(**d)


def _check_run(run: Path, ffpp_root=None, prereg=None) -> tuple[TrainConfig, dict, Path]:
    """Kiểm run (done, best.pt, prereg, config_hash, artifact tiền xử lý); trả (cfg, info, ffpp_root).
    ffpp_root None -> cfg.ffpp_root. `ffpp_root`, `prereg` (đường dẫn) thay cho giá trị lúc train khi eval trên máy
    khác; nội dung vẫn phải trùng (manifest hash, prereg_hash)."""
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
        pre = Path(prereg) if prereg is not None else cfg.prereg
        if pre is None or not Path(pre).exists():
            raise ValueError(f"Run có prereg_hash nhưng không thấy file prereg {pre} (khác máy: dùng --prereg)")
        now = file_hash(pre)
        if now != info["prereg_hash"]:
            raise ValueError(f"Hash prereg khác lúc train ({info['prereg_hash']} != {now}); không chấm điểm")
    if config_hash(cfg) != info.get("config_hash"):
        raise ValueError(f"config dựng lại có hash {config_hash(cfg)} != config_hash trong run_info {info.get('config_hash')}")
    for k in ("ffpp_raw_hash", "ffpp_aug_hash"):
        if k not in info:
            raise ValueError(f"run_info.json thiếu {k} (run cũ); không xác định được artifact tiền xử lý")
    root = Path(ffpp_root) if ffpp_root is not None else cfg.ffpp_root
    if root is None or not (Path(root) / "ffpp_raw" / "manifest.parquet").is_file():
        raise ValueError(f"Không thấy {root}/ffpp_raw/manifest.parquet (khác máy: dùng --ffpp-root)")
    now = ffpp_hashes(dataclasses.replace(cfg, ffpp_root=Path(root)))
    for k, v in now.items():
        if v != info[k]:
            raise ValueError(f"{k} khác lúc train ({info[k]} != {v}): artifact tiền xử lý FF++ đã đổi; không chấm điểm")
    return cfg, info, Path(root)


def _score(model, core, bs: int, device, num_workers: int, desc: str = "eval") -> np.ndarray:
    lg = score_images(model, core, range(len(core)), bs, device, num_workers, desc).numpy().astype(np.float64)
    if not np.isfinite(lg).all():
        raise ValueError("Logit không hữu hạn khi chấm điểm")
    return lg


def _frame(df: pd.DataFrame, **const) -> pd.DataFrame:
    for k, v in const.items():
        df[k] = v
    return df


def _ffpp_test(model, ffpp_root, bs, device, num_workers=0):
    """Trả (df, expected_source). Tập video kỳ vọng lấy từ ffpp_raw/manifest.parquet (gồm video không còn frame
    keep); video kỳ vọng không có frame -> một dòng frame_idx=-1, logit=0.0, flag="no_usable_frame"."""
    from .data import ffpp as D

    allow_read("eval", "ffpp_test")
    core = D.FFppRaw(ffpp_root, "test")
    lg = _score(model, core, bs, device, num_workers, "eval ffpp_test")
    df = pd.DataFrame({"video_id": [f"{t}_{s}" for t, _, s in core.items],  # nhãn lấy từ items, không đọc ảnh lần hai
                       "frame_idx": [int(f) for _, f, _ in core.items],
                       "label": [0 if s == "real" else 1 for _, _, s in core.items], "logit": lg, "cell": "", "flag": ""})
    expected = D.expected_videos(ffpp_root, "test")
    extra = set(df.video_id) - set(expected)
    if extra:
        raise ValueError(f"{len(extra)} video được chấm không thuộc tập test kỳ vọng (ví dụ {sorted(extra)[0]!r})")
    miss = sorted(set(expected) - set(df.video_id))
    if miss:
        df = pd.concat([df, pd.DataFrame({"video_id": miss, "frame_idx": -1, "label": [expected[v] for v in miss],
                                          "logit": 0.0, "cell": "", "flag": "no_usable_frame"})], ignore_index=True)
    return df, "ffpp_raw_manifest"


def evaluate_run(run, datasets: list[str], *, ffpp_root=None, prereg=None, vision_config=None,
                 device=None, batch_size: int = 16) -> Path:
    """Chấm điểm best.pt của `run` trên `datasets`, ghi `run/scores_{dataset}.parquet` mỗi dataset; trả thư mục run."""
    run = Path(run)
    bad = [d for d in datasets if d not in DATASETS]
    if bad:
        raise ValueError(f"datasets không hợp lệ {bad}; chỉ {list(DATASETS)}")
    if not datasets:
        raise ValueError("datasets rỗng")
    cfg, info, ffpp_root = _check_run(run, ffpp_root, prereg)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Seed như train() trước khi dựng model: với `vision_config` ngẫu nhiên (test) phần đóng băng không nằm trong
    # best.pt nên phải tái tạo y hệt; với CLIP thật (from_pretrained) không ảnh hưởng.
    torch.manual_seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    if model.trunk.commit != info.get("model_commit"):
        raise ValueError(f"commit model khác lúc train ({info.get('model_commit')} != {model.trunk.commit})")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_trainable(ck["trainable"])
    model.eval()
    frames, eval_info = [], {}
    for ds in datasets:
        df, src = _ffpp_test(model, ffpp_root, batch_size, device, cfg.num_workers)
        eval_info["ffpp_test"] = {"expected_source": src}
        frames.append(_frame(df, dataset=ds))
    df = pd.concat(frames, ignore_index=True)
    rid = f"{cfg.mode}/seed{cfg.seed}"
    df = _frame(df, run=rid, mode=cfg.mode, seed=int(cfg.seed), split="", best_step=int(ck["step"]))  # cột split giữ cho tương thích
    df = df[COLUMNS].astype({"frame_idx": "int64", "label": "int64", "logit": "float64", "best_step": "int64"})
    for ds, g in df.groupby("dataset", sort=False):  # một file mỗi dataset: chấm lại dataset nào chỉ thay file đó
        write_parquet_atomic(g.reset_index(drop=True), run / f"scores_{ds}.parquet")
    if eval_info:  # nguồn tập video kỳ vọng (artifacts_only = máy không có data_root FF++)
        path = run / "eval_info.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        write_json_atomic({**old, **eval_info}, path)
    return run
