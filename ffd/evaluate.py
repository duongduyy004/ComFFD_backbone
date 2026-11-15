"""Chấm điểm best.pt của một run trên FF++ val, FF++ test, Celeb-DF; ghi `scores_{dataset}.parquet` và `result.json`.

Quy ước:
- Cần `done.flag`, `best.pt`, `run_info.json`; Detector dựng lại từ `run_info["config"]`, commit trunk phải bằng
  `model_commit`; `manifest_sha256.raw` của data hiện tại phải bằng giá trị lúc train (lệch -> ValueError).
- `allow_read("eval", ...)` gọi ngay trước khi mở từng nguồn (ffpp_val / ffpp_test / celebdf).
- scores_{dataset}.parquet: video_id, frame_idx, label, method, logit (dataset ∈ val, ffpp_test, celebdf).
- Dataset không nằm trong `datasets` -> không có khóa trong result.json; celebdf thiếu dữ liệu ->
  {"skipped": "không có <data_root>/celebdf/manifest.parquet"}.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from . import metrics
from .config import TrainConfig, config_hash
from .data import ffpp as D
from .guards import allow_read
from .io_utils import write_json_atomic, write_parquet_atomic
from .model import Detector
from .train import score_frames

DATASETS = ("val", "ffpp_test", "celebdf")
_GUARD = {"val": "ffpp_val", "ffpp_test": "ffpp_test", "celebdf": "celebdf"}
_SCORE_COLS = ["video_id", "frame_idx", "label", "method", "logit"]


def _cfg_from_info(info: dict, data_root=None) -> TrainConfig:
    d = info.get("config")
    if not d:
        raise ValueError("run_info.json không có khóa 'config'; không dựng lại được Detector")
    d = dict(d)
    for k in ("data_root", "ckpt_root"):
        if d.get(k) is not None:
            d[k] = Path(d[k])
    if data_root is not None:  # eval trên máy khác: đường dẫn thay, nội dung vẫn phải trùng (kiểm hash)
        d["data_root"] = Path(data_root)
    for k in ("lora_targets", "tap_layers"):
        d[k] = tuple(d.get(k, ()))
    return TrainConfig(**d)


def _frames_of(cfg: TrainConfig, ds: str) -> tuple[pd.DataFrame, int]:
    allow_read("eval", _GUARD[ds])  # ngay trước khi mở
    if ds == "val":
        return D.frames(cfg.data_root, "ffpp", "val"), D.dropped_videos(cfg.data_root, "ffpp", "val")
    if ds == "ffpp_test":
        return D.frames(cfg.data_root, "ffpp", "test"), D.dropped_videos(cfg.data_root, "ffpp", "test")
    return D.frames(cfg.data_root, "celebdf", None), D.dropped_videos(cfg.data_root, "celebdf", None)


def evaluate_run(run, *, data_root=None, datasets=DATASETS, vision_config=None, device=None,
                 batch_size: int = 32) -> Path:
    """Chấm best.pt của `run` trên `datasets`; ghi scores_*.parquet + result.json, trả đường dẫn result.json."""
    run = Path(run)
    datasets = tuple(datasets)
    bad = [d for d in datasets if d not in DATASETS]
    if bad or not datasets:
        raise ValueError(f"datasets không hợp lệ {bad}; chỉ {list(DATASETS)} (không rỗng)")
    for n in ("done.flag", "best.pt", "run_info.json"):
        if not (run / n).exists():
            raise ValueError(f"Thiếu {n} trong {run}")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    cfg = _cfg_from_info(info, data_root)
    if config_hash(cfg) != info.get("config_hash"):
        raise ValueError(f"config dựng lại có hash {config_hash(cfg)} != config_hash run_info {info.get('config_hash')}")
    rec = info.get("data_hashes") or {}
    rec_man = rec.get("manifest_sha256") or {}
    if D.data_hashes(cfg.data_root)["manifest_sha256"]["raw"] != rec_man.get("raw"):
        raise ValueError("manifest_raw khác lúc train (dữ liệu tiền xử lý đã đổi); không chấm điểm")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Seed như train() trước khi dựng model: với vision_config ngẫu nhiên (test) phần đóng băng phải tái tạo y hệt.
    torch.manual_seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    if model.trunk.commit != info.get("model_commit"):
        raise ValueError(f"commit model khác lúc train ({info.get('model_commit')} != {model.trunk.commit})")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_trainable(ck["trainable"])
    model.eval()

    done = json.loads((run / "done.flag").read_text(encoding="utf-8"))
    res = {
        "arm": cfg.arm,
        "config": {
            "model_name": cfg.model_name, "model_commit": info["model_commit"], "lr": cfg.lr,
            "max_epochs": cfg.max_epochs, "patience": cfg.patience, "p_jpeg": cfg.p_jpeg,
            "batch_size": cfg.batch_size, "seed": cfg.seed,
            "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "targets": list(cfg.lora_targets)},
            "params_json_sha256": rec.get("params_json_sha256"),
            "manifest_sha256": {k: rec_man.get(k) for k in ("raw", "aug")},
        },
        "best_epoch": int(done["best_epoch"]), "epochs_run": int(done["epochs_run"]),
        "stopped_early": bool(done["stopped_early"]), "total_steps": int(done["total_steps"]),
    }
    for ds in datasets:
        if ds == "celebdf" and not (D.celebdf_dir(cfg.data_root) / "manifest.parquet").exists():
            res[ds] = {"skipped": f"không có {cfg.data_root}/celebdf/manifest.parquet"}
            continue
        df, dropped = _frames_of(cfg, ds)
        df = df.copy()
        df["logit"] = score_frames(model, df, batch_size, device, cfg.num_workers, f"eval {ds}").astype("float64")
        write_parquet_atomic(df[_SCORE_COLS].reset_index(drop=True), run / f"scores_{ds}.parquet")
        res[ds] = metrics.frame_block(df, dropped)
        if ds == "ffpp_test":
            res[ds]["auc_per_method"] = metrics.auc_per_method(df)
    out = run / "result.json"
    write_json_atomic(res, out)
    return out
