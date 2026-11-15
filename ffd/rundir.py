"""Thư mục run `<ckpt_root>/<run_name>` (mặc định checkpoints/clip/raw, checkpoints/clip/raw_aug): chống ghi đè
run đã xong, dọn kết quả cũ khi --force, run_info.json.

Quy tắc:
- Có `result.json` (đã eval xong) mà không `force` -> FileExistsError (kể cả khi resume).
- Không `resume`, không `force` mà còn file kết quả (STALE hoặc scores_*.parquet) -> FileExistsError.
- `force` xóa toàn bộ file kết quả (và error.json) rồi chạy lại từ đầu.
- `resume`: config_hash trong run_info.json phải khớp config hiện tại.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .config import TrainConfig, config_hash, run_name_of
from .io_utils import write_json_atomic

# File kết quả của một run (cùng scores_*.parquet). error.json chỉ bị xóa khi force, không chặn chạy lại.
STALE = ("best.pt", "last.pt", "train_log.csv", "result.json", "run_info.json", "done.flag")


def run_dir(cfg: TrainConfig) -> Path:
    return Path(cfg.ckpt_root) / run_name_of(cfg)


def _results(d: Path) -> list[Path]:
    return [d / n for n in STALE if (d / n).exists()] + sorted(d.glob("scores_*.parquet"))


def prepare_run_dir(cfg: TrainConfig, *, resume: bool, force: bool) -> Path:
    d = run_dir(cfg)
    if force:
        for p in _results(d) + [d / "error.json"]:
            p.unlink(missing_ok=True)
    if (d / "result.json").exists():
        raise FileExistsError(f"Run đã xong (có result.json): {d}; dùng force để chạy lại")
    if not resume:
        left = [p.name for p in _results(d)]
        if left:
            raise FileExistsError(f"Thư mục run còn kết quả cũ {left} trong {d}; dùng resume hoặc force")
    info = d / "run_info.json"
    if resume and info.exists():
        old = json.loads(info.read_text(encoding="utf-8")).get("config_hash")
        if old != config_hash(cfg):
            raise ValueError(f"config_hash khác run cũ ({old} != {config_hash(cfg)}); không resume")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "peft"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def _gpu():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def write_run_info(d, cfg: TrainConfig, extra: dict | None = None) -> None:
    """Ghi run_info.json: config đầy đủ (evaluate dựng lại Detector từ đây), config_hash, model_commit (extra),
    phiên bản thư viện, GPU, sha256 của params.json/manifest (`data_hashes`)."""
    from .data.ffpp import data_hashes

    extra = dict(extra or {})
    info = {
        "config": json.loads(json.dumps(dataclasses.asdict(cfg), default=str)),
        "config_hash": config_hash(cfg),
        "run_name": run_name_of(cfg),
        "model_commit": extra.pop("model_commit", None),
        "versions": _versions(),
        "gpu": _gpu(),
        "data_hashes": data_hashes(cfg.data_root),
        **extra,
    }
    write_json_atomic(info, Path(d) / "run_info.json")
