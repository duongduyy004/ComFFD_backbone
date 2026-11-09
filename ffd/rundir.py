"""Thư mục run: bố cục, chống ghi đè run đã xong, run_info.json.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import json
from pathlib import Path

from compsynth_ff.io_utils import write_json_atomic

from .config import TrainConfig, config_hash, file_hash


def run_dir(cfg: TrainConfig) -> Path:
    if cfg.mode == "compffd":
        return Path(cfg.runs_root) / "compffd" / cfg.split / f"seed{cfg.seed}"
    return Path(cfg.runs_root) / "aug" / cfg.mode / f"seed{cfg.seed}"


def prepare_run_dir(cfg: TrainConfig, *, resume: bool, force: bool) -> Path:
    d = run_dir(cfg)
    if (d / "best.pt").exists() and (d / "done.flag").exists() and not force:
        raise FileExistsError(f"Run đã xong: {d} (dùng force để chạy lại)")
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


def write_run_info(dir, cfg: TrainConfig, extra: dict) -> None:
    """Ghi run_info.json. extra có thể chứa model_revision, split_file (đường dẫn để băm), ..."""
    extra = dict(extra or {})
    split_file = extra.pop("split_file", None)
    info = {
        "config_hash": config_hash(cfg),
        "prereg_hash": file_hash(cfg.prereg) if cfg.prereg else None,
        "split_hash": file_hash(split_file) if split_file else None,
        "versions": _versions(),
        "model_name": cfg.model_name,
        "model_revision": extra.pop("model_revision", None),
        "gpu": _gpu(),
        "seed": cfg.seed,
        "mode": cfg.mode,
        **extra,
    }
    write_json_atomic(info, Path(dir) / "run_info.json")
