"""Chốt chặn leakage cho giai đoạn train: train không được đọc test hay cross-dataset.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from typing import Literal

from compsynth_ff.guards import LeakageError  # dùng lại cùng lớp lỗi của preprocessing

__all__ = ["LeakageError", "allow_read", "val_set_for"]

_TRAIN_FORBIDDEN = {"ffpp_test", "compsynth_test", "cross"}


def allow_read(stage: Literal["train", "eval"], what: str) -> None:
    """Giai đoạn `train` đọc `ffpp_test`, `compsynth_test` hoặc `cross` -> LeakageError."""
    if stage not in ("train", "eval"):
        raise ValueError(f"stage phải là 'train' hoặc 'eval', nhận {stage!r}")
    if stage == "train" and what in _TRAIN_FORBIDDEN:
        raise LeakageError(f"Giai đoạn train không được đọc {what!r} (chống leakage)")


def val_set_for(cfg) -> Literal["ffpp_val_raw", "compsynth_val_seen"]:
    """Tập val chọn checkpoint: arm -> FF++ val raw; compffd -> CompSynth val, chỉ ô seen."""
    return "compsynth_val_seen" if cfg.mode == "compffd" else "ffpp_val_raw"
