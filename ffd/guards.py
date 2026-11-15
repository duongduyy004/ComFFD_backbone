"""Chốt chặn leakage cho giai đoạn train: train không được đọc test.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from typing import Literal

__all__ = ["LeakageError", "allow_read", "assert_subset", "val_set_for"]


class LeakageError(Exception):
    """Vi phạm quy tắc chống leakage (đọc sai split, train chạm test)."""


def assert_subset(subset: str, allowed: set, what: str) -> None:
    if subset not in allowed:
        raise LeakageError(f"{what}: subset '{subset}' không được phép (chỉ {sorted(allowed)})")

_TRAIN_ALLOWED = {"ffpp_train", "ffpp_val", "ffpp_aug"}
_EVAL_ALLOWED = {"ffpp_test"}  # chỉ tập test, không train/val


def allow_read(stage: Literal["train", "eval"], what: str) -> None:
    """Giai đoạn `train` chỉ được đọc nguồn train/val/aug (ffpp_train, ffpp_val, ffpp_aug); test hay nguồn lạ -> LeakageError."""
    if stage not in ("train", "eval"):
        raise ValueError(f"stage phải là 'train' hoặc 'eval', nhận {stage!r}")
    if stage == "train" and what not in _TRAIN_ALLOWED:
        raise LeakageError(f"Giai đoạn train không được đọc {what!r} (chống leakage); chỉ {sorted(_TRAIN_ALLOWED)}")
    if stage == "eval" and what not in _EVAL_ALLOWED:
        raise LeakageError(f"Giai đoạn eval chỉ được đọc {sorted(_EVAL_ALLOWED)}, nhận {what!r}")


def val_set_for(cfg) -> Literal["ffpp_val_raw"]:
    """Tập val chọn checkpoint: FF++ val raw, giống nhau cho mọi arm (spec augmentation 11/11)."""
    return "ffpp_val_raw"
