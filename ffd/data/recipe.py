"""Bộ bốc recipe cho một mục fake của arm. Thuần numpy.

Spec augmentation 11/11 §5: recipe chỉ còn (source, method). Vùng và mask do compsynth_ff.TrainCore.sample_mask
bốc SAU recipe, dùng chung cho comp và sb, nên hai arm có cùng phân phối mask; chỉ nội dung vùng ghép khác nhau.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

ARMS = ("raw", "raw+comp", "raw+sb")


class Recipe(NamedTuple):
    source: Literal["raw", "comp", "sb"]
    method: str


def draw_source(rng: np.random.Generator, arm: str, rho: float) -> str:
    """Bốc nguồn MỘT lần cho mỗi mục (trước mọi resample frame): raw, hoặc comp/sb với xác suất rho."""
    if arm not in ARMS:
        raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")
    if arm != "raw" and rng.random() < rho:
        return "comp" if arm == "raw+comp" else "sb"
    return "raw"


def draw_method(rng: np.random.Generator, methods: tuple) -> str:
    return str(methods[int(rng.integers(0, len(methods)))])


def sample_recipe(rng: np.random.Generator, arm: str, rho: float, methods: tuple, source: str | None = None,
                  method: str | None = None) -> Recipe | None:
    """Recipe cho MỘT frame. `source`/`method` cho sẵn thì giữ nguyên. Method bốc đều trên `methods` của frame.
    Trả None nếu frame không có method dùng được hoặc method cho sẵn không có trên frame (caller đổi FRAME)."""
    if source is None:
        source = draw_source(rng, arm, rho)
    if method is None:
        if not methods:
            return None
        method = draw_method(rng, methods)
    if method not in methods:
        return None
    return Recipe(source, method)


def sample_item(rng: np.random.Generator, arm: str, rho: float, n_frames: int, methods_fn, k: int,
                max_tries: int = 50) -> tuple:
    """(frame_index, Recipe). Nguồn bốc một lần; method bốc ở frame đầu có method; chỉ FRAME được resample.
    methods_fn(i) -> tuple method dùng được ở frame i. Quá max_tries thì RuntimeError."""
    source = draw_source(rng, arm, rho)
    m = None
    for _ in range(max_tries):
        methods = tuple(methods_fn(k))
        if m is None and methods:
            m = draw_method(rng, methods)
        rec = sample_recipe(rng, arm, rho, methods, source=source, method=m)
        if rec is not None:
            return k, rec
        k = int(rng.integers(0, n_frames))
    raise RuntimeError(f"Không bốc được recipe sau {max_tries} lần (arm={arm})")
