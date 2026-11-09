"""Bộ bốc recipe cho một mục fake của arm. Thuần numpy.

Công bằng giữa arm: comp và sb dùng CHUNG cách bốc (method, region) và cùng số lần gọi rng cho phần đó;
chỉ phần `borrowed` (chỉ sb) bốc thêm SAU, nên hai nguồn khác nhau đúng ở nội dung vùng ghép.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from compsynth_ff.composite import pick_borrowed_method

ARMS = ("raw", "raw+comp", "raw+sb")


class Recipe(NamedTuple):
    source: Literal["raw", "comp", "sb"]
    method: str
    region: str | None
    borrowed: str | None


def draw_source(rng: np.random.Generator, arm: str, rho: float) -> str:
    """Bốc nguồn MỘT lần cho mỗi mục (trước mọi resample frame): raw, hoặc comp/sb với xác suất rho."""
    if arm not in ARMS:
        raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")
    if arm != "raw" and rng.random() < rho:
        return "comp" if arm == "raw+comp" else "sb"
    return "raw"


def draw_method(rng: np.random.Generator, methods: tuple) -> str:
    return str(methods[int(rng.integers(0, len(methods)))])


def sample_recipe(rng: np.random.Generator, arm: str, rho: float, methods: tuple, valid_regions: dict,
                  source: str | None = None, method: str | None = None) -> Recipe | None:
    """Recipe cho MỘT frame. `source`/`method` cho sẵn thì giữ nguyên (không bốc lại), chỉ bốc phần còn thiếu.
    Method bốc đều trên `methods` của frame (spec 5.1: method trước). Trả None nếu (source, method) không dùng được
    trên frame này: method không có trong `methods`, hoặc comp/sb mà valid_regions[method] rỗng; khi đó caller đổi FRAME.
    sb: borrowed = method, nên ô mask (region, method) cùng phân phối với comp."""
    if source is None:
        source = draw_source(rng, arm, rho)
    if method is None:
        if not methods:
            return None
        method = draw_method(rng, methods)
    if method not in methods:
        return None
    if source == "raw":
        return Recipe("raw", method, None, None)
    regs = valid_regions.get(method)
    if not regs:
        return None
    r = str(regs[int(rng.integers(0, len(regs)))])
    return Recipe(source, method, r, method if source == "sb" else None)


def sample_item(rng: np.random.Generator, arm: str, rho: float, n_frames: int, info_fn, k: int,
                max_tries: int = 50) -> tuple:
    """(frame_index, Recipe). Nguồn bốc một lần; method bốc ở frame đầu có methods; chỉ FRAME được resample.
    info_fn(i) -> (methods, valid_regions). Quá max_tries thì RuntimeError."""
    source = draw_source(rng, arm, rho)
    m = None
    for _ in range(max_tries):
        methods, vr = info_fn(k)
        if m is None and methods:
            m = draw_method(rng, methods)
        rec = sample_recipe(rng, arm, rho, methods, vr, source=source, method=m)
        if rec is not None:
            return k, rec
        k = int(rng.integers(0, n_frames))
    raise RuntimeError(f"Không bốc được recipe sau {max_tries} lần (arm={arm})")
