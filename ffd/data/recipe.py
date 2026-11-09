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


def sample_recipe(rng: np.random.Generator, arm: str, rho: float, methods: tuple,
                  valid_regions: dict) -> Recipe | None:
    """Trả None nếu frame không có gì để bốc (resample frame khác)."""
    if arm not in ARMS:
        raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")
    source = "raw"
    if arm != "raw" and rng.random() < rho:
        source = "comp" if arm == "raw+comp" else "sb"
    if source == "raw":
        if not methods:
            return None
        return Recipe("raw", str(methods[int(rng.integers(0, len(methods)))]), None, None)
    ok = [m for m in methods if valid_regions.get(m)]
    if not ok:
        return None
    m = str(ok[int(rng.integers(0, len(ok)))])
    regs = valid_regions[m]
    r = str(regs[int(rng.integers(0, len(regs)))])
    if source == "comp":
        return Recipe("comp", m, r, None)
    cand = [x for x in ok if r in valid_regions[x]]  # chứa m nên không rỗng
    return Recipe("sb", m, r, pick_borrowed_method(rng, cand))
