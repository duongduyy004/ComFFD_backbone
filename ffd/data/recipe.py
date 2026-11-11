"""Bộ bốc nguồn và method cho một mục của arm. Thuần numpy.

Spec augmentation can_bang_duong_noi §4: arm `raw` (ảnh gốc) và `raw+rep`:
- lớp fake: với xác suất ρ dùng ảnh thay thế (rep), ngược lại fake gốc;
- lớp real: với xác suất ρ dùng real có đường nối (seam), ngược lại real gốc.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import numpy as np

ARMS = ("raw", "raw+rep")


def _check(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")


def draw_source(rng: np.random.Generator, arm: str, rho: float) -> str:
    """Nguồn của mục fake: "rep" với xác suất rho (chỉ raw+rep), ngược lại "raw"."""
    _check(arm)
    return "rep" if arm == "raw+rep" and rng.random() < rho else "raw"


def draw_real_source(rng: np.random.Generator, arm: str, rho: float) -> str:
    """Nguồn của mục real: "seam" với xác suất rho (chỉ raw+rep), ngược lại "raw"."""
    _check(arm)
    return "seam" if arm == "raw+rep" and rng.random() < rho else "raw"


def draw_method(rng: np.random.Generator, methods: tuple) -> str:
    return str(methods[int(rng.integers(0, len(methods)))])


def sample_item(rng: np.random.Generator, n_frames: int, methods_fn, k: int, max_tries: int = 50) -> tuple:
    """(frame_index, method): method đều trên method có ảnh ở frame k; frame không có method nào thì bốc frame khác.
    methods_fn(i) -> tuple method. Quá max_tries thì RuntimeError."""
    for _ in range(max_tries):
        methods = tuple(methods_fn(k))
        if methods:
            return k, draw_method(rng, methods)
        k = int(rng.integers(0, n_frames))
    raise RuntimeError(f"Không tìm được frame có fake sau {max_tries} lần")
