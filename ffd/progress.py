"""Thanh tiến trình (tqdm) cho các lệnh `ffd.cli`, cùng kiểu với `compsynth_ff.progress`. Ghi ra stderr; stdout giữ nguyên.

Trong terminal: thanh cập nhật tại chỗ. Khi stderr chuyển ra file (chạy nền, nohup): in thưa (mỗi 30 s) để log không phình.
Tắt bằng `--no-progress` (đặt ENABLED = False). Không ảnh hưởng logic, seed hay file kết quả.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import sys

ENABLED = True


def bar(iterable=None, *, total: int | None = None, desc: str = "", unit: str = "it", initial: int = 0,
        leave: bool = True):
    from tqdm import tqdm

    tty = sys.stderr.isatty()
    return tqdm(iterable, total=total, desc=desc, unit=unit, initial=initial, leave=leave, file=sys.stderr,
                disable=not ENABLED, dynamic_ncols=True, mininterval=0.5 if tty else 30.0, ascii=not tty)
