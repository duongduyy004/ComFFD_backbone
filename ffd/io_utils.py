"""I/O dùng chung của detector: ghi file nguyên tử, đọc ảnh, JPEG, seed theo mục.

Độc lập với preprocessing (compsynth_ff); giữ cùng công thức để kết quả không đổi khi tách (13/11/2026):
`item_seed` = sha1("|".join(parts))[:8], `jpeg` = cv2.imencode JPEG rồi imdecode.
Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json_atomic(obj, path: Path) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8"))


def read_image(path: Path) -> np.ndarray:
    """Ảnh màu BGR uint8 (PNG/JPEG); chạy được với đường dẫn Unicode trên Windows."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Không đọc được ảnh: {path}")
    return img


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        raise IOError("Không mã hóa được JPEG")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def item_seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8], 16)
