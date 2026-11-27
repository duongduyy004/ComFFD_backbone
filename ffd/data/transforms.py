"""Transform ảnh train/eval: resize về 224x224, JPEG (xác suất p_jpeg), lật ngang, BGR->RGB, thang [0, 1].
Chuẩn hóa mean/std theo backbone nằm trong model (`ffd.backbone.Normalize`, khóa `image_mean`/`image_std`).

Preprocessing lưu crop ở độ phân giải gốc (JPEG q95, không vuông). `to_size` kéo thẳng về 224x224, không giữ tỉ lệ.
INTER_AREA khi thu nhỏ cả hai chiều, INTER_CUBIC trong mọi trường hợp còn lại
(`thiet_ke_preprocessing_v2_2026-11-26.md` mục 7). Ảnh đã 224x224 đi qua nguyên vẹn.
JPEG lúc train áp **sau** resize.
JPEG gọi qua `ffd.io_utils.jpeg` (tra tên lúc gọi, để test đếm được tại một điểm).
Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .. import io_utils as _io

SIZE = 224


def to_size(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Kéo ảnh (H, W[, C]) về size x size. Đã đúng kích thước -> trả nguyên ảnh."""
    h, w = img.shape[:2]
    if (h, w) == (size, size):
        return img
    interp = cv2.INTER_AREA if h >= size and w >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    if img_bgr.shape != (SIZE, SIZE, 3):
        raise ValueError(f"Ảnh phải {SIZE}x{SIZE}x3 sau to_size, nhận {img_bgr.shape}")
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
    return x


def train_transform(img_bgr: np.ndarray, rng: np.random.Generator, p_jpeg: float,
                    q: int) -> tuple[torch.Tensor, bool, bool]:
    """Resize về 224, JPEG chất lượng q với xác suất p_jpeg, rồi lật ngang p=0.5.
    Trả (Tensor 3x224x224, đã JPEG?, đã lật?).
    Luôn bốc rng.random() cho JPEG để chuỗi ngẫu nhiên không phụ thuộc p_jpeg."""
    img_bgr = to_size(img_bgr)
    did = bool(rng.random() < p_jpeg)
    if did:
        img_bgr = _io.jpeg(img_bgr, q)
    flipped = bool(rng.random() < 0.5)
    if flipped:
        img_bgr = img_bgr[:, ::-1]
    return _to_tensor(img_bgr), did, flipped


def eval_transform(img_bgr: np.ndarray) -> torch.Tensor:
    return _to_tensor(to_size(img_bgr))
