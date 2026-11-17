"""Transform ảnh train/eval (đúng 224x224, không resize): JPEG (xác suất p_jpeg), lật ngang, BGR->RGB, chuẩn hóa CLIP.

JPEG gọi qua `ffd.io_utils.jpeg` (tra tên lúc gọi, để test đếm được tại một điểm).
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import io_utils as _io

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIZE = 224
_MEAN = torch.tensor(CLIP_MEAN).view(3, 1, 1)
_STD = torch.tensor(CLIP_STD).view(3, 1, 1)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    if img_bgr.shape[:2] != (SIZE, SIZE):
        raise ValueError(f"Ảnh phải {SIZE}x{SIZE}, nhận {img_bgr.shape[:2]} (không resize)")
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
    return (x - _MEAN) / _STD


def train_transform(img_bgr: np.ndarray, rng: np.random.Generator, p_jpeg: float,
                    q: int) -> tuple[torch.Tensor, bool, bool]:
    """JPEG chất lượng q với xác suất p_jpeg, rồi lật ngang p=0.5. Trả (Tensor 3x224x224, đã JPEG?, đã lật?).
    Luôn bốc rng.random() cho JPEG để chuỗi ngẫu nhiên không phụ thuộc p_jpeg."""
    did = bool(rng.random() < p_jpeg)
    if did:
        img_bgr = _io.jpeg(img_bgr, q)
    flipped = bool(rng.random() < 0.5)
    if flipped:
        img_bgr = img_bgr[:, ::-1]
    return _to_tensor(img_bgr), did, flipped


def eval_transform(img_bgr: np.ndarray) -> torch.Tensor:
    return _to_tensor(img_bgr)
