"""Transform ảnh train/eval: JPEG (nếu chưa có), lật ngang, BGR->RGB, chuẩn hóa CLIP.

JPEG gọi qua `compsynth_ff.composite.jpeg` (tra tên lúc gọi, để test đếm được tại một điểm).
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import numpy as np
import torch
from compsynth_ff import composite as _comp

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIZE = 336
_MEAN = torch.tensor(CLIP_MEAN).view(3, 1, 1)
_STD = torch.tensor(CLIP_STD).view(3, 1, 1)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    if img_bgr.shape[:2] != (SIZE, SIZE):
        raise ValueError(f"Ảnh phải {SIZE}x{SIZE}, nhận {img_bgr.shape[:2]} (không resize)")
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
    return (x - _MEAN) / _STD


def train_transform(img_bgr: np.ndarray, rng: np.random.Generator, already_jpeg: bool,
                    jpeg_q: tuple = (75, 95)) -> torch.Tensor:
    """JPEG q ~ U(jpeg_q) nếu chưa có (real, fake_raw); lật p=0.5. Trả FloatTensor 3x336x336."""
    if not already_jpeg:
        q = int(rng.integers(jpeg_q[0], jpeg_q[1] + 1))
        img_bgr = _comp.jpeg(img_bgr, q)
    if rng.random() < 0.5:
        img_bgr = img_bgr[:, ::-1]
    return _to_tensor(img_bgr)


def eval_transform(img_bgr: np.ndarray) -> torch.Tensor:
    return _to_tensor(img_bgr)
