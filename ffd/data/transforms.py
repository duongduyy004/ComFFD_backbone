"""Transform ảnh train/eval (đúng 224x224, không resize): JPEG (xác suất p_jpeg), lật ngang, BGR->RGB, chuẩn hóa CLIP.

JPEG gọi qua `ffd.io_utils.jpeg` (tra tên lúc gọi, để test đếm được tại một điểm).
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import cv2
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



# Arm raw_strongaug: suy giảm chất lượng áp NHƯ NHAU cho real và fake (không cần nhãn). Hằng số cố định trước khi
# chạy, không chỉnh theo val/test (17/11/2026). Thứ tự: resize xuống rồi lên, blur, JPEG, lật.
STRONG = {"p_resize": 0.5, "resize_scale": (0.4, 1.0), "p_blur": 0.5, "blur_sigma": (0.1, 1.5), "jpeg_q_min": 40}
_DOWN = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)
_UP = (cv2.INTER_LINEAR, cv2.INTER_CUBIC)


def _lerp(lo_hi: tuple, u: float) -> float:
    return lo_hi[0] + u * (lo_hi[1] - lo_hi[0])


def strong_train_transform(img_bgr: np.ndarray, rng: np.random.Generator, p_jpeg: float,
                           q_max: int) -> tuple[torch.Tensor, bool, bool]:
    """Resize xuống tỉ lệ s ~ U(resize_scale) rồi lên 224 (xác suất p_resize); Gaussian blur sigma ~ U(blur_sigma)
    (p_blur); JPEG q nguyên đều trong [jpeg_q_min, q_max] (p_jpeg, cùng p_jpeg với arm raw); lật ngang p=0.5.
    Trả (Tensor 3x224x224, đã JPEG?, đã lật?). Luôn bốc đủ 9 số ngẫu nhiên để chuỗi rng không phụ thuộc các xác suất."""
    qmin = STRONG["jpeg_q_min"]
    if q_max < qmin:
        raise ValueError(f"q_max={q_max} < jpeg_q_min={qmin}")
    u = rng.random(9)  # 0 resize?, 1 scale, 2 interp xuống, 3 interp lên, 4 blur?, 5 sigma, 6 jpeg?, 7 q, 8 lật
    img = np.ascontiguousarray(img_bgr)
    if img.shape[:2] != (SIZE, SIZE):
        raise ValueError(f"Ảnh phải {SIZE}x{SIZE}, nhận {img.shape[:2]} (không resize)")
    if u[0] < STRONG["p_resize"]:
        s = max(8, int(round(SIZE * _lerp(STRONG["resize_scale"], u[1]))))
        img = cv2.resize(img, (s, s), interpolation=_DOWN[min(int(u[2] * len(_DOWN)), len(_DOWN) - 1)])
        img = cv2.resize(img, (SIZE, SIZE), interpolation=_UP[min(int(u[3] * len(_UP)), len(_UP) - 1)])
    if u[4] < STRONG["p_blur"]:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=_lerp(STRONG["blur_sigma"], u[5]))
    did = bool(u[6] < p_jpeg)
    if did:
        img = _io.jpeg(img, min(q_max, qmin + int(u[7] * (q_max - qmin + 1))))
    flipped = bool(u[8] < 0.5)
    if flipped:
        img = img[:, ::-1]
    return _to_tensor(img), did, flipped


def eval_transform(img_bgr: np.ndarray) -> torch.Tensor:
    return _to_tensor(img_bgr)
