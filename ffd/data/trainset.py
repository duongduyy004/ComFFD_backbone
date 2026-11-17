"""Tập train cố định: raw train (+ ảnh tăng cường nếu arm raw_aug*; arm raw_aug_filt* lọc và cân bằng lại; arm
raw_aug_filt_loc thêm nhãn patch từ mask). Mỗi epoch đi hết mọi ảnh một lần (việc xáo trộn thuộc train loop).
Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import io_utils
from ..config import ALL_ARMS, FILTER_ARMS, LOC_ARMS, STRONG_ARMS
from ..guards import LeakageError, allow_read
from . import ffpp
from .augfilter import balance_aug, filter_aug
from .transforms import strong_train_transform, train_transform


def pool_mask(mask: np.ndarray, grid: int) -> np.ndarray:
    """Mask (H, W) -> (grid, grid) float32: tỉ lệ pixel mask (>0) trong mỗi ô cell x cell, cell = H // grid."""
    h, w = mask.shape
    if h != w or h % grid:
        raise ValueError(f"mask {h}x{w} không chia hết cho lưới {grid}")
    c = h // grid
    return (mask > 0).astype(np.float32).reshape(grid, c, grid, c).mean(axis=(1, 3)).astype(np.float32)


class TrainList(Dataset):
    """items: path, label, source ∈ {raw, rep, seam}, video_id (+ mask_path ở arm raw_aug_filt*).
    Phần tử i -> {image, label, source, jpeg}; arm raw_aug_filt_loc thêm patch_target [grid, grid], has_loc.
    `aug_stats`: thống kê lọc + cân bằng (rỗng ở arm cũ)."""

    def __init__(self, data_root, arm: str, seed: int, p_jpeg: float, jpeg_q: int, grid: int = 16):
        if arm not in ALL_ARMS:
            raise ValueError(f"arm phải thuộc {ALL_ARMS}, nhận {arm!r}")
        allow_read("train", "ffpp_train")
        raw = ffpp.frames(data_root, "ffpp", "train")
        parts = [pd.DataFrame({"path": raw.path, "label": raw.label.astype(int), "source": "raw",
                               "video_id": raw.video_id})]
        self.aug_stats: dict = {}
        if arm == "raw_aug" or arm in FILTER_ARMS:
            allow_read("train", "ffpp_aug")
            aug = ffpp.aug_frames(data_root)
            bad = sorted(set(aug.tid) - set(raw.tid.astype(str)))
            if bad:
                raise LeakageError(f"manifest_aug có tid không thuộc train (ví dụ {bad[0]!r})")
            if arm in FILTER_ARMS:
                aug, fst = filter_aug(aug)
                aug, bst = balance_aug(aug, int((raw.label == 0).sum()), int((raw.label == 1).sum()))
                self.aug_stats = {**fst, **bst}
            df = pd.DataFrame({"path": aug.path, "label": aug.label.astype(int), "source": aug.kind,
                               "video_id": aug.video_id})
            if arm in FILTER_ARMS:
                df["mask_path"] = aug.mask_path.where(aug.kind == "rep", np.nan)
                if arm in LOC_ARMS:
                    for mp in df.mask_path[df.source == "rep"]:
                        if not Path(mp).is_file():
                            raise FileNotFoundError(f"thiếu mask của ảnh rep: {mp}")
                    reps = df.mask_path[df.source == "rep"]
                    if len(reps):  # đọc một mask: cạnh phải chia hết cho lưới (lỗi sớm, không giữa epoch)
                        h, w = io_utils.read_image(Path(reps.iloc[0])).shape[:2]
                        if h != w or h % int(grid):
                            raise ValueError(f"mask {h}x{w} không chia hết cho lưới {grid} ({reps.iloc[0]})")
            parts.append(df)
        self.items = pd.concat(parts, ignore_index=True)
        self.arm, self.grid = arm, int(grid)
        self.seed, self.p_jpeg, self.jpeg_q, self.epoch = seed, p_jpeg, jpeg_q, 0
        self._paths = list(self.items.path)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng(io_utils.item_seed(str(self.seed), str(self.epoch), str(i)))
        tf = strong_train_transform if self.arm in STRONG_ARMS else train_transform  # strong: chỉ ảnh raw, mọi lớp như nhau
        x, jpg, flipped = tf(io_utils.read_image(self._paths[i]), rng, self.p_jpeg, self.jpeg_q)
        row = self.items.iloc[i]
        out = {"image": x, "label": float(row.label), "source": str(row.source), "jpeg": jpg}
        if self.arm in LOC_ARMS:
            target = np.zeros((self.grid, self.grid), np.float32)
            if row.source == "rep":
                target = pool_mask(io_utils.read_image(Path(row.mask_path)).max(axis=2), self.grid)
                if flipped:
                    target = target[:, ::-1]  # lật mask cùng ảnh
            out["patch_target"] = torch.from_numpy(np.ascontiguousarray(target))
            out["has_loc"] = bool(row.source != "raw" or row.label == 0)  # fake raw không có giám sát vị trí
        return out
