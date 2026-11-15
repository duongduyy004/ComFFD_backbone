"""Tập train cố định: raw train (+ toàn bộ ảnh tăng cường nếu arm raw_aug). Mỗi epoch đi hết mọi ảnh một lần
(việc xáo trộn thuộc train loop). Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .. import io_utils
from ..config import ARMS
from ..guards import LeakageError, allow_read
from . import ffpp
from .transforms import train_transform


class TrainList(Dataset):
    """items: path, label, source ∈ {raw, rep, seam}, video_id. Phần tử i -> {image, label, source, jpeg}."""

    def __init__(self, data_root, arm: str, seed: int, p_jpeg: float, jpeg_q: int):
        if arm not in ARMS:
            raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")
        allow_read("train", "ffpp_train")
        raw = ffpp.frames(data_root, "ffpp", "train")
        parts = [pd.DataFrame({"path": raw.path, "label": raw.label.astype(int), "source": "raw",
                               "video_id": raw.video_id})]
        if arm == "raw_aug":
            allow_read("train", "ffpp_aug")
            aug = ffpp.aug_frames(data_root)
            bad = sorted(set(aug.tid) - set(raw.tid.astype(str)))
            if bad:
                raise LeakageError(f"manifest_aug có tid không thuộc train (ví dụ {bad[0]!r})")
            parts.append(pd.DataFrame({"path": aug.path, "label": aug.label.astype(int), "source": aug.kind,
                                       "video_id": aug.video_id}))
        self.items = pd.concat(parts, ignore_index=True)
        self.seed, self.p_jpeg, self.jpeg_q, self.epoch = seed, p_jpeg, jpeg_q, 0
        self._paths = list(self.items.path)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng(io_utils.item_seed(str(self.seed), str(self.epoch), str(i)))
        x, jpg = train_transform(io_utils.read_image(self._paths[i]), rng, self.p_jpeg, self.jpeg_q)
        row = self.items.iloc[i]
        return {"image": x, "label": float(row.label), "source": str(row.source), "jpeg": jpg}
