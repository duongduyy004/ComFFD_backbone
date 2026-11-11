"""Dataset train theo arm (spec augmentation can_bang_duong_noi §4).

- Ảnh gốc: `ffpp_raw/train` (FFppRawCore) — real và 5 fake, mỗi video căn độc lập.
- Ảnh tăng cường (chỉ arm raw+rep): `ffpp_aug/train` (AugCore) — rep (nhãn fake), seam (nhãn real).
real:fake = 1:1 (mục chẵn real, lẻ fake). ρ cho cả hai lớp.
Ảnh gốc nén JPEG một lần trong train_transform; ảnh tăng cường đã nén khi tạo (already_jpeg=True).
Frame không có đủ cả rep lẫn seam → dùng ảnh gốc, `fallback=True`, giữ nguồn đã bốc (cùng tập frame cho hai lớp).
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import numpy as np
import torch
from compsynth_ff.composite import item_seed
from compsynth_ff.config import load_config
from compsynth_ff.dataset import AugCore, FFppRawCore

from .recipe import ARMS, draw_real_source, draw_source, sample_item
from .transforms import train_transform

MAX_TRIES = 50


class ArmTrainDataset(torch.utils.data.Dataset):
    def __init__(self, ff_cfg, arm: str, rho: float, seed: int):
        if arm not in ARMS:
            raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")
        self.ff_cfg, self.arm, self.rho, self.seed = ff_cfg, arm, float(rho), int(seed)
        self.raw = FFppRawCore(ff_cfg, "train")  # guard: chỉ video train
        self._idx = {it: i for i, it in enumerate(self.raw.items)}
        self.real_frames = sorted((t, f) for t, f, m in self.raw.items if m == "real")
        self._methods: dict = {}
        for t, f, m in self.raw.items:
            if m != "real":
                self._methods.setdefault((t, f), []).append(m)
        self.aug = AugCore(ff_cfg) if arm == "raw+rep" else None
        self.epoch = 0

    def __len__(self) -> int:
        return 2 * len(self.real_frames)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def _raw_img(self, t, f, m):
        return self.raw[self._idx[(t, f, m)]]["image"]

    def _item(self, img_bgr, rng, already_jpeg, L0, source, method, region, fallback=False):
        x = train_transform(img_bgr, rng, already_jpeg, tuple(self.ff_cfg.jpeg_q))
        return {"image": x, "targets": {"L0": L0, "source": source, "method": method, "region": region,
                                        "fallback": bool(fallback)}}

    def _aug(self, kind, t, f, rng):
        """Ảnh tăng cường loại `kind` của frame (t, f), CHỈ khi frame có cả rep lẫn seam (Final I1): seam bị skip
        nhiều hơn rep, nên nếu không ghép cặp thì tỉ lệ ảnh tăng cường thực tế sẽ lệch giữa hai lớp."""
        paths = getattr(self.aug, kind)(t, f)
        other = getattr(self.aug, "seam" if kind == "rep" else "rep")(t, f)
        if not paths or not other:
            return None
        p = sorted(paths)[int(rng.integers(0, len(paths)))]
        return self.aug.read(p), self.aug.meta(p)

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(item_seed(str(self.seed), str(self.epoch), str(idx)))
        k = idx // 2
        if idx % 2 == 0:
            t, f = self.real_frames[k]
            src = draw_real_source(rng, self.arm, self.rho)
            if src == "seam":
                got = self._aug("seam", t, f, rng)
                if got is not None:
                    img, meta = got
                    return self._item(img, rng, True, 0, "seam", "", meta["regions"])
                return self._item(self._raw_img(t, f, "real"), rng, False, 0, "seam", "", "", True)
            return self._item(self._raw_img(t, f, "real"), rng, False, 0, "raw", "", "")
        src = draw_source(rng, self.arm, self.rho)
        k, m = sample_item(rng, len(self.real_frames), lambda i: self._methods.get(self.real_frames[i], ()), k,
                           MAX_TRIES)
        t, f = self.real_frames[k]
        if src == "rep":
            got = self._aug("rep", t, f, rng)
            if got is not None:
                img, meta = got
                return self._item(img, rng, True, 1, "rep", meta["method"], meta["regions"])
            return self._item(self._raw_img(t, f, m), rng, False, 1, "rep", m, "", True)
        return self._item(self._raw_img(t, f, m), rng, False, 1, "raw", m, "")


def build_train_dataset(cfg, ff_cfg=None):
    """arm -> ArmTrainDataset. `ff_cfg` mặc định nạp từ cfg.ffpp_config."""
    if ff_cfg is None:
        ff_cfg = load_config(cfg.ffpp_config)
    return ArmTrainDataset(ff_cfg, cfg.mode, cfg.rho, cfg.seed)
