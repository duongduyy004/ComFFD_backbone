"""Dataset train theo arm (spec augmentation 11/11 §5).

Công bằng: ba arm chỉ khác NỘI DUNG lớp fake. comp và sb cùng `sample_sigma_q` (cùng σ, q), cùng method,
và CÙNG mask M₀ = TrainCore.sample_mask (cùng seed → cùng M₀), cùng đường ghép/feather/JPEG.
sample_mask trả None → cả hai arm dùng fake gốc của method đã bốc (fallback=True). real:fake = 1:1 (mục chẵn real, lẻ fake).
real và fake_raw: JPEG đúng một lần trong train_transform, q cùng khoảng ff_cfg.jpeg_q;
comp/sb đã JPEG trong composite nên train_transform(already_jpeg=True).
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import numpy as np
import torch
from compsynth_ff.align import EYE_DIST_336
from compsynth_ff.composite import DEFAULT_T, composite, item_seed, sample_sigma_q, selfblend
from compsynth_ff.config import load_config
from compsynth_ff.dataset import TrainCore
from compsynth_ff.guards import assert_subset
from compsynth_ff.select_cells import load_frames

from .recipe import sample_item
from .transforms import train_transform

MAX_TRIES = 50


def _assert_train_only(ff_cfg, frames, what):
    keys = set(frames)
    fr = load_frames(ff_cfg)
    sub = fr[[(t, int(f)) in keys for t, f in zip(fr.tid, fr.frame_idx)]].subset
    for s in sub.unique():
        assert_subset(s, {"train"}, what)


class ArmTrainDataset(torch.utils.data.Dataset):
    def __init__(self, ff_cfg, arm: str, rho: float, seed: int, sb_params: dict | None = None):
        self.ff_cfg, self.arm, self.rho, self.seed = ff_cfg, arm, float(rho), int(seed)
        self.sb_params = dict(DEFAULT_T if sb_params is None else sb_params)
        self.core = TrainCore(ff_cfg)  # chỉ video train
        _assert_train_only(ff_cfg, self.core.real_frames, "ArmTrainDataset")
        self.epoch = 0

    def __len__(self) -> int:
        return 2 * len(self.core.real_frames)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def _item(self, img_bgr, rng, already_jpeg, L0, source, method, region, fallback=False):
        x = train_transform(img_bgr, rng, already_jpeg, tuple(self.ff_cfg.jpeg_q))
        return {"image": x, "targets": {"L0": L0, "source": source, "method": method, "region": region,
                                        "fallback": bool(fallback)}}

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(item_seed(str(self.seed), str(self.epoch), str(idx)))
        c, n = self.core, len(self.core.real_frames)
        k = idx // 2
        if idx % 2 == 0:
            t, f = c.real_frames[k]
            return self._item(c.read_crop(t, f, "real"), rng, False, 0, "real", "", "")
        k, rec = sample_item(rng, self.arm, self.rho, n, lambda i: tuple(c.usable(*c.real_frames[i])), k,
                             MAX_TRIES)
        t, f = c.real_frames[k]
        if rec.source == "raw":
            return self._item(c.read_crop(t, f, rec.method), rng, False, 1, "raw", rec.method, "")
        real = c.read_crop(t, f, "real")
        sigma, q = sample_sigma_q(rng, EYE_DIST_336, self.ff_cfg)
        ms = c.sample_mask(t, f, rng, method=rec.method)
        if ms is None:  # không bốc được mask đủ lớn: cả hai arm về fake gốc (spec §2, theo dõi bằng QC Q9)
            return self._item(c.read_crop(t, f, rec.method), rng, False, 1, rec.source, rec.method, "", True)
        if rec.source == "comp":
            img, _ = composite(real, c.read_crop(t, f, rec.method), ms.M0, ms.M0, sigma, q)
        else:
            img = selfblend(real, ms.M0, sigma, q, rng, t_params=self.sb_params)
        return self._item(img, rng, True, 1, rec.source, rec.method, "+".join(ms.regions))


def build_train_dataset(cfg, ff_cfg=None):
    """arm -> ArmTrainDataset. `ff_cfg` mặc định nạp từ cfg.ffpp_config."""
    if ff_cfg is None:
        ff_cfg = load_config(cfg.ffpp_config)
    return ArmTrainDataset(ff_cfg, cfg.mode, cfg.rho, cfg.seed, cfg.selfblend)
