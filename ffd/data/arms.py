"""Dataset train theo arm và chế độ compffd.

Công bằng: ba arm chỉ khác NỘI DUNG lớp fake. comp và sb cùng `sample_sigma_q` (cùng σ, q),
cùng phân phối (method, region), cùng đường ghép/feather/JPEG (composite). real:fake = 1:1 (mục chẵn real, lẻ fake).
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
from compsynth_ff.regions import REGIONS
from compsynth_ff.select_cells import load_frames

from .recipe import sample_recipe
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
        self.core = TrainCore(ff_cfg, None)  # mọi ô hợp lệ, chỉ video train
        _assert_train_only(ff_cfg, self.core.real_frames, "ArmTrainDataset")
        self.epoch = 0
        self._info: dict = {}

    def __len__(self) -> int:
        return 2 * len(self.core.real_frames)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def _frame_info(self, t, f):
        k = (t, f)
        if k not in self._info:
            c = self.core
            methods = tuple(c.usable(t, f))
            vr = {m: tuple(r for r in REGIONS if (r, m) in c.allowed and c.cell_ok(t, f, m, r)) for m in methods}
            self._info[k] = (methods, vr)
        return self._info[k]

    def _item(self, img_bgr, rng, already_jpeg, L0, source, method, region):
        x = train_transform(img_bgr, rng, already_jpeg, tuple(self.ff_cfg.jpeg_q))
        return {"image": x, "targets": {"L0": L0, "source": source, "method": method, "region": region}}

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(item_seed(str(self.seed), str(self.epoch), str(idx)))
        c, n = self.core, len(self.core.real_frames)
        k = idx // 2
        if idx % 2 == 0:
            t, f = c.real_frames[k]
            return self._item(c.read_crop(t, f, "real"), rng, False, 0, "real", "", "")
        for _ in range(MAX_TRIES):
            t, f = c.real_frames[k]
            methods, vr = self._frame_info(t, f)
            rec = sample_recipe(rng, self.arm, self.rho, methods, vr)
            if rec is not None:
                break
            k = int(rng.integers(0, n))
        else:
            raise RuntimeError(f"Không bốc được recipe sau {MAX_TRIES} lần (arm={self.arm}, idx={idx})")
        if rec.source == "raw":
            return self._item(c.read_crop(t, f, rec.method), rng, False, 1, "raw", rec.method, "")
        real = c.read_crop(t, f, "real")
        sigma, q = sample_sigma_q(rng, EYE_DIST_336, self.ff_cfg)
        if rec.source == "comp":
            rm, dm = c.cell_masks(t, f, rec.region, rec.method)
            img, _ = composite(real, c.read_crop(t, f, rec.method), rm, dm, sigma, q)
            method = rec.method
        else:
            rm, dm = c.cell_masks(t, f, rec.region, rec.borrowed)
            img = selfblend(real, rm & dm, sigma, q, rng, t_params=self.sb_params)
            method = rec.borrowed
        return self._item(img, rng, True, 1, rec.source, method, rec.region)


class CompFFDDataset(torch.utils.data.Dataset):
    """compffd: TrainCore(split) (numpy, BGR, đã JPEG) bọc train_transform(already_jpeg=True)."""

    def __init__(self, ff_cfg, split_path, seed: int):
        self.ff_cfg, self.seed, self.epoch = ff_cfg, int(seed), 0
        self.core = TrainCore(ff_cfg, split_path, rho=0.0)
        _assert_train_only(ff_cfg, self.core.real_frames, "CompFFDDataset")

    def __len__(self) -> int:
        return len(self.core)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)
        self.core.set_epoch(e)

    def __getitem__(self, idx: int) -> dict:
        it = self.core[idx]
        rng = np.random.default_rng(item_seed(str(self.seed), str(self.epoch), str(idx), "flip"))
        cell = it["targets"]["cell"]
        region, _, method = cell.partition("|") if "|" in cell else ("", "", "")
        x = train_transform(it["image"], rng, True, tuple(self.ff_cfg.jpeg_q))
        L0 = int(it["targets"]["L0"])
        return {"image": x, "targets": {"L0": L0, "source": "compffd" if L0 else "real",
                                        "method": method, "region": region}}


def build_train_dataset(cfg, ff_cfg=None):
    """arm -> ArmTrainDataset; compffd -> CompFFDDataset. `ff_cfg` mặc định nạp từ cfg.ffpp_config."""
    if ff_cfg is None:
        ff_cfg = load_config(cfg.ffpp_config)
    if cfg.mode == "compffd":
        return CompFFDDataset(ff_cfg, cfg.split, cfg.seed)
    return ArmTrainDataset(ff_cfg, cfg.mode, cfg.rho, cfg.seed, cfg.selfblend)
