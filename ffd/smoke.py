"""Smoke trên GPU với ViT-L thật: đo VRAM đỉnh, giây/step; xuất ảnh để duyệt tham số T của self-blend.

Ghi `<runs_root>/smoke/smoke.json` (peak_vram_gb, sec_per_step, batch_size, steps, model_revision, model_commit) và
`smoke_samples/{comp,sb}_XX.png` (16 ảnh mỗi loại; chuẩn hóa CLIP đã đảo ngược, lưu RGB->BGR bằng cv2).
Không ghi vào run dir của train, không chạm val/test.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import progress
from .config import TrainConfig
from .data.arms import ArmTrainDataset
from .data.transforms import CLIP_MEAN, CLIP_STD
from .model import Detector

N_SAMPLES = 16


def _to_png(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    rgb = (x.cpu() * std + mean).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _dump_samples(ff_cfg, kind: str, out: Path) -> int:
    """Lưu N_SAMPLES ảnh `kind` (rep | seam) của ffpp_aug/train để xem bằng mắt."""
    from compsynth_ff.aug_build import aug_root

    man = pd.read_parquet(aug_root(ff_cfg) / "manifest.parquet")
    paths = sorted(man[(man.kind == kind) & (man.skipped == "")].path)
    if len(paths) < N_SAMPLES:
        raise RuntimeError(f"Chỉ có {len(paths)}/{N_SAMPLES} ảnh '{kind}' trong ffpp_aug")
    for n, i in enumerate(np.linspace(0, len(paths) - 1, N_SAMPLES).astype(int)):
        shutil.copyfile(Path(ff_cfg.out_root) / paths[i], out / f"{kind}_{n:02d}.jpg")
    return N_SAMPLES


def smoke(cfg: TrainConfig, steps: int, *, ff_cfg=None) -> Path:
    if steps < 2:
        raise ValueError(f"steps phải >= 2 (bỏ step đầu khỏi thời gian), nhận {steps}")
    if not torch.cuda.is_available():
        raise RuntimeError("smoke cần CUDA (ViT-L thật).")
    from compsynth_ff.config import load_config

    ff_cfg = ff_cfg if ff_cfg is not None else load_config(cfg.ffpp_config)
    out = Path(cfg.runs_root) / "smoke"
    samples = out / "smoke_samples"
    samples.mkdir(parents=True, exist_ok=True)
    _dump_samples(ff_cfg, "rep", samples)
    _dump_samples(ff_cfg, "seam", samples)

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    model = Detector(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = ArmTrainDataset(ff_cfg, "raw+rep", cfg.rho, cfg.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                         num_workers=cfg.num_workers, drop_last=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    done, t0 = 0, None
    pb = progress.bar(total=steps, desc="smoke", unit="step")
    while done < steps:
        for batch in loader:
            x = batch["image"].to(device)
            tg = {k: v.to(device) for k, v in batch["targets"].items() if torch.is_tensor(v)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.head.loss(model(x), tg)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            pb.update(1)
            pb.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if done == 1:  # bỏ step đầu (khởi tạo cuDNN/cache) khỏi thời gian
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if done == steps:
                break
    torch.cuda.synchronize()
    pb.close()
    dt = time.perf_counter() - t0
    res = {"peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
           "sec_per_step": dt / max(steps - 1, 1) if steps > 1 else dt,
           "batch_size": cfg.batch_size, "steps": steps,
           "model_revision": cfg.model_revision, "model_commit": model.trunk.commit}
    p = out / "smoke.json"
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return p
