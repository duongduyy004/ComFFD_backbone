"""Smoke trên GPU với ViT-L thật: đo VRAM đỉnh, giây/step; xuất ảnh để duyệt tham số T của self-blend.

Ghi `<runs_root>/smoke/smoke.json` (peak_vram_gb, sec_per_step, batch_size, steps, model_revision, model_commit) và
`smoke_samples/{comp,sb}_XX.png` (16 ảnh mỗi loại; chuẩn hóa CLIP đã đảo ngược, lưu RGB->BGR bằng cv2).
Không ghi vào run dir của train, không chạm val/test.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np
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


def _dump_samples(cfg: TrainConfig, ff_cfg, arm: str, source: str, out: Path) -> int:
    """Lưu N_SAMPLES ảnh có source == `source` (cùng arm/rho/T như train) để xem bằng mắt."""
    ds = ArmTrainDataset(ff_cfg, arm, 1.0, cfg.seed, cfg.selfblend)  # rho=1: mọi mẫu fake đi qua nhánh arm
    n = 0
    for idx in range(1, len(ds), 2):  # chỉ số lẻ = fake
        it = ds[idx]
        if it["targets"]["source"] != source:
            continue
        cv2.imwrite(str(out / f"{source}_{n:02d}.png"), _to_png(it["image"]))
        n += 1
        if n == N_SAMPLES:
            break
    if n < N_SAMPLES:
        raise RuntimeError(f"Chỉ lấy được {n}/{N_SAMPLES} ảnh '{source}' từ arm {arm}")
    return n


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
    arm_cfg = dataclasses.replace(cfg, mode="raw+comp") if cfg.mode == "compffd" else cfg
    _dump_samples(arm_cfg, ff_cfg, "raw+comp", "comp", samples)
    _dump_samples(arm_cfg, ff_cfg, "raw+sb", "sb", samples)

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    model = Detector(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = ArmTrainDataset(ff_cfg, "raw+comp", cfg.rho, cfg.seed, cfg.selfblend)
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
