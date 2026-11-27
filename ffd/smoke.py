"""Smoke trên GPU với backbone thật (model_name): đo VRAM đỉnh, giây/bước; chép ảnh rep/seam để duyệt bằng mắt.

Ghi `<ckpt_root>/smoke/smoke.json` (peak_vram_gb, sec_per_step, batch_size, steps, model_revision, model_commit) và
`<ckpt_root>/smoke/samples/{rep,seam}_XX.<đuôi gốc>` (16 ảnh mỗi loại, chép nguyên file, trải đều theo manifest_aug).
Train trên TrainList arm raw_aug (chỉ đọc train/aug), không ghi vào run dir, không chạm val/test.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import progress
from .config import TrainConfig
from .data.trainset import TrainList
from .guards import allow_read
from .io_utils import write_json_atomic
from .model import Detector
from .train import epoch_batches, loader_kwargs

N_SAMPLES = 16


def dump_samples(data_root, kind: str, out: Path) -> int:
    """Chép N_SAMPLES ảnh `kind` (rep | seam) của manifest_aug, trải đều theo thứ tự đường dẫn."""
    from .data.ffpp import aug_frames

    allow_read("train", "ffpp_aug")
    man = aug_frames(data_root)
    paths = sorted(man[man.kind == kind].path, key=str)
    if len(paths) < N_SAMPLES:
        raise RuntimeError(f"Chỉ có {len(paths)}/{N_SAMPLES} ảnh '{kind}' trong manifest_aug")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(np.linspace(0, len(paths) - 1, N_SAMPLES).astype(int)):
        src = Path(paths[i])
        shutil.copyfile(src, out / f"{kind}_{n:02d}{src.suffix}")
    return N_SAMPLES


def smoke(cfg: TrainConfig, steps: int) -> Path:
    if steps < 2:
        raise ValueError(f"steps phải >= 2 (bỏ bước đầu khỏi thời gian), nhận {steps}")
    if not torch.cuda.is_available():
        raise RuntimeError("smoke cần CUDA (backbone thật).")
    out = Path(cfg.ckpt_root) / "smoke"
    for kind in ("rep", "seam"):
        dump_samples(cfg.data_root, kind, out / "samples")

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    model = Detector(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = TrainList(cfg.data_root, "raw_aug", cfg.seed, cfg.p_jpeg, cfg.jpeg_q)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    done, t0, epoch = 0, None, 0
    pb = progress.bar(total=steps, desc="smoke", unit="step")
    while done < steps:
        epoch += 1
        ds.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=epoch_batches(len(ds), cfg.batch_size, cfg.seed, epoch),
                                             **loader_kwargs(cfg.num_workers, device, cfg.seed))
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.head.loss(model(x), {"L0": y})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            pb.update(1)
            pb.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if done == 1:  # bỏ bước đầu (khởi tạo cuDNN/cache) khỏi thời gian
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if done == steps:
                break
    torch.cuda.synchronize()
    pb.close()
    res = {"peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
           "sec_per_step": (time.perf_counter() - t0) / (steps - 1),
           "batch_size": cfg.batch_size, "steps": steps,
           "model_revision": cfg.model_revision, "model_commit": model.trunk.commit}
    p = out / "smoke.json"
    write_json_atomic(res, p)
    return p
