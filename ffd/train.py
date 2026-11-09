"""Vòng train: val định kỳ, chọn best.pt theo (AUC video, BCE), last.pt + resume, chặn NaN.

Quy ước:
- 1 "step" = 1 lần cập nhật optimizer (gồm `grad_accum` micro-batch). Mỗi epoch xáo thứ tự bằng
  Generator(seed, epoch), drop_last; vị trí trong epoch suy ra từ step nên resume không cần
  đọc lại dữ liệu đã qua. `dataset.set_epoch(e)` gọi ở tiến trình chính mỗi đầu epoch, DataLoader
  `persistent_workers=False` nên worker tạo lại mỗi epoch và thấy epoch mới.
- Thứ tự trong một step: cập nhật -> train_log -> val (nếu eval_every) -> last.pt (nếu ckpt_every).
  Step cuối luôn val và lưu last.pt.
- Cần CUDA: không có CUDA và `allow_cpu=False` -> RuntimeError. `allow_cpu=True` chỉ dùng trong test
  (CPU, không autocast). `ff_cfg` cho phép test truyền Config tổng hợp thay vì nạp từ cfg.ffpp_config.
- Val: arm -> FFppRawCore(val); compffd -> EvalCore(val) lọc real + ô seen của split (cfg.split là
  đường dẫn file split, giống build_train_dataset). video_id compffd: "{tid}|real" hoặc "{tid}|{region}|{method}".
- Chọn best: AUC video cao hơn hẳn (> tol); hòa trong tol thì BCE thấp hơn.
Trạng thái: chưa chạy thử (chỉ test CPU với CLIP tí hon ngẫu nhiên).
"""
from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from compsynth_ff.config import load_config
from compsynth_ff.io_utils import write_json_atomic

from .config import TrainConfig
from .data.arms import build_train_dataset
from .data.transforms import eval_transform
from .guards import allow_read, val_set_for
from .model import Detector
from .rundir import prepare_run_dir, run_dir, write_run_info  # noqa: F401  (run_dir cho test/tiện ích)
from .stats import auc, video_scores


def is_better(new, best, tol: float = 1e-4) -> bool:
    """new/best = (AUC video, BCE). AUC hơn > tol thắng; |ΔAUC| <= tol thì BCE thấp hơn thắng."""
    if best is None:
        return True
    if new[0] > best[0] + tol:
        return True
    if new[0] < best[0] - tol:
        return False
    return new[1] < best[1]


# ---------------------------------------------------------------- val
def _val_items(cfg: TrainConfig, ff_cfg):
    """Trả (core, rows); rows = list (index trong core, video_id, frame_idx, label)."""
    from compsynth_ff import dataset as D

    allow_read("train", val_set_for(cfg))
    if cfg.mode != "compffd":
        core = D.FFppRawCore(ff_cfg, "val")
        rows = [(i, f"{t}_{s}", int(f), 0 if s == "real" else 1) for i, (t, f, s) in enumerate(core.items)]
        return core, rows
    sp = json.loads(Path(cfg.split).read_text(encoding="utf-8"))
    seen = {tuple(x) for x in sp["seen"]}
    core = D.EvalCore(ff_cfg, "val")
    rows = []
    for i in range(len(core)):
        m = core.meta(i)
        if m["kind"] == "real":
            rows.append((i, f"{m['tid']}|real", m["frame_idx"], 0))
        elif (m["region"], m["method"]) in seen:
            rows.append((i, f"{m['tid']}|{m['region']}|{m['method']}", m["frame_idx"], 1))
    return core, rows


def _autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def val_scores(model, cfg: TrainConfig, ff_cfg, batch_size: int | None = None, device=None) -> pd.DataFrame:
    """[video_id, frame_idx, label, logit]; model trở về chế độ cũ (train/eval) sau khi tính."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    bs = batch_size or cfg.batch_size
    core, rows = _val_items(cfg, ff_cfg)
    was = model.training
    model.eval()
    logits = []
    try:
        for s in range(0, len(rows), bs):
            chunk = rows[s:s + bs]
            x = torch.stack([eval_transform(core[r[0]]["image"]) for r in chunk]).to(device)
            with _autocast(device):
                logits.append(model(x)["logit"].float().cpu())
    finally:
        model.train(was)
    lg = torch.cat(logits).numpy() if logits else np.zeros(0)
    return pd.DataFrame({"video_id": [r[1] for r in rows], "frame_idx": [r[2] for r in rows],
                         "label": [r[3] for r in rows], "logit": lg})


def _bce(df: pd.DataFrame) -> float:
    z, y = df.logit.to_numpy(np.float64), df.label.to_numpy(np.float64)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def _val_metrics(df: pd.DataFrame) -> dict:
    v = video_scores(df)
    return {"auc_video": auc(v.label, v.score), "auc_frame": auc(df.label, df.logit), "bce": _bce(df)}


# ---------------------------------------------------------------- tiện ích
def _atomic_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_last(path: Path, payload: dict) -> None:
    _atomic_save(payload, path)


def _append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _truncate_jsonl(path: Path, step: int) -> None:
    if not path.exists():
        return
    keep = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and json.loads(ln)["step"] <= step]
    path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _worker_init(seed: int):
    def fn(worker_id: int):
        s = (seed * 1_000_003 + worker_id) % (2**32)
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
    return fn


def _epoch_batches(n: int, bs: int, seed: int, epoch: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    return [perm[i:i + bs] for i in range(0, n - bs + 1, bs)]  # drop_last


def _video_of(ds, idx: int) -> str:
    """tid của video mà mục idx lấy từ (real ở idx chẵn; fake lẻ của compffd lấy từ khe fake)."""
    c, k = ds.core, idx // 2
    if idx % 2 == 1 and hasattr(c, "fake_slots"):
        return str(c.fake_slots[k % len(c.fake_slots)][0])
    return str(c.real_frames[k % len(c.real_frames)][0])


# ---------------------------------------------------------------- train
def train(cfg: TrainConfig, *, resume: bool = False, force: bool = False, vision_config=None,
          ff_cfg=None, allow_cpu: bool = False, log_every: int = 20, device=None) -> Path:
    """Huấn luyện một run, trả thư mục run. Không CUDA và allow_cpu=False -> RuntimeError."""
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError("Không có CUDA: train cần GPU (allow_cpu=True chỉ dành cho test).")
    device = torch.device(device or ("cuda" if has_cuda else "cpu"))
    allow_read("train", val_set_for(cfg))
    ff_cfg = ff_cfg if ff_cfg is not None else load_config(cfg.ffpp_config)

    d = prepare_run_dir(cfg, resume=resume, force=force)
    if resume and not (d / "last.pt").exists():
        raise FileNotFoundError(f"Không có last.pt để resume trong {d}")
    if not resume:
        write_run_info(d, cfg, {"split_file": cfg.split if cfg.mode == "compffd" else None})

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_frac * cfg.total_steps), cfg.total_steps)

    step, best = 0, None  # best = {"step", "auc_video", "bce"} | None
    if resume:
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
        model.load_trainable(ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        _set_rng_state(ck["rng"])
        step, best = int(ck["step"]), ck["best"]
        _truncate_jsonl(d / "val_curve.jsonl", step)
        _truncate_jsonl(d / "train_log.jsonl", step)

    ds = build_train_dataset(cfg, ff_cfg=ff_cfg)
    accum = cfg.grad_accum
    per_epoch = len(_epoch_batches(len(ds), cfg.batch_size, cfg.seed, 0))
    if per_epoch < accum:
        raise ValueError(f"Dữ liệu quá nhỏ: {per_epoch} batch/epoch < grad_accum {accum}")
    micro = step * accum  # số micro-batch đã dùng
    model.train()

    def do_val(s: int):
        nonlocal best
        m = _val_metrics(val_scores(model, cfg, ff_cfg, device=device))
        _append_jsonl(d / "val_curve.jsonl", {"step": s, **m})
        if is_better((m["auc_video"], m["bce"]), None if best is None else (best["auc_video"], best["bce"])):
            best = {"step": s, "auc_video": m["auc_video"], "bce": m["bce"]}
            _atomic_save({"trainable": model.trainable_state_dict(), **best}, d / "best.pt")

    def fail(s: int, vids: list, loss):
        err = {"step": s, "loss": float(loss), "video_ids": sorted(set(vids))}
        write_json_atomic(err, d / "error.json")
        raise FloatingPointError(f"Loss không hữu hạn ở step {s}: {float(loss)}; video_id={err['video_ids']}")

    while step < cfg.total_steps:
        epoch, off = divmod(micro, per_epoch)
        ds.set_epoch(epoch)
        batches = _epoch_batches(len(ds), cfg.batch_size, cfg.seed, epoch)[off:]
        loader = torch.utils.data.DataLoader(
            ds, batch_sampler=batches, num_workers=cfg.num_workers, persistent_workers=False,
            worker_init_fn=_worker_init(cfg.seed) if cfg.num_workers else None)
        acc_loss, acc_n, acc_vids = 0.0, 0, []
        for bi, batch in enumerate(loader):
            x = batch["image"].to(device)
            tg = {k: v.to(device) for k, v in batch["targets"].items() if torch.is_tensor(v)}
            acc_vids += [_video_of(ds, i) for i in batches[bi]]
            with _autocast(device):
                loss, _ = model.head.loss(model(x), tg)
            if not torch.isfinite(loss):
                fail(step + 1, acc_vids, loss.detach())
            (loss / accum).backward()
            acc_loss += float(loss.detach())
            acc_n += 1
            micro += 1
            if acc_n < accum:
                continue
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == cfg.total_steps:
                _append_jsonl(d / "train_log.jsonl", {"step": step, "loss": acc_loss / acc_n,
                                                      "lr": sched.get_last_lr()[0], "grad_norm": gn})
            acc_loss, acc_n, acc_vids = 0.0, 0, []
            last = step == cfg.total_steps
            if step % cfg.eval_every == 0 or last:
                do_val(step)
            if step % cfg.ckpt_every == 0 or last:
                save_last(d / "last.pt", {"trainable": model.trainable_state_dict(), "optimizer": opt.state_dict(),
                                          "scheduler": sched.state_dict(), "rng": _rng_state(),
                                          "step": step, "best": best})
            if last:
                break
    (d / "done.flag").write_text(f"step={step}\n", encoding="utf-8")
    return d
