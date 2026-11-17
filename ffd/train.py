"""Vòng train theo epoch, early stopping, best.pt / last.pt / train_log.csv trong checkpoints/clip/<run>.

Quy ước:
- Epoch đánh số từ 1. Mỗi epoch đi qua mọi ảnh của TrainList đúng một lần theo
  `torch.randperm(N, Generator().manual_seed(seed * 100_003 + epoch))`, cắt thành batch `batch_size`, batch cuối
  nhỏ vẫn dùng (drop_last=False). Không class weight, không sampler cân bằng. `ds.set_epoch(epoch)` gọi ở tiến
  trình chính trước khi dựng DataLoader (persistent_workers=False nên worker thấy epoch mới).
- AdamW + warmup tuyến tính (`warmup_frac`) + cosine trên `max_epochs * ceil(N / batch_size)` bước; clip grad norm 1.0.
- Cuối mỗi epoch: val FF++ (frame, eval_transform) -> auc_frame, bce, acc, precision, recall, f1 (ngưỡng logit >= 0,
  `metrics.threshold_metrics`; chỉ để theo dõi, không dùng chọn best) -> cập nhật best -> một dòng train_log.csv
  -> last.pt. Best theo (AUC frame, BCE): AUC hơn > 1e-4 thắng, hòa trong 1e-4 thì BCE thấp hơn.
  `bad_epochs` tăng khi AUC frame không vượt best cũ quá 1e-4 (hòa mà BCE tốt hơn vẫn cập nhật best.pt nhưng
  vẫn tính là epoch xấu); dừng khi `bad_epochs >= patience` hoặc hết `max_epochs`. `stopped_early` chỉ True khi
  dừng vì patience TRƯỚC max_epochs (đủ patience đúng ở epoch cuối -> False).
- Resume chỉ ở biên epoch, từ last.pt; best.pt và train_log.csv được đưa về khớp last.pt (bỏ phần ghi sau nó).
  train_log.csv cũ (11 cột, trước 16/11) được ghi lại với header LOG_COLS mới, ô thiếu để trống.
- Cột `seconds` = thời gian thực của epoch (train + val). `frac_real` = tỉ lệ ảnh nhãn 0 trong epoch;
  `frac_rep`/`frac_seam` = tỉ lệ ảnh nguồn rep/seam; `frac_jpeg` = tỉ lệ ảnh đã nén JPEG.
- Cần CUDA: không có CUDA và `allow_cpu=False` -> RuntimeError. bf16 autocast trên CUDA.
Arm loc: loss tối ưu = BCE_cls + loc_lambda * L_loc, nhưng `train_loss` luôn log BCE_cls (như các arm khác); cột `train_loc_loss` = L_loc trung bình theo ảnh có giám sát (arm khác
  để trống). Resume với hằng số arm khác run_info -> ValueError.
Trạng thái: chưa chạy thử trên GPU (chỉ test CPU với CLIP tí hon ngẫu nhiên).
"""
from __future__ import annotations

import contextlib
import csv
import functools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metrics, progress
from .config import FILTER_ARMS, LOC_ARMS, TrainConfig, run_name_of
from .data.augfilter import arm_constants, check_arm_constants
from .data.ffpp import frames
from .data.trainset import TrainList
from .data.transforms import eval_transform
from .guards import allow_read
from .io_utils import read_image, write_json_atomic
from .heads.loc import loc_loss
from .model import Detector
from .rundir import prepare_run_dir, write_run_info
from .stats import auc

LOG_COLS = ["epoch", "step", "train_loss", "lr", "val_auc_frame", "val_bce",
            "val_acc", "val_precision", "val_recall", "val_f1",
            "frac_real", "frac_rep", "frac_seam", "frac_jpeg", "seconds", "train_loc_loss"]
TOL = 1e-4


def is_better(new, best, tol: float = TOL) -> bool:
    """new/best = (AUC frame, BCE). AUC hơn > tol thắng; |ΔAUC| <= tol thì BCE thấp hơn thắng."""
    if best is None:
        return True
    if new[0] > best[0] + tol:
        return True
    if new[0] < best[0] - tol:
        return False
    return new[1] < best[1]


def epoch_batches(n: int, bs: int, seed: int, epoch: int) -> list[list[int]]:
    """Hoán vị range(n) theo (seed, epoch), cắt thành batch bs; batch cuối nhỏ giữ nguyên."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100_003 + epoch)).tolist()
    return [perm[i:i + bs] for i in range(0, n, bs)]


# ---------------------------------------------------------------- DataLoader
def loader_kwargs(num_workers: int, device, seed: int = 0) -> dict:
    """Tham số DataLoader dùng chung cho train/val/eval. Chỉ ảnh hưởng tốc độ, không đổi ảnh hay thứ tự:
    pin_memory khi CUDA, prefetch 4 batch/worker, worker_init_fn seed + OpenCV 1 luồng."""
    kw = {"num_workers": num_workers, "pin_memory": torch.device(device).type == "cuda", "persistent_workers": False}
    if num_workers:
        kw.update(prefetch_factor=4, worker_init_fn=functools.partial(_seed_worker, seed=seed))
    return kw


def _seed_worker(worker_id: int, seed: int) -> None:
    """worker_init_fn cấp module (pickle được khi Windows dùng spawn). OpenCV mỗi worker 1 luồng."""
    import cv2

    cv2.setNumThreads(1)
    s = (seed * 1_000_003 + worker_id) % (2**32)
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def _autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class EvalImages(torch.utils.data.Dataset):
    """eval_transform(read_image(path)) theo đúng thứ tự paths (cấp module để pickle được với spawn)."""

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, j: int) -> torch.Tensor:
        return eval_transform(read_image(self.paths[j]))


@torch.no_grad()
def score_frames(model, df: pd.DataFrame, bs: int, device, num_workers: int = 0, desc: str = "eval") -> np.ndarray:
    """Logit float32 cho từng dòng df (cột path), cùng thứ tự df. Model ở eval trong lúc chấm, trả lại chế độ cũ."""
    loader = torch.utils.data.DataLoader(EvalImages(df.path), batch_size=bs, shuffle=False,
                                         **loader_kwargs(num_workers, device))
    was = model.training
    model.eval()
    out = []
    try:
        for x in progress.bar(loader, desc=desc, unit="batch", leave=False):
            with _autocast(device):
                out.append(model(x.to(device, non_blocking=True))["logit"].float().cpu())
    finally:
        model.train(was)
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


# ---------------------------------------------------------------- val
def _bce(logit: np.ndarray, y: np.ndarray) -> float:
    z, y = logit.astype(np.float64), y.astype(np.float64)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def _validate(model, val_df: pd.DataFrame, cfg: TrainConfig, device) -> dict:
    """{"auc_frame", "bce", "acc", "precision", "recall", "f1"} trên FF++ val. Một lớp hoặc logit/metric không hữu hạn -> ValueError."""
    y = val_df.label.to_numpy(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Val chỉ có một lớp: không tính được AUC")
    lg = score_frames(model, val_df, cfg.batch_size, device, cfg.num_workers, "val")
    if not np.isfinite(lg).all():
        raise ValueError("Val có logit không hữu hạn")
    m = {"auc_frame": float(auc(y, lg)), "bce": _bce(lg, y), **metrics.threshold_metrics(y, lg)}
    if not all(np.isfinite(v) for v in m.values()):
        raise ValueError(f"Metric val không hữu hạn: {m}")
    return m


# ---------------------------------------------------------------- checkpoint / log
def _atomic_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _append_log(path: Path, row: dict) -> None:
    """Ghi một dòng theo LOG_COLS; cột thiếu trong row (log cũ) để trống."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, restval="")
        if new:
            w.writeheader()
        w.writerow(row)


def _truncate_log(path: Path, epoch: int) -> None:
    """Bỏ các dòng ghi sau last.pt (epoch > epoch đã lưu); ghi lại với header LOG_COLS (log cũ thiếu cột -> ô trống)."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= epoch]
    path.unlink()
    for r in rows:
        _append_log(path, r)


def _reconcile_best(path: Path, best, best_state) -> None:
    """best.pt phải khớp `best` trong last.pt: best.pt ghi sau last.pt (mồ côi) được ghi lại từ best_state."""
    on_disk = None
    if path.exists():
        on_disk = int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
    want = None if best is None else int(best["epoch"])
    if on_disk == want:
        return
    path.unlink(missing_ok=True)
    if best is not None:
        _atomic_save({"trainable": best_state, **best}, path)


# ---------------------------------------------------------------- train
def train(cfg: TrainConfig, *, resume: bool = False, force: bool = False, vision_config=None,
          allow_cpu: bool = False, device=None) -> Path:
    """Huấn luyện một run, trả thư mục run. Không CUDA và allow_cpu=False -> RuntimeError."""
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError("Không có CUDA: train cần GPU (allow_cpu=True chỉ dành cho test).")
    device = torch.device(device or ("cuda" if has_cuda else "cpu"))

    d = prepare_run_dir(cfg, resume=resume, force=force)
    if resume and not (d / "last.pt").exists():
        raise FileNotFoundError(f"Không có last.pt để resume trong {d}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    commit = model.trunk.commit
    consts = arm_constants(cfg.arm)
    if resume:
        p = d / "run_info.json"
        info = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if info.get("model_commit") != commit:
            raise ValueError(f"model_commit khác run cũ ({info.get('model_commit')} != {commit}); không resume")
        check_arm_constants(cfg.arm, info, "không resume")

    ds = TrainList(cfg.data_root, cfg.arm, cfg.seed, cfg.p_jpeg, cfg.jpeg_q, grid=model.grid)  # tự gọi allow_read train/aug
    if not resume:  # ghi sau khi dựng model (commit HF đã resolve) và TrainList (aug_stats)
        extra = {"model_commit": commit}
        if cfg.arm in FILTER_ARMS:  # arm cũ: run_info giữ nguyên cấu trúc
            extra.update(arm_constants=consts, aug_stats=ds.aug_stats)
        elif consts:  # raw_strongaug: chỉ có hằng số suy giảm, không có aug_stats
            extra.update(arm_constants=consts)
        write_run_info(d, cfg, extra)
    use_loc = cfg.arm in LOC_ARMS
    allow_read("train", "ffpp_val")
    val_df = frames(cfg.data_root, "ffpp", "val")
    labels = ds.items.label.to_numpy(int)
    sources = ds.items.source.to_numpy(str)
    vids = ds.items.video_id.astype(str).to_numpy()
    n = len(ds)
    if n == 0:
        raise ValueError("Tập train rỗng")
    per_epoch = math.ceil(n / cfg.batch_size)
    total = cfg.max_epochs * per_epoch

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)

    epoch, step, bad = 0, 0, 0
    best, best_state = None, None  # best = {"epoch", "step", "auc_frame", "bce"}
    if resume:
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
        model.load_trainable(ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        _set_rng_state(ck["rng"])
        epoch, step, bad = int(ck["epoch"]), int(ck["step"]), int(ck["bad_epochs"])
        best, best_state = ck["best"], ck["best_state"]
        _reconcile_best(d / "best.pt", best, best_state)
        _truncate_log(d / "train_log.csv", epoch)

    name = run_name_of(cfg)
    model.train()
    while epoch < cfg.max_epochs and bad < cfg.patience:
        epoch += 1
        t0 = time.perf_counter()
        ds.set_epoch(epoch)
        batches = epoch_batches(n, cfg.batch_size, cfg.seed, epoch)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches,
                                             **loader_kwargs(cfg.num_workers, device, cfg.seed))
        loss_sum, n_jpeg = 0.0, 0  # loss_sum: chỉ BCE_cls (không gồm λ·L_loc)
        loc_sum, n_sup = 0.0, 0  # L_loc theo ảnh có giám sát (không theo batch)
        pb = progress.bar(loader, total=len(batches), desc=f"train {name} e{epoch}/{cfg.max_epochs}", unit="batch")
        for bi, batch in enumerate(pb):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            with _autocast(device):
                out = model(x)
                loss, _ = model.head.loss(out, {"L0": y})
                cls_loss = loss
                if use_loc:
                    has = batch["has_loc"].to(device)
                    l_loc = loc_loss(out["patch_logit"], batch["patch_target"].to(device), has)
                    loss = loss + consts["loc_lambda"] * l_loc
            if not torch.isfinite(loss):
                err = {"epoch": epoch, "step": step + 1, "loss": float(loss.detach()),
                       "video_ids": sorted(set(vids[batches[bi]].tolist()))}
                write_json_atomic(err, d / "error.json")
                pb.close()
                raise FloatingPointError(f"Loss không hữu hạn ở epoch {epoch} step {step + 1}; "
                                         f"video_id={err['video_ids']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss_sum += float(cls_loss.detach()) * len(x)
            if use_loc:
                k = int(has.sum())
                loc_sum += float(l_loc.detach()) * k
                n_sup += k
            n_jpeg += int(batch["jpeg"].sum())
            pb.set_postfix(loss=f"{float(cls_loss.detach()):.4f}")
        pb.close()

        m = _validate(model, val_df, cfg, device)
        old_auc = None if best is None else best["auc_frame"]
        if is_better((m["auc_frame"], m["bce"]), None if best is None else (best["auc_frame"], best["bce"])):
            best = {"epoch": epoch, "step": step, "auc_frame": m["auc_frame"], "bce": m["bce"]}
            best_state = model.trainable_state_dict()
            _atomic_save({"trainable": best_state, **best}, d / "best.pt")
        bad = 0 if old_auc is None or m["auc_frame"] > old_auc + TOL else bad + 1
        _append_log(d / "train_log.csv", {
            "epoch": epoch, "step": step, "train_loss": loss_sum / n, "lr": sched.get_last_lr()[0],
            "val_auc_frame": m["auc_frame"], "val_bce": m["bce"], "val_acc": m["acc"],
            "val_precision": m["precision"], "val_recall": m["recall"], "val_f1": m["f1"],
            "frac_real": float(np.mean(labels == 0)), "frac_rep": float(np.mean(sources == "rep")),
            "frac_seam": float(np.mean(sources == "seam")), "frac_jpeg": n_jpeg / n,
            "seconds": time.perf_counter() - t0,
            **({"train_loc_loss": loc_sum / n_sup} if use_loc and n_sup else {})})
        _atomic_save({"trainable": model.trainable_state_dict(), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "rng": _rng_state(), "epoch": epoch, "step": step,
                      "best": best, "best_state": best_state, "bad_epochs": bad}, d / "last.pt")
        print(f"[{name}] epoch {epoch}: loss {loss_sum / n:.4f} | "
              f"val ACC {m['acc']:.4f} AUC {m['auc_frame']:.4f} "
              f"PREC {m['precision']:.4f} REC {m['recall']:.4f} F1 {m['f1']:.4f} BCE {m['bce']:.4f} | "
              f"best e{best['epoch']}, bad {bad}/{cfg.patience}", file=sys.stderr, flush=True)

    write_json_atomic({"epochs_run": epoch, "best_epoch": None if best is None else best["epoch"],
                       "stopped_early": bad >= cfg.patience and epoch < cfg.max_epochs, "total_steps": step}, d / "done.flag")
    return d
