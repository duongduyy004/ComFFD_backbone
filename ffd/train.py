"""Vòng train: val định kỳ, chọn best.pt theo (AUC video, BCE), last.pt + resume, chặn NaN.

Quy ước:
- 1 "step" = 1 lần cập nhật optimizer (gồm `grad_accum` micro-batch). Mỗi micro-batch đúng batch_size/2 real
  + batch_size/2 fake (`_epoch_batches`, batch_size phải chẵn); thứ tự xáo bằng Generator(seed, epoch),
  drop_last; vị trí trong epoch suy ra từ step nên resume không cần đọc lại dữ liệu đã qua. `dataset.set_epoch(e)` gọi ở tiến trình chính mỗi đầu epoch, DataLoader
  `persistent_workers=False` nên worker tạo lại mỗi epoch và thấy epoch mới.
- Thứ tự trong một step: cập nhật -> train_log -> val (nếu eval_every) -> last.pt (nếu ckpt_every).
  Step cuối luôn val và lưu last.pt.
- Cần CUDA: không có CUDA và `allow_cpu=False` -> RuntimeError. `allow_cpu=True` chỉ dùng trong test
  (CPU, không autocast). `ff_cfg` cho phép test truyền Config tổng hợp thay vì nạp từ cfg.ffpp_config.
- Val: arm -> FFppRawCore(val); compffd -> EvalCore(val) lọc real + ô seen của split (cfg.split là TÊN,
  file là <out_root>/splits/<split>.json qua `config.split_path`). video_id compffd: "{tid}|real" hoặc
  "{tid}|{region}|{method}". `allow_read("train", ...)` gọi ngay trước khi mở từng tập.
- grad_accum: một step luôn gồm đúng `grad_accum` micro-batch, kể cả khi vượt biên epoch (bộ tích lũy nằm
  ngoài vòng epoch) nên vị trí resume = step * grad_accum micro-batch là chính xác.
- Resume: best.pt mồ côi (step > last.step) được ghi lại từ `best_state` trong last.pt (hoặc xóa nếu chưa có best).
- AUC val không hữu hạn (một lớp, NaN) -> ValueError, không rơi xuống so sánh BCE.
- Chọn best: AUC video cao hơn hẳn (> tol); hòa trong tol thì BCE thấp hơn.
Trạng thái: chưa chạy thử (chỉ test CPU với CLIP tí hon ngẫu nhiên).
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from compsynth_ff.config import load_config
from compsynth_ff.io_utils import write_json_atomic

from . import progress
from .config import TrainConfig, split_path
from .data.arms import build_train_dataset
from .data.transforms import eval_transform
from .guards import allow_read
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
def _val_items(cfg: TrainConfig, ff_cfg, subset: str = "val"):
    """Trả (core, rows); rows = list (index trong core, video_id, frame_idx, label)."""
    from compsynth_ff import dataset as D

    allow_read("train", f"{'compsynth' if cfg.mode == 'compffd' else 'ffpp'}_{subset}")
    if cfg.mode != "compffd":
        core = D.FFppRawCore(ff_cfg, subset)
        rows = [(i, f"{t}_{s}", int(f), 0 if s == "real" else 1) for i, (t, f, s) in enumerate(core.items)]
        return core, rows
    sp = json.loads(split_path(cfg, ff_cfg).read_text(encoding="utf-8"))
    seen = {tuple(x) for x in sp["seen"]}
    core = D.EvalCore(ff_cfg, subset)
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
def val_scores(model, cfg: TrainConfig, ff_cfg, batch_size: int | None = None, device=None,
               subset: str = "val") -> pd.DataFrame:
    """[video_id, frame_idx, label, logit]; model trở về chế độ cũ (train/eval) sau khi tính."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    bs = batch_size or cfg.batch_size
    core, rows = _val_items(cfg, ff_cfg, subset)
    was = model.training
    model.eval()
    logits = []
    try:
        for s in progress.bar(range(0, len(rows), bs), desc=f"val {subset}", unit="batch", leave=False):
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
    if df.label.nunique() < 2:
        raise ValueError("Val chỉ có một lớp: không tính được AUC")
    if not np.isfinite(df.logit.to_numpy(np.float64)).all():
        raise ValueError("Val có logit không hữu hạn")
    v = video_scores(df)
    m = {"auc_video": auc(v.label, v.score), "auc_frame": auc(df.label, df.logit), "bce": _bce(df)}
    if not all(np.isfinite(x) for x in m.values()):
        raise ValueError(f"Metric val không hữu hạn: {m}")
    return m


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


def _seed_worker(worker_id: int, seed: int) -> None:
    """worker_init_fn cấp module (pickle được khi Windows dùng spawn); bọc bằng functools.partial(seed=...)."""
    s = (seed * 1_000_003 + worker_id) % (2**32)
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def _epoch_batches(n: int, bs: int, seed: int, epoch: int) -> list[list[int]]:
    """Batch cân bằng: mỗi batch đúng bs/2 mục real (chỉ số chẵn) + bs/2 mục fake (chỉ số lẻ).

    Xáo riêng danh sách chẵn và lẻ bằng cùng Generator(seed, epoch), ghép từng nửa, drop_last theo danh sách
    ngắn hơn, rồi xáo thứ tự trong batch. Tất định theo (seed, epoch) nên resume suy ra vị trí từ step.
    """
    if bs < 2 or bs % 2:
        raise ValueError(f"batch_size phải chẵn và >= 2 (nửa real, nửa fake), nhận {bs}")
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    real = torch.arange(0, n, 2)[torch.randperm((n + 1) // 2, generator=g)]
    fake = torch.arange(1, n, 2)[torch.randperm(n // 2, generator=g)]
    h = bs // 2
    out = []
    for b in range(min(len(real), len(fake)) // h):  # drop_last
        idx = torch.cat([real[b * h:(b + 1) * h], fake[b * h:(b + 1) * h]])
        out.append(idx[torch.randperm(bs, generator=g)].tolist())
    return out


def _video_of(ds, idx: int) -> str:
    """tid của video mà mục idx lấy từ (real ở idx chẵn; fake lẻ của compffd lấy từ khe fake)."""
    c, k = ds.core, idx // 2
    if idx % 2 == 1 and hasattr(c, "fake_slots"):
        return str(c.fake_slots[k % len(c.fake_slots)][0])
    return str(c.real_frames[k % len(c.real_frames)][0])


def _reconcile_best(path: Path, step: int, best, best_state) -> None:
    """Sau resume, best.pt phải khớp `best` trong last.pt. best.pt ghi ở step > last.step (mồ côi) bị bỏ;
    nếu last.pt có best_state thì ghi lại best.pt từ đó, không thì xóa (chưa có best)."""
    on_disk = None
    if path.exists():
        on_disk = int(torch.load(path, map_location="cpu", weights_only=False)["step"])
    want = None if best is None else int(best["step"])
    if on_disk == want:
        return
    path.unlink(missing_ok=True)
    if best is not None and best_state is not None:
        _atomic_save({"trainable": best_state, **best}, path)


# ---------------------------------------------------------------- train
def train(cfg: TrainConfig, *, resume: bool = False, force: bool = False, vision_config=None,
          ff_cfg=None, allow_cpu: bool = False, log_every: int = 20, device=None) -> Path:
    """Huấn luyện một run, trả thư mục run. Không CUDA và allow_cpu=False -> RuntimeError."""
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError("Không có CUDA: train cần GPU (allow_cpu=True chỉ dành cho test).")
    device = torch.device(device or ("cuda" if has_cuda else "cpu"))
    ff_cfg = ff_cfg if ff_cfg is not None else load_config(cfg.ffpp_config)
    allow_read("train", "compsynth_train" if cfg.mode == "compffd" else "ffpp_train")
    sp_file = split_path(cfg, ff_cfg) if cfg.mode == "compffd" else None  # lỗi sớm nếu thiếu split

    d = prepare_run_dir(cfg, resume=resume, force=force)
    if resume and not (d / "last.pt").exists():
        raise FileNotFoundError(f"Không có last.pt để resume trong {d}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    commit = model.trunk.commit
    if not resume:  # ghi sau khi dựng model để có commit HF đã resolve
        write_run_info(d, cfg, {"split_file": sp_file, "model_commit": commit}, ff_cfg=ff_cfg)
    else:
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8")) if (d / "run_info.json").exists() else {}
        if info.get("model_commit") != commit:
            raise ValueError(f"model_commit khác run cũ ({info.get('model_commit')} != {commit}); không resume")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_frac * cfg.total_steps), cfg.total_steps)

    step, best, best_state = 0, None, None  # best = {"step", "auc_video", "bce"} | None
    if resume:
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
        model.load_trainable(ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        _set_rng_state(ck["rng"])
        step, best, best_state = int(ck["step"]), ck["best"], ck.get("best_state")
        _reconcile_best(d / "best.pt", step, best, best_state)
        _truncate_jsonl(d / "val_curve.jsonl", step)
        _truncate_jsonl(d / "train_log.jsonl", step)

    ds = build_train_dataset(cfg, ff_cfg=ff_cfg)
    accum = cfg.grad_accum
    per_epoch = len(_epoch_batches(len(ds), cfg.batch_size, cfg.seed, 0))
    if per_epoch < 1:
        raise ValueError("Dữ liệu quá nhỏ: không đủ một batch/epoch")
    micro = step * accum  # số micro-batch đã dùng
    model.train()

    def do_val(s: int):
        nonlocal best, best_state
        m = _val_metrics(val_scores(model, cfg, ff_cfg, device=device))
        _append_jsonl(d / "val_curve.jsonl", {"step": s, **m})
        if is_better((m["auc_video"], m["bce"]), None if best is None else (best["auc_video"], best["bce"])):
            best = {"step": s, "auc_video": m["auc_video"], "bce": m["bce"]}
            best_state = model.trainable_state_dict()
            _atomic_save({"trainable": best_state, **best}, d / "best.pt")

    def fail(s: int, vids: list, loss):
        err = {"step": s, "loss": float(loss), "video_ids": sorted(set(vids))}
        write_json_atomic(err, d / "error.json")
        pb.close()
        raise FloatingPointError(f"Loss không hữu hạn ở step {s}: {float(loss)}; video_id={err['video_ids']}")

    acc_loss, acc_n, acc_vids = 0.0, 0, []  # ngoài vòng epoch: một step luôn đủ `accum` micro-batch
    run_name = f"compffd/{cfg.split}" if cfg.mode == "compffd" else cfg.mode
    pb = progress.bar(total=cfg.total_steps, initial=step, desc=f"train {run_name} seed{cfg.seed}", unit="step")
    if best is not None:
        pb.set_postfix(best_auc=f"{best['auc_video']:.4f}")
    while step < cfg.total_steps:
        epoch, off = divmod(micro, per_epoch)
        ds.set_epoch(epoch)
        batches = _epoch_batches(len(ds), cfg.batch_size, cfg.seed, epoch)[off:]
        loader = torch.utils.data.DataLoader(
            ds, batch_sampler=batches, num_workers=cfg.num_workers, persistent_workers=False,
            worker_init_fn=functools.partial(_seed_worker, seed=cfg.seed) if cfg.num_workers else None)
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
            pb.update(1)
            if step % log_every == 0 or step == cfg.total_steps:
                pb.set_postfix(loss=f"{acc_loss / acc_n:.4f}",
                               best_auc="-" if best is None else f"{best['auc_video']:.4f}")
                _append_jsonl(d / "train_log.jsonl", {"step": step, "loss": acc_loss / acc_n,
                                                      "lr": sched.get_last_lr()[0], "grad_norm": gn})
            acc_loss, acc_n, acc_vids = 0.0, 0, []
            last = step == cfg.total_steps
            if step % cfg.eval_every == 0 or last:
                do_val(step)
            if step % cfg.ckpt_every == 0 or last:
                save_last(d / "last.pt", {"trainable": model.trainable_state_dict(), "optimizer": opt.state_dict(),
                                          "scheduler": sched.state_dict(), "rng": _rng_state(),
                                          "step": step, "best": best, "best_state": best_state})
            if last:
                break
    pb.close()
    (d / "done.flag").write_text(f"step={step}\n", encoding="utf-8")
    return d
