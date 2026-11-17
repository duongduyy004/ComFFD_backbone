"""Chẩn đoán vì sao arm raw_aug thấp hơn raw. Không train lại; chỉ đọc kết quả có sẵn và chấm bằng best.pt.

Ghi `<ckpt_root>/diagnose/`:
- diagnose.md: tóm tắt 5 mục (in cả ra stdout);
- aug_scores.parquet: logit của raw và raw_aug trên mẫu ảnh rep/seam/real gốc cùng frame (train);
- low_rep_grid.png: 20 cặp [real gốc | rep] có r thấp và mask nhỏ nhất, để xem bằng mắt.

Mục:
1. AUC theo method trên FF++ test (từ result.json).
2. Phân bố logit theo lớp trên FF++ test và Celeb-DF (từ scores_*.parquet): trung vị, p90, tỉ lệ đoán fake (logit >= 0).
3. Điểm trên ảnh tăng cường (train): rep theo r và theo nhóm mask_frac_final; seam; real gốc cùng frame.
   rep có r thấp / mask nhỏ bị chấm gần như real -> dấu hiệu nhiễu nhãn.
4. Quá trình train: best_epoch, epochs_run, đường AUC val theo epoch (train_log.csv, done.flag).
5. Tách rep–seam và seam–real: AUC rep vs seam quá cao kể cả khi r thấp -> có thể học dấu vết ghép/nén.

Chỉ đọc ảnh train (manifest_aug, manifest_raw split train) để chấm; test/Celeb-DF chỉ đọc file điểm đã có.
Phân tích sau khi đã xem kết quả test: chỉ để hiểu nguyên nhân, không dùng để chỉnh tham số.
Trạng thái: chưa chạy thử trên GPU (chỉ test CPU với CLIP tí hon).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import tables
from .data import ffpp as D
from .evaluate import _cfg_from_info
from .guards import allow_read
from .io_utils import atomic_write_bytes, read_image, write_parquet_atomic
from .model import Detector
from .stats import auc
from .train import score_frames

ARMS = ("raw", "raw_aug")
R_BINS = (0.25, 0.5, 0.75, 1.0)


def _load_best(run: Path, data_root, device, vision_config=None):
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    cfg = _cfg_from_info(info, data_root)
    torch.manual_seed(cfg.seed)
    model = Detector(cfg, vision_config=vision_config).to(device)
    if model.trunk.commit != info.get("model_commit"):
        raise ValueError(f"commit model khác lúc train ({info.get('model_commit')} != {model.trunk.commit})")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_trainable(ck["trainable"])
    model.eval()
    return cfg, model


def _fake_rate(lg) -> float:
    lg = np.asarray(lg, float)
    return float(np.mean(lg >= 0)) if len(lg) else float("nan")


def _safe_auc(y, s) -> float:
    y = np.asarray(y)
    return auc(y, s) if len(np.unique(y)) == 2 else float("nan")


# ---------------------------------------------------------------- 1, 2, 4: đọc kết quả có sẵn
def per_method(ck: Path) -> str:
    rows = []
    res = {a: json.loads((ck / a / "result.json").read_text(encoding="utf-8")) for a in ARMS}
    for m in ("DF", "F2F", "FS", "NT", "FSh"):
        a, b = (res[x]["ffpp_test"].get("auc_per_method", {}).get(m) for x in ARMS)
        rows.append([m, tables.f4(a), tables.f4(b), tables.f4(None if a is None or b is None else b - a)])
    return tables.md(["method", "AUC raw", "AUC raw_aug", "diff"], rows)


def score_dist(ck: Path) -> str:
    rows = []
    for ds in ("ffpp_test", "celebdf"):
        for a in ARMS:
            p = ck / a / f"scores_{ds}.parquet"
            if not p.exists():
                rows.append(f"{ds} {a}: thiếu {p.name}")
                continue
            s = pd.read_parquet(p)
            for lab, name in ((0, "real"), (1, "fake")):
                lg = s.logit[s.label == lab].to_numpy(float)
                rows.append([ds, a, name, len(lg), tables.f4(np.median(lg)), tables.f4(np.quantile(lg, 0.9)),
                             tables.f4(_fake_rate(lg))])
    return tables.md(["dataset", "arm", "lớp", "n", "median logit", "p90 logit", "tỉ lệ đoán fake"], rows)


def train_curves(ck: Path) -> str:
    rows, curves = [], []
    for a in ARMS:
        done = json.loads((ck / a / "done.flag").read_text(encoding="utf-8"))
        log = pd.read_csv(ck / a / "train_log.csv")
        rows.append([a, done.get("best_epoch"), done.get("epochs_run"), done.get("total_steps"),
                     done.get("stopped_early"), tables.f4(log.train_loss.iloc[-1]),
                     tables.f4(log.val_auc_frame.max()), round(float(log.seconds.mean()) / 60, 1)])
        curves.append(f"- {a}: val AUC theo epoch = " + ", ".join(f"{v:.4f}" for v in log.val_auc_frame))
    t = tables.md(["arm", "best_epoch", "epochs_run", "total_steps", "stopped_early", "train_loss cuối",
                   "val AUC max", "phút/epoch"], rows)
    return t + "\n\n" + "\n".join(curves)


# ---------------------------------------------------------------- 3, 5: chấm ảnh tăng cường
def aug_sample(data_root, n_per_kind: int, seed: int = 0) -> pd.DataFrame:
    """Mẫu rep + seam (manifest_aug, có r, mask_frac_final) và ảnh real gốc cùng (tid, frame_idx) (manifest_raw train)."""
    allow_read("train", "ffpp_aug")
    root = D.ffpp_dir(data_root)
    aug = pd.read_parquet(root / "manifest_aug.parquet")
    aug["tid"] = aug.tid.astype(str)
    parts = []
    for kind in ("rep", "seam"):
        g = aug[aug.kind == kind]
        parts.append(g.sample(n=min(n_per_kind, len(g)), random_state=seed))
    smp = pd.concat(parts, ignore_index=True)
    allow_read("train", "ffpp_train")
    raw = pd.read_parquet(root / "manifest_raw.parquet")
    raw = raw[(raw.split == "train") & (raw.method == "real")].copy()
    raw["tid"] = raw.tid.astype(str)
    keys = smp[["tid", "frame_idx"]].drop_duplicates()
    real = raw.merge(keys, on=["tid", "frame_idx"])
    real = real.assign(kind="real", r=np.nan, mask_frac_final=np.nan)
    cols = ["path", "kind", "label", "tid", "frame_idx", "method", "r", "mask_frac_final"]
    out = pd.concat([smp[cols], real[cols]], ignore_index=True)
    out["path"] = out.path.map(lambda p: root / p)
    return out


def aug_tables(df: pd.DataFrame) -> str:
    """df có cột kind, r, mask_frac_final, logit_raw, logit_raw_aug."""
    out = []
    rows = []
    for kind in ("rep", "seam", "real"):
        g = df[df.kind == kind]
        rows.append([kind, len(g), *(tables.f4(_fake_rate(g[f"logit_{a}"])) for a in ARMS),
                     *(tables.f4(np.median(g[f"logit_{a}"])) for a in ARMS)])
    out.append("Tỉ lệ đoán fake và median logit theo loại ảnh (train):\n\n" + tables.md(
        ["loại", "n", "đoán fake raw", "đoán fake raw_aug", "median raw", "median raw_aug"], rows))

    rep = df[df.kind == "rep"]
    rows = []
    for r in R_BINS:
        g = rep[np.isclose(rep.r, r)]
        rows.append([r, len(g), *(tables.f4(_fake_rate(g[f"logit_{a}"])) for a in ARMS)])
    out.append("rep theo độ đậm r (đoán fake = mô hình nhận ra là fake):\n\n" + tables.md(
        ["r", "n", "đoán fake raw", "đoán fake raw_aug"], rows))

    if len(rep):
        q = rep.mask_frac_final.quantile([0.25, 0.5, 0.75]).to_numpy()
        edges = [-np.inf, *q, np.inf]
        rows = []
        for i in range(4):
            g = rep[(rep.mask_frac_final > edges[i]) & (rep.mask_frac_final <= edges[i + 1])]
            lab = f"Q{i + 1} (≤ {edges[i + 1]:.4f})" if i < 3 else f"Q4 (> {edges[3]:.4f})"
            rows.append([lab, len(g), *(tables.f4(_fake_rate(g[f"logit_{a}"])) for a in ARMS)])
        out.append("rep theo diện tích mask cuối (mask_frac_final, tứ phân vị):\n\n" + tables.md(
            ["nhóm", "n", "đoán fake raw", "đoán fake raw_aug"], rows))

    seam, real = df[df.kind == "seam"], df[df.kind == "real"]
    rows = []
    for name, pos, neg in (("rep vs seam", rep, seam), ("rep vs real gốc", rep, real), ("seam vs real gốc", seam, real)):
        for r in (None, 0.25):
            p = pos if r is None or name == "seam vs real gốc" else pos[np.isclose(pos.r, r)]
            if r is not None and name == "seam vs real gốc":
                continue
            y = np.r_[np.ones(len(p)), np.zeros(len(neg))]
            rows.append([name + ("" if r is None else f" (rep r={r})"),
                         *(tables.f4(_safe_auc(y, np.r_[p[f"logit_{a}"], neg[f"logit_{a}"]])) for a in ARMS)])
    out.append("Độ tách (AUC; seam vs real gốc ≈ 0.5 nghĩa là mô hình không coi đường nối là fake):\n\n" + tables.md(
        ["cặp", "AUC raw", "AUC raw_aug"], rows))
    return "\n\n".join(out)


def low_rep_grid(df: pd.DataFrame, path: Path, n: int = 20, tile: int = 224) -> int:
    """n cặp [real gốc | rep] có r thấp nhất rồi mask nhỏ nhất; ghi PNG lưới 4 cặp mỗi hàng."""
    rep = df[df.kind == "rep"].sort_values(["r", "mask_frac_final"]).head(n)
    real = {(t, int(f)): p for t, f, p in df[df.kind == "real"][["tid", "frame_idx", "path"]].itertuples(index=False)}
    pairs = []
    for row in rep.itertuples():
        key = (row.tid, int(row.frame_idx))
        if key not in real:
            continue
        a = cv2.resize(read_image(Path(real[key])), (tile, tile))
        b = cv2.resize(read_image(Path(row.path)), (tile, tile))
        cv2.putText(b, f"r={row.r:g} m={row.mask_frac_final:.3f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        pairs.append(np.hstack([a, b]))
    if not pairs:
        return 0
    per = 4
    rows = []
    for i in range(0, len(pairs), per):
        chunk = pairs[i:i + per]
        chunk += [np.zeros_like(pairs[0])] * (per - len(chunk))
        rows.append(np.hstack(chunk))
    ok, buf = cv2.imencode(".png", np.vstack(rows))
    if not ok:
        raise IOError(f"Không mã hóa được {path}")
    atomic_write_bytes(path, buf.tobytes())
    return len(pairs)


# ---------------------------------------------------------------- điều phối
def diagnose(ckpt_root, *, data_root=None, n_per_kind: int = 2000, batch_size: int = 128, num_workers: int = 0,
             device=None, vision_config=None) -> Path:
    ck = Path(ckpt_root)
    for a in ARMS:
        for n in ("result.json", "best.pt", "run_info.json", "done.flag", "train_log.csv"):
            if not (ck / a / n).exists():
                raise FileNotFoundError(f"Thiếu {ck / a / n}: chạy `main` xong trước khi chẩn đoán")
    out = ck / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    info = json.loads((ck / "raw_aug" / "run_info.json").read_text(encoding="utf-8"))
    df = aug_sample(_cfg_from_info(info, data_root).data_root, n_per_kind)
    for a in ARMS:
        _, model = _load_best(ck / a, data_root, device, vision_config)
        df[f"logit_{a}"] = score_frames(model, df, batch_size, device, num_workers, f"diagnose {a}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_parquet_atomic(df.assign(path=df.path.map(str)), out / "aug_scores.parquet")
    n_grid = low_rep_grid(df, out / "low_rep_grid.png")

    parts = [
        "# Chẩn đoán raw_aug so với raw",
        "Phân tích sau khi đã xem kết quả test; chỉ để hiểu nguyên nhân, không dùng để chỉnh tham số. "
        "Ngưỡng đoán fake: logit ≥ 0.",
        "## 1. AUC theo method (FF++ test)\n\n" + per_method(ck),
        "## 2. Phân bố logit theo lớp\n\n" + score_dist(ck),
        f"## 3 và 5. Điểm trên ảnh tăng cường (train, mẫu ≤ {n_per_kind} mỗi loại)\n\n" + aug_tables(df),
        "## 4. Quá trình train\n\n" + train_curves(ck),
        f"## Ảnh xem tay\n\n`low_rep_grid.png`: {n_grid} cặp [real gốc | rep], r thấp và mask nhỏ nhất.",
    ]
    text = "\n\n".join(parts) + "\n"
    atomic_write_bytes(out / "diagnose.md", text.encode("utf-8"))
    print(text, flush=True)
    return out / "diagnose.md"
