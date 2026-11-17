"""Bootstrap theo VIDEO cho so sánh arm trên một dataset (mặc định celebdf). Chỉ đọc scores_<ds>.parquet.

Dùng: py scripts/video_bootstrap.py --ckpt-root <results> --base raw --arms raw_aug raw_strongaug [--ds celebdf]
In: AUC frame và AUC video (điểm video = logit trung bình) của từng arm, diff (arm - base) kèm CI95 bootstrap
resample VIDEO (B_frame=500, B_video=2000, seed 0), tỉ lệ real đoán fake (logit >= 0), AUC video theo nguồn real.
Khác `compare` (bootstrap theo frame, lạc quan vì frame cùng video tương quan). Trạng thái: chưa chạy thử trên GPU box.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _load(root: Path, arm: str, ds: str) -> pd.DataFrame:
    return pd.read_parquet(root / arm / f"scores_{ds}.parquet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--base", default="raw")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ds", default="celebdf")
    ap.add_argument("--b-frame", type=int, default=500)
    ap.add_argument("--b-video", type=int, default=2000)
    a = ap.parse_args(argv)
    root, arms = Path(a.ckpt_root), [a.base, *a.arms]
    d = {m: _load(root, m, a.ds) for m in arms}
    b = d[a.base][["video_id", "frame_idx", "label"]].copy()
    for m in arms:
        if not (d[m].video_id.values == b.video_id.values).all() or not (d[m].frame_idx.values == b.frame_idx.values).all():
            raise SystemExit(f"{m}: thứ tự/tập frame khác {a.base}")
        b[m] = d[m].logit.values
    rng = np.random.default_rng(0)
    grp = b.groupby("video_id").indices
    vids = np.array(list(grp))
    lab = b.label.values
    vl = b.groupby("video_id").agg(label=("label", "first"), **{m: (m, "mean") for m in arms})
    real = b[b.label == 0]
    print(f"{a.ds}: {len(b)} frame, {len(vids)} video")
    print("arm | AUC frame | AUC video | real đoán fake (logit>=0) | AUC video theo nguồn real")
    for m in arms:
        src = {}
        for s in sorted({v.split('__')[0] for v in vl.index if v.split('__')[0] and vl.loc[v, "label"] == 0}):
            v = vl[(vl.index.str.startswith(s)) | (vl.label == 1)]
            src[s] = round(roc_auc_score(v.label, v[m]), 4)
        print(f"{m} | {roc_auc_score(lab, b[m]):.4f} | {roc_auc_score(vl.label, vl[m]):.4f} | "
              f"{(real[m] >= 0).mean():.3f} | {src}")
    sf = [rng.choice(vids, len(vids)) for _ in range(a.b_frame)]
    sv = [rng.integers(0, len(vl), len(vl)) for _ in range(a.b_video)]

    def fd(sel, m):
        idx = np.concatenate([grp[v] for v in sel])
        return roc_auc_score(lab[idx], b[m].values[idx]) - roc_auc_score(lab[idx], b[a.base].values[idx])

    print(f"\ndiff so với {a.base} (bootstrap theo video):")
    for m in a.arms:
        xf = np.array([fd(s, m) for s in sf])
        xv = []
        for s in sv:
            v = vl.iloc[s]
            if v.label.nunique() == 2:
                xv.append(roc_auc_score(v.label, v[m]) - roc_auc_score(v.label, v[a.base]))
        pf = roc_auc_score(lab, b[m]) - roc_auc_score(lab, b[a.base])
        pv = roc_auc_score(vl.label, vl[m]) - roc_auc_score(vl.label, vl[a.base])
        print(f"{m}: AUC frame {pf:+.4f} CI95 [{np.quantile(xf, .025):+.4f}, {np.quantile(xf, .975):+.4f}] | "
              f"AUC video {pv:+.4f} CI95 [{np.quantile(xv, .025):+.4f}, {np.quantile(xv, .975):+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
