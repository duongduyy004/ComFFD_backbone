"""Chỉ số mức frame cho một tập đánh giá. Ngưỡng logit >= 0 (xác suất 0.5), lớp dương = fake (1).
Module không import torch. Trạng thái: chưa chạy thử ngoài test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from . import stats


def frame_block(df: pd.DataFrame, n_videos_dropped: int) -> dict:
    """df: cột video_id, label, logit. Trả dict thuần kiểu Python (JSON được)."""
    y = df["label"].to_numpy().astype(int)
    s = df["logit"].to_numpy().astype(float)
    pred = (s >= 0).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    a, lo, hi = stats.bootstrap_ci(y, s, groups=np.arange(len(y)), n=2000, seed=0)
    vs = stats.video_scores(df)
    return {
        "auc": stats.auc(y, s),
        "acc": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "confusion_matrix": [[int(v) for v in row] for row in cm],
        "n_frames": int(len(df)),
        "n_videos": int(df["video_id"].nunique()),
        "n_videos_dropped": int(n_videos_dropped),
        "auc_video": stats.auc(vs["label"], vs["score"]),
        "auc_ci95": [float(lo), float(hi)],
    }


def auc_per_method(df: pd.DataFrame) -> dict[str, float]:
    """AUC từng method fake so với mọi frame real (method == "real")."""
    real = df[df["method"] == "real"]
    out = {}
    for m in sorted(set(df.loc[df["label"] == 1, "method"])):
        sub = pd.concat([real, df[df["method"] == m]])
        out[str(m)] = stats.auc(sub["label"], sub["logit"])
    return out
