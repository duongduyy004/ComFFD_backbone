"""Thống kê báo cáo: điểm theo video, AUC, bootstrap CI theo video, DeLong ghép cặp.

DeLong nhanh (Sun & Xu 2014): với m mẫu dương, n mẫu âm, midrank tx (dương), ty (âm), tz (gộp):
  V10_i = (tz_i - tx_i) / n   (i thuộc dương),   V01_j = 1 - (tz_j - ty_j) / m   (j thuộc âm),
  AUC = mean(V10) = mean(V01),  S = cov(V10)/m + cov(V01)/n,
  Var(AUC1 - AUC2) = S11 + S22 - 2*S12,  z = (AUC1 - AUC2)/sqrt(Var),  p = 2*sf(|z|).
Midrank xử lý ties. Module không import torch.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score


def video_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Trung bình `logit` theo video_id -> [video_id, label, score]."""
    g = df.groupby("video_id", sort=True)
    out = g.agg(label=("label", "first"), score=("logit", "mean")).reset_index()
    return out[["video_id", "label", "score"]]


def auc(y, s) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(s)))


def bootstrap_ci(y, s, groups, n=1000, seed=0, alpha=0.05) -> tuple[float, float, float]:
    """(AUC, lo, hi); resample theo `groups` có hoàn lại, CI percentile.
    Lần rút chỉ có một lớp bị bỏ và rút lại cho đủ n lần hợp lệ."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    vals = []
    attempts = 0
    while len(vals) < n:
        attempts += 1
        if attempts > 100 * n + 1000:
            raise ValueError("không rút được resample có đủ hai lớp")
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, s[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return auc(y, s), float(lo), float(hi)


def _components(y, scores):
    """V10 (k x m), V01 (k x n) cho k bộ điểm."""
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    v10, v01 = [], []
    for s in scores:
        P, N = s[pos], s[~pos]
        tx, ty, tz = rankdata(P), rankdata(N), rankdata(np.concatenate([P, N]))
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    return np.array(v10), np.array(v01), m, n


def delong_paired(y, s1, s2) -> tuple[float, float, float]:
    """(AUC1 - AUC2, z, p hai phía). Raise ValueError nếu y chỉ có một lớp."""
    y = np.asarray(y)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    if not (len(y) == len(s1) == len(s2)):
        raise ValueError("y, s1, s2 phải cùng độ dài")
    if len(np.unique(y)) != 2:
        raise ValueError("y phải có đúng hai lớp")
    v10, v01, m, n = _components(y, [s1, s2])
    d = float(v10[0].mean() - v10[1].mean())
    S = np.cov(v10) / m + np.cov(v01) / n
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    if var <= 1e-18:
        if abs(d) <= 1e-12:
            return d, 0.0, 1.0
        return d, float(np.copysign(np.inf, d)), 0.0
    z = d / np.sqrt(var)
    return d, float(z), float(2 * norm.sf(abs(z)))


def paired_frames(a: pd.DataFrame, b: pd.DataFrame, key="video_id"):
    """Giao theo key, cùng thứ tự (sắp theo key) ở cả hai; n_dropped = số key chỉ ở một bên."""
    ka, kb = set(a[key]), set(b[key])
    common = sorted(ka & kb)
    n_dropped = len(ka ^ kb)
    sa = a[a[key].isin(common)].sort_values(key).reset_index(drop=True)
    sb = b[b[key].isin(common)].sort_values(key).reset_index(drop=True)
    return sa, sb, n_dropped
