"""Lọc rep/seam yếu và cân bằng lại đường nối cho arm `raw_aug_filt*` (hàm thuần, xác định, không đọc đĩa).

Lọc: giữ dòng có r >= FILTER_MIN_R và mask_frac_final >= tau_m, tau_m = median mask_frac_final của MỌI dòng rep
chưa lọc (toàn cục); cùng tiêu chí cho rep và seam. Cân bằng: port `balance_aug` của preprocessing
(preprocess_ffpp.py, không import) nhưng nhận n_real/n_fake trực tiếp và trả stats thay vì drop records.
Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

import pandas as pd

from ..config import ARMS, FILTER_ARMS, LOC_ARMS
from ..io_utils import item_seed

FILTER_MIN_R = 0.75
LOC_LAMBDA = 1.0


def arm_constants(arm: str) -> dict:
    """Hằng số gắn arm (không nằm trong TrainConfig). Arm cũ -> {}."""
    if arm not in ARMS:
        raise ValueError(f"arm phải thuộc {ARMS}, nhận {arm!r}")
    if arm not in FILTER_ARMS:
        return {}
    c = {"filter_min_r": FILTER_MIN_R, "filter_mask": "median_rep"}
    if arm in LOC_ARMS:
        c["loc_lambda"] = LOC_LAMBDA
    return c


def check_arm_constants(arm: str, info: dict, action: str) -> None:
    """Arm filt: run_info.json phải có `arm_constants` đúng bằng hằng số hiện tại, nếu không ValueError.
    Dùng chung cho resume (train) và bỏ qua/eval lại (experiment.ensure_run). Arm cũ: không kiểm."""
    if arm in FILTER_ARMS and info.get("arm_constants") != arm_constants(arm):
        raise ValueError(f"hằng số arm khác run cũ ({info.get('arm_constants')} != {arm_constants(arm)}); {action}")


def filter_aug(aug: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Lọc ảnh aug. Trả (aug đã giữ, stats). Rep hoặc seam rỗng sau lọc -> ValueError."""
    if aug[["r", "mask_frac_final"]].isna().any().any():
        raise ValueError("filter_aug: r hoặc mask_frac_final có NaN")
    rep = aug.kind == "rep"
    n_rep, n_seam = int(rep.sum()), int((~rep).sum())
    if not n_rep:
        raise ValueError("filter_aug: không có dòng rep để tính tau_m")
    tau_m = float(aug.loc[rep, "mask_frac_final"].median())
    keep = (aug.r >= FILTER_MIN_R) & (aug.mask_frac_final >= tau_m)
    kept = aug[keep].reset_index(drop=True)
    k_rep, k_seam = int((kept.kind == "rep").sum()), int((kept.kind == "seam").sum())
    if not k_rep or not k_seam:
        raise ValueError(f"filter_aug: sau lọc còn rep={k_rep}, seam={k_seam} (r>={FILTER_MIN_R}, "
                         f"mask_frac_final>={tau_m:.6g}); dữ liệu quá nhỏ hoặc ngưỡng quá cao")
    return kept, {"tau_m": tau_m, "n_rep_before": n_rep, "n_seam_before": n_seam,
                  "n_rep_kept": k_rep, "n_seam_kept": k_seam}


def balance_aug(aug: pd.DataFrame, n_real: int, n_fake: int) -> tuple[pd.DataFrame, dict]:
    """Cân bằng p_real = S/(R+S) với p_fake = P/(F+P); R, F = số ảnh raw train label 0/1.

    Bên cao hơn bị bớt: seam giữ P*R//F, rep giữ S*F//R (số nguyên), chỉ bớt khi vẫn cao hơn bên kia dù bên kia
    thêm một mẫu (giữ idempotent). Bỏ theo khóa tăng của item_seed(..., "balance"). Trả (aug đã giữ, stats).
    """
    R, F = int(n_real), int(n_fake)
    S = int((aug.kind == "seam").sum())
    P = int((aug.kind == "rep").sum())
    p_real = S / (R + S) if (R + S) else 0.0
    p_fake = P / (F + P) if (F + P) else 0.0
    out, n_s, n_p = aug, S, P
    if p_real > p_fake and p_real > (P + 1) / (F + P + 1):
        kind, keep_n, total = "seam", (P * R // F if F else 0), S
    elif p_fake > p_real and p_fake > (S + 1) / (R + S + 1):
        kind, keep_n, total = "rep", (S * F // R if R else 0), P
    else:
        kind = None
    if kind is not None:
        def key(i):
            r = aug.loc[i]
            parts = (r.tid, int(r.frame_idx)) if kind == "seam" else (r.tid, int(r.frame_idx), r.method)
            return (item_seed(*parts, "balance"), *parts)
        idx = [i for i, k in zip(aug.index, aug.kind) if k == kind]
        drop_idx = sorted(idx, key=key)[:max(0, total - keep_n)]
        out = aug.drop(index=drop_idx).reset_index(drop=True)
        n_s, n_p = int((out.kind == "seam").sum()), int((out.kind == "rep").sum())
    if not n_s or not n_p:
        raise ValueError(f"balance_aug: sau cân bằng còn rep={n_p}, seam={n_s} (R={R}, F={F}, S={S}, P={P})")
    return out, {"n_rep_after": n_p, "n_seam_after": n_s, "p_real": p_real, "p_fake": p_fake}
