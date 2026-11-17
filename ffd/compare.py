"""So sánh hai arm -> JSON trong `ckpt_root`.

`compare_runs` (raw vs raw_aug) -> `compare.json`, khóa `auc_raw`, `auc_raw_aug` (định dạng v2, không đổi);
`compare_pair(ckpt_root, a, b)` -> `compare_<a>__<b>.json`, khóa `auc_<a>`, `auc_<b>`; PAIRS là bốn cặp của v3.
Ghép frame theo (video_id, frame_idx); tập frame hoặc nhãn khác nhau -> ValueError.
diff = AUC(b) - AUC(a); CI = paired bootstrap 2000 lần trên frame (seed 0); p = DeLong ghép cặp.
Module không import torch. Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import stats
from .io_utils import write_json_atomic

NOTE = "Exploratory: 1 run/arm, không prereg; frame trong cùng video tương quan nên CI và p-value lạc quan."
DATASETS = ("ffpp_test", "celebdf")
PAIRS = (("raw", "raw_aug_filt"), ("raw", "raw_aug_filt_loc"), ("raw_aug", "raw_aug_filt"),
         ("raw_aug_filt", "raw_aug_filt_loc"))
_KEY = ["video_id", "frame_idx"]


def _load(run: Path) -> dict:
    return json.loads((run / "result.json").read_text(encoding="utf-8"))


def _compare_dataset(name: str, dir_a: Path, dir_b: Path, arm_a: str, arm_b: str) -> dict:
    a = pd.read_parquet(dir_a / f"scores_{name}.parquet")
    b = pd.read_parquet(dir_b / f"scores_{name}.parquet")
    for arm, df in ((arm_a, a), (arm_b, b)):
        if df.duplicated(_KEY).any():
            raise ValueError(f"{name}: scores của {arm} có (video_id, frame_idx) trùng")
    ka = set(map(tuple, a[_KEY].to_numpy()))
    kb = set(map(tuple, b[_KEY].to_numpy()))
    if ka != kb:
        raise ValueError(f"{name}: hai arm chấm hai tập frame khác nhau "
                         f"({len(ka - kb)} chỉ ở {arm_a}, {len(kb - ka)} chỉ ở {arm_b})")
    m = a.merge(b, on=_KEY, suffixes=("_a", "_b")).sort_values(_KEY).reset_index(drop=True)
    if not (m["label_a"] == m["label_b"]).all():
        raise ValueError(f"{name}: label không khớp giữa hai arm")
    y, s1, s2 = m["label_a"].to_numpy(), m["logit_a"].to_numpy(), m["logit_b"].to_numpy()
    diff, lo, hi = stats.paired_bootstrap_diff(y, s1, s2, n=2000, seed=0)
    _, _, p = stats.delong_paired(y, s1, s2)
    return {f"auc_{arm_a}": stats.auc(y, s1), f"auc_{arm_b}": stats.auc(y, s2),
            "diff": diff, "diff_ci95": [lo, hi], "delong_p": p}


def _compare(ckpt_root: Path, arm_a: str, arm_b: str, path: Path) -> Path:
    dir_a, dir_b = ckpt_root / arm_a, ckpt_root / arm_b
    r1, r2 = _load(dir_a), _load(dir_b)
    out: dict = {}
    for name in DATASETS:
        v1, v2 = r1.get(name), r2.get(name)
        skipped = [x["skipped"] for x in (v1, v2) if isinstance(x, dict) and "skipped" in x]
        if v1 is None or v2 is None:
            out[name] = {"skipped": "thiếu kết quả ở một trong hai run"}
        elif skipped:
            out[name] = {"skipped": "; ".join(skipped)}
        else:
            out[name] = _compare_dataset(name, dir_a, dir_b, arm_a, arm_b)
    for k in ("total_steps", "epochs_run", "best_epoch"):
        out[k] = {arm_a: r1.get(k), arm_b: r2.get(k)}
    out["note"] = NOTE
    write_json_atomic(out, path)
    return path


def compare_runs(ckpt_root: Path) -> Path:
    """raw vs raw_aug -> `ckpt_root/compare.json` (định dạng v2)."""
    ckpt_root = Path(ckpt_root)
    return _compare(ckpt_root, "raw", "raw_aug", ckpt_root / "compare.json")


def compare_pair(ckpt_root: Path, a: str, b: str) -> Path:
    """Arm a vs arm b (diff = AUC(b) - AUC(a)) -> `ckpt_root/compare_<a>__<b>.json`."""
    ckpt_root = Path(ckpt_root)
    return _compare(ckpt_root, a, b, ckpt_root / f"compare_{a}__{b}.json")
