"""So sánh hai arm (raw vs raw_aug) -> `ckpt_root/compare.json`.

Ghép frame theo (video_id, frame_idx); tập frame hoặc nhãn khác nhau -> ValueError.
diff = AUC(raw_aug) - AUC(raw); CI = paired bootstrap 2000 lần trên frame (seed 0); p = DeLong ghép cặp.
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
_KEY = ["video_id", "frame_idx"]


def _load(run: Path) -> dict:
    return json.loads((run / "result.json").read_text(encoding="utf-8"))


def _compare_dataset(name: str, raw_dir: Path, aug_dir: Path) -> dict:
    a = pd.read_parquet(raw_dir / f"scores_{name}.parquet")
    b = pd.read_parquet(aug_dir / f"scores_{name}.parquet")
    for arm, df in (("raw", a), ("raw_aug", b)):
        if df.duplicated(_KEY).any():
            raise ValueError(f"{name}: scores của {arm} có (video_id, frame_idx) trùng")
    ka = set(map(tuple, a[_KEY].to_numpy()))
    kb = set(map(tuple, b[_KEY].to_numpy()))
    if ka != kb:
        raise ValueError(f"{name}: hai arm chấm hai tập frame khác nhau ({len(ka - kb)} chỉ ở raw, {len(kb - ka)} chỉ ở raw_aug)")
    m = a.merge(b, on=_KEY, suffixes=("_raw", "_aug")).sort_values(_KEY).reset_index(drop=True)
    if not (m["label_raw"] == m["label_aug"]).all():
        raise ValueError(f"{name}: label không khớp giữa hai arm")
    y, s1, s2 = m["label_raw"].to_numpy(), m["logit_raw"].to_numpy(), m["logit_aug"].to_numpy()
    diff, lo, hi = stats.paired_bootstrap_diff(y, s1, s2, n=2000, seed=0)
    _, _, p = stats.delong_paired(y, s1, s2)
    return {"auc_raw": stats.auc(y, s1), "auc_raw_aug": stats.auc(y, s2),
            "diff": diff, "diff_ci95": [lo, hi], "delong_p": p}


def compare_runs(ckpt_root: Path) -> Path:
    ckpt_root = Path(ckpt_root)
    raw_dir, aug_dir = ckpt_root / "raw", ckpt_root / "raw_aug"
    r1, r2 = _load(raw_dir), _load(aug_dir)
    out: dict = {}
    for name in DATASETS:
        v1, v2 = r1.get(name), r2.get(name)
        skipped = [x["skipped"] for x in (v1, v2) if isinstance(x, dict) and "skipped" in x]
        if v1 is None or v2 is None:
            out[name] = {"skipped": "thiếu kết quả ở một trong hai run"}
        elif skipped:
            out[name] = {"skipped": "; ".join(skipped)}
        else:
            out[name] = _compare_dataset(name, raw_dir, aug_dir)
    for k in ("total_steps", "epochs_run", "best_epoch"):
        out[k] = {"raw": r1.get(k), "raw_aug": r2.get(k)}
    out["note"] = NOTE
    path = ckpt_root / "compare.json"
    write_json_atomic(out, path)
    return path
