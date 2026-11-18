"""Thư mục cache đặc trưng TỔNG HỢP (MINH HỌA, không phải CLIP thật) đúng định dạng ffd.features.

Tín hiệu nhãn mạnh ở slot 11 (tầng 12), yếu ở slot 24 (pooler); còn lại là nhiễu.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# dataset -> (số video real, số video fake)
_N_VIDEOS = {"train": (10, 30), "val": (6, 18), "ffpp_test": (6, 18), "celebdf": (8, 12)}
N_FRAMES = 4


def _vid(ds: str, i: int, label: int) -> tuple[str, str]:
    if ds == "celebdf":
        src = "Celeb-synthesis" if label else ("Celeb-real" if i % 2 == 0 else "YouTube-real")
        return f"{src}__{i:03d}", "fake" if label else "real"
    m = "DF" if label else "real"
    return f"{m}/{i:03d}", m


def make_feat_dir(root, n_layers: int = 24, dim: int = 8, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    fd = Path(root) / "local"
    for ds, (nr, nf) in _N_VIDEOS.items():
        rows = []
        for i, label in enumerate([0] * nr + [1] * nf):
            vid, m = _vid(ds, i, label)
            rows += [{"video_id": vid, "frame_idx": f, "label": label, "method": m, "path": f"{vid}/{f:05d}.png"}
                     for f in range(N_FRAMES)]
        idx = pd.DataFrame(rows)
        y = idx.label.to_numpy()[:, None]
        x = rng.normal(size=(len(idx), n_layers + 1, dim))
        x[:, 11] += 2.0 * y
        x[:, n_layers] += 0.3 * y
        out = fd / ds
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "cls.npy", x.astype(np.float16))
        idx.to_parquet(out / "index.parquet", index=False)
        meta = {"commit": "local", "manifest_sha256": "MINH_HOA", "params_json_sha256": "MINH_HOA",
                "transform_id": "MINH_HOA", "N": len(idx), "num_layers": n_layers, "dim": dim, "dataset": ds,
                "n_videos_dropped": 0}
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return fd
