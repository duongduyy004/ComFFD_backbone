"""Sinh thư mục ffpp_root TỔNG HỢP (MINH HỌA, không phải FF++ thật) đúng hợp đồng dữ liệu của detector:
ffpp_raw/{manifest.parquet, <split>/<method>/<video>/<frame>.png} và ffpp_aug/{manifest.parquet, train/{rep,seam}/*.jpg}.
Không dùng preprocessing. Ảnh 336x336 ít entropy (màu nền theo video + nhiễu nhẹ) để file nhỏ.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

METHODS = ("DF", "F2F", "FS", "NT", "FSh")
RAW_HASH = "rawMINHHOA01"
AUG_HASH = "augMINHHOA01"
SIZE = 336


def _img(rng, fake: bool) -> np.ndarray:
    base = rng.integers(40, 200, size=3)
    img = np.empty((SIZE, SIZE, 3), np.uint8)
    img[:] = base
    if fake:  # vệt sáng giữa mặt cho lớp fake
        img[120:220, 120:220] = np.clip(base + 40, 0, 255)
    img += rng.integers(0, 4, size=(SIZE, SIZE, 1), dtype=np.uint8)
    return img


def _write(path: Path, img: np.ndarray, ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 85] if ext == ".jpg" else [])
    assert ok
    buf.tofile(str(path))


def make_ffpp_root(root: Path, n_tid=None, n_frames=None, aug_k: int = 2, seed: int = 0) -> Path:
    """n_tid / n_frames: số video nguồn và số frame keep mỗi video theo split."""
    n_tid = n_tid or {"train": 8, "val": 3, "test": 3}
    n_frames = n_frames or {"train": 8, "val": 4, "test": 4}
    rng = np.random.default_rng(seed)
    root = Path(root)
    raw, aug, k = [], [], 0
    for split in ("train", "val", "test"):
        for i in range(n_tid[split]):
            tid = f"{k:03d}"
            k += 1
            frames = [10 * j for j in range(n_frames[split])]
            for m in ("real",) + METHODS:
                src = tid if m == "real" else f"{(k + 50):03d}"
                vdir = Path("ffpp_raw") / split / m / (tid if m == "real" else f"{tid}_{src}")
                for f in frames:
                    rel = vdir / f"{f:05d}.png"
                    _write(root / rel, _img(rng, m != "real"), ".png")
                    raw.append(dict(split=split, method=m, tid=tid, src=src, video_id=f"{tid}_{m}", frame_idx=f,
                                    keep=True, drop_reason="", path=rel.as_posix(), det_score=0.99, face_px=200.0,
                                    yaw=0.0, config_hash=RAW_HASH))
            if split != "train":
                continue
            for f in frames:
                for kk in range(aug_k):
                    for kind in ("rep", "seam"):
                        rel = Path("ffpp_aug") / "train" / kind / f"{tid}_{f:05d}_k{kk}.jpg"
                        _write(root / rel, _img(rng, kind == "rep"), ".jpg")
                        aug.append(dict(kind=kind, tid=tid, frame_idx=f, k=kk,
                                        method=METHODS[(f + kk) % len(METHODS)] if kind == "rep" else "",
                                        regions="mouth", skipped="", path=rel.as_posix(), q=85, config_hash=AUG_HASH))
                # một dòng skip để kiểm bộ lọc skipped
                aug.append(dict(kind="seam", tid=tid, frame_idx=f, k=aug_k, method="", regions="",
                                skipped="no_neighbor", path="", q=0, config_hash=AUG_HASH))
    for name, rows in (("ffpp_raw", raw), ("ffpp_aug", aug)):
        (root / name).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(root / name / "manifest.parquet", index=False)
    return root
