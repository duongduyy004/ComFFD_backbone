"""Sinh data_root TỔNG HỢP (MINH HỌA, không phải FF++/Celeb-DF thật) đúng hợp đồng dữ liệu của preprocess_ffpp:
<root>/ffpp/{raw/..png, aug/train/{rep,seam}/..png, manifest_raw.parquet, manifest_aug.parquet, drops.parquet, params.json}
<root>/celebdf/{test/..png, manifest.parquet, drops.parquet}. Ảnh PNG 224x224 ít entropy. Không import preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SIZE = 224
METHODS = ("DF", "F2F", "FS", "NT", "FSh")
N_TID = {"train": 4, "val": 2, "test": 2}
N_FRAMES = 4

RAW_COLS = ["path", "split", "label", "method", "tid", "video_id", "frame_idx", "crop_x0", "crop_y0", "crop_x1",
            "crop_y1", "src_w", "src_h", "det_score", "face_px", "yaw", "eye_dist"]
AUG_COLS = ["path", "mask_path", "kind", "label", "tid", "video_id", "frame_idx", "method", "regions", "k_reg", "r",
            "erode_k", "blur1_k", "blur2_k", "blur2_sigma", "m0_frac", "mask_frac_final", "delta", "tries", "de_bg",
            "geo_min", "seed", "mask_tries"]
DROP_COLS = ["level", "split", "method", "tid", "video_id", "frame_idx", "kind", "reason"]
CELEB_COLS = ["path", "label", "source_dir", "video_id", "frame_idx", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
              "src_w", "src_h", "det_score", "face_px", "yaw"]


def _img(rng, fake: bool) -> np.ndarray:
    base = rng.integers(40, 200, size=3)
    img = np.empty((SIZE, SIZE, 3), np.uint8)
    img[:] = base
    if fake:
        img[80:150, 80:150] = np.clip(base + 40, 0, 255)
    return img


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


def _fill(cols: list, **kw) -> dict:
    """Điền giá trị mặc định cho các cột không quan trọng."""
    row = {c: 0 for c in cols}
    row.update(kw)
    return row


def _df(rows: list, cols: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols)


def make_data_root(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    ffpp, cel = root / "ffpp", root / "celebdf"
    raw, aug, k = [], [], 0
    for split in ("train", "val", "test"):
        for _ in range(N_TID[split]):
            tid = f"{k:03d}"
            k += 1
            for m in ("real",) + METHODS:
                vid = tid if m == "real" else f"{tid}_{(k + 50):03d}"
                for f in range(N_FRAMES):
                    rel = Path("raw") / split / m / vid / f"{f:05d}.png"
                    _write(ffpp / rel, _img(rng, m != "real"))
                    raw.append(_fill(RAW_COLS, path=rel.as_posix(), split=split, label=int(m != "real"), method=m,
                                     tid=tid, video_id=vid, frame_idx=f, crop_x1=SIZE, crop_y1=SIZE,
                                     src_w=480, src_h=360, det_score=0.99, face_px=200.0))
            if split != "train":
                continue
            for f in range(N_FRAMES):
                for kind in ("rep", "seam"):
                    rel = Path("aug") / "train" / kind / f"{tid}_{f:05d}.png"
                    _write(ffpp / rel, _img(rng, kind == "rep"))
                    # method của seam = method mượn mask (không dùng cho nhãn)
                    aug.append(_fill(AUG_COLS, path=rel.as_posix(), mask_path="", kind=kind, label=int(kind == "rep"),
                                     tid=tid, video_id=tid, frame_idx=f, method=METHODS[f % len(METHODS)],
                                     regions="mouth"))
    drops = [
        _fill(DROP_COLS, level="video", split="test", method="DF", tid="006", video_id="006_056", frame_idx=-1,
              kind="", reason="no_face"),
        # hai dòng sau không được đếm: mức frame; kind khác rỗng
        _fill(DROP_COLS, level="frame", split="test", method="NT", tid="007", video_id="007_057", frame_idx=1,
              kind="", reason="blur"),
        _fill(DROP_COLS, level="video", split="train", method="rep", tid="000", video_id="000", frame_idx=-1,
              kind="rep", reason="no_neighbor"),
    ]
    ffpp.mkdir(parents=True, exist_ok=True)
    _df(raw, RAW_COLS).to_parquet(ffpp / "manifest_raw.parquet", index=False)
    _df(aug, AUG_COLS).to_parquet(ffpp / "manifest_aug.parquet", index=False)
    _df(drops, DROP_COLS).to_parquet(ffpp / "drops.parquet", index=False)
    (ffpp / "params.json").write_text(json.dumps({"MINH_HOA": True, "size": SIZE}), encoding="utf-8")

    cel_rows = []
    for v, (label, src) in enumerate([(0, "Celeb-real"), (0, "YouTube-real"), (1, "Celeb-synthesis"),
                                      (1, "Celeb-synthesis")]):
        for f in range(N_FRAMES):
            rel = Path("test") / f"v{v}" / f"{f:05d}.png"
            _write(cel / rel, _img(rng, bool(label)))
            cel_rows.append(_fill(CELEB_COLS, path=rel.as_posix(), label=label, source_dir=src, video_id=f"v{v}",
                                  frame_idx=f, crop_x1=SIZE, crop_y1=SIZE))
    _df(cel_rows, CELEB_COLS).to_parquet(cel / "manifest.parquet", index=False)
    _df([_fill(DROP_COLS, level="video", split="test", video_id="v9", frame_idx=-1, kind="", reason="no_face",
               method="", tid="")], DROP_COLS).to_parquet(cel / "drops.parquet", index=False)
    return root
