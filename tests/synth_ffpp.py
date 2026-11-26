"""Sinh data_root TỔNG HỢP (MINH HỌA, không phải FF++/Celeb-DF thật) đúng hợp đồng dữ liệu của preprocess_ffpp:
<root>/ffpp/{raw/..jpg, aug/train/{rep,seam}/..jpg (+ mask .png), manifest_raw.parquet, manifest_aug.parquet, drops.parquet,
params.json}, <root>/celebdf/{test/..jpg, manifest.parquet, drops.parquet}. Như preprocessing v2: ảnh JPEG q95 4:4:4
ở kích thước crop gốc, không vuông, khác nhau theo target (`w`, `h` trong manifest); mask PNG cùng kích thước với ảnh.
Ít entropy. Không import preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

JPEG_Q = 95
SIZES = ((238, 178), (300, 224), (180, 136), (264, 200))  # (h, w) theo target, có cả nhỏ hơn và lớn hơn 224
METHODS = ("DF", "F2F", "FS", "NT", "FSh")
N_TID = {"train": 4, "val": 2, "test": 2}
N_FRAMES = 4
R_CYCLE = (0.5, 0.75, 1.0, 1.0)  # r và mask_frac_final của ảnh aug tổng hợp (MINH HỌA)

RAW_COLS = ["path", "split", "label", "method", "tid", "video_id", "frame_idx", "crop_x0", "crop_y0", "crop_x1",
            "crop_y1", "src_w", "src_h", "det_score", "face_px", "yaw", "eye_dist", "w", "h", "jpeg_q"]
AUG_COLS = ["path", "mask_path", "kind", "label", "tid", "video_id", "frame_idx", "method", "regions", "k_reg", "r",
            "erode_k", "blur1_k", "blur2_k", "blur2_sigma", "m0_frac", "mask_frac_final", "delta", "tries", "de_bg",
            "geo_min", "seed", "mask_tries", "de_vis", "w", "h", "jpeg_q"]
DROP_COLS = ["level", "split", "method", "tid", "video_id", "frame_idx", "kind", "reason"]
CELEB_COLS = ["path", "label", "source_dir", "video_id", "frame_idx", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
              "src_w", "src_h", "det_score", "face_px", "yaw", "w", "h", "jpeg_q"]


def _box(hw) -> tuple[slice, slice]:
    """Vùng "fake"/mask: lệch trái (bất đối xứng ngang, để test lật mask có nghĩa)."""
    h, w = hw
    return slice(h * 3 // 8, h * 5 // 8), slice(w // 4, w // 2)


def _img(rng, fake: bool, hw) -> np.ndarray:
    base = rng.integers(40, 200, size=3)
    img = np.empty((*hw, 3), np.uint8)
    img[:] = base
    if fake:
        img[_box(hw)] = np.clip(base + 40, 0, 255)
    return img


def _write(path: Path, img: np.ndarray) -> None:
    """Ảnh màu -> JPEG q95 4:4:4; mask 1 kênh -> PNG (như save_img / save_mask của preprocessing v2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                             cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    else:
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
            hw = SIZES[k % len(SIZES)]
            k += 1
            for m in ("real",) + METHODS:
                vid = tid if m == "real" else f"{tid}_{(k + 50):03d}"
                for f in range(N_FRAMES):
                    rel = Path("raw") / split / m / vid / f"{f:05d}.jpg"
                    _write(ffpp / rel, _img(rng, m != "real", hw))
                    raw.append(_fill(RAW_COLS, path=rel.as_posix(), split=split, label=int(m != "real"), method=m,
                                     tid=tid, video_id=vid, frame_idx=f, crop_x1=hw[1], crop_y1=hw[0],
                                     src_w=480, src_h=360, det_score=0.99, face_px=200.0, w=hw[1], h=hw[0],
                                     jpeg_q=JPEG_Q))
            if split != "train":
                continue
            for f in range(N_FRAMES):
                for kind in ("rep", "seam"):
                    rel = Path("aug") / "train" / kind / f"{tid}_{f:05d}.jpg"
                    _write(ffpp / rel, _img(rng, kind == "rep", hw))
                    mrel = Path("aug") / "train" / "masks" / kind / f"{tid}_{f:05d}.png"
                    mask = np.zeros(hw, np.uint8)
                    mask[_box(hw)] = 255
                    _write(ffpp / mrel, mask)
                    # method của seam = method mượn mask (không dùng cho nhãn)
                    aug.append(_fill(AUG_COLS, path=rel.as_posix(), mask_path=mrel.as_posix(), kind=kind,
                                     label=int(kind == "rep"), tid=tid, video_id=tid, frame_idx=f,
                                     method=METHODS[f % len(METHODS)], regions="mouth",
                                     r=R_CYCLE[f % len(R_CYCLE)], mask_frac_final=0.02 * (f + 1),
                                     de_vis=10.0 if kind == "rep" else 2.0, w=hw[1], h=hw[0], jpeg_q=JPEG_Q))
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
    (ffpp / "params.json").write_text(json.dumps({"MINH_HOA": True, "jpeg_quality": JPEG_Q}), encoding="utf-8")

    cel_rows = []
    for v, (label, src) in enumerate([(0, "Celeb-real"), (0, "YouTube-real"), (1, "Celeb-synthesis"),
                                      (1, "Celeb-synthesis")]):
        hw = SIZES[v % len(SIZES)]
        for f in range(N_FRAMES):
            rel = Path("test") / f"v{v}" / f"{f:05d}.jpg"
            _write(cel / rel, _img(rng, bool(label), hw))
            cel_rows.append(_fill(CELEB_COLS, path=rel.as_posix(), label=label, source_dir=src, video_id=f"v{v}",
                                  frame_idx=f, crop_x1=hw[1], crop_y1=hw[0], w=hw[1], h=hw[0], jpeg_q=JPEG_Q,
                                  face_px=float(hw[1] // 2 + 3 * f)))
    _df(cel_rows, CELEB_COLS).to_parquet(cel / "manifest.parquet", index=False)
    _df([_fill(DROP_COLS, level="video", split="test", video_id="v9", frame_idx=-1, kind="", reason="no_face",
               method="", tid="")], DROP_COLS).to_parquet(cel / "drops.parquet", index=False)
    return root
