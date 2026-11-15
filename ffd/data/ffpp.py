"""Đọc dữ liệu đã tiền xử lý TỪ ĐĨA (không import preprocessing). Hợp đồng do preprocess_ffpp.py sinh:

<data_root>/ffpp/{manifest_raw.parquet, manifest_aug.parquet, drops.parquet, params.json}, ảnh PNG 224x224 tại `path`
(posix, tương đối với <data_root>/ffpp); <data_root>/celebdf/{manifest.parquet, drops.parquet}.
`video_id` của video fake FF++ (ví dụ "071_054") KHÔNG duy nhất giữa các method, nên `frames()` trả
`video_id = "<method>/<video_id>"` và giữ bản gốc ở `src_video_id`.
Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd


def ffpp_dir(data_root) -> Path:
    return Path(data_root) / "ffpp"


def celebdf_dir(data_root) -> Path:
    return Path(data_root) / "celebdf"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path} (data_root có đúng out_root của preprocessing không?)")
    return pd.read_parquet(path)


def frames(data_root, dataset: Literal["ffpp", "celebdf"], split: str | None) -> pd.DataFrame:
    """Frame một split FF++ (lọc `split`) hoặc toàn Celeb-DF. Cột: path (tuyệt đối), label, method, video_id,
    frame_idx (+ src_video_id với FF++). Celeb-DF: method = "real"/"fake" theo label."""
    if dataset == "ffpp":
        root = ffpp_dir(data_root)
        man = _read(root / "manifest_raw.parquet")
        man = man[man.split == split].copy()
        man["src_video_id"] = man.video_id.astype(str)
        man["video_id"] = man.method.astype(str) + "/" + man.src_video_id
    elif dataset == "celebdf":
        root = celebdf_dir(data_root)
        man = _read(root / "manifest.parquet").copy()
        man["method"] = man.label.map({0: "real", 1: "fake"})
        man["video_id"] = man.video_id.astype(str)
    else:
        raise ValueError(f"dataset phải là 'ffpp' hoặc 'celebdf', nhận {dataset!r}")
    man["path"] = man.path.map(lambda p: root / p)
    return man.reset_index(drop=True)


def aug_frames(data_root) -> pd.DataFrame:
    """Ảnh tăng cường (chỉ video train): rep (label 1), seam (label 0). Nhãn theo `kind`, không theo `method`."""
    root = ffpp_dir(data_root)
    man = _read(root / "manifest_aug.parquet")
    bad = man.label.astype(int) != (man.kind == "rep").astype(int)
    if bad.any():
        raise ValueError(f"manifest_aug: {int(bad.sum())} dòng có label != (kind == 'rep')")
    out = man[["path", "label", "kind", "tid", "video_id", "frame_idx"]].copy()
    out["tid"] = out.tid.astype(str)
    out["path"] = out.path.map(lambda p: root / p)
    return out.reset_index(drop=True)


def dropped_videos(data_root, dataset: Literal["ffpp", "celebdf"], split: str | None) -> int:
    """Số video bị loại hẳn (level == "video"; FF++ thêm kind == "" và đúng split), tính theo video duy nhất."""
    if dataset == "ffpp":
        d = _read(ffpp_dir(data_root) / "drops.parquet")
        d = d[(d.level == "video") & (d.kind == "") & (d.split == split)]
        return len(d[["method", "video_id"]].drop_duplicates())
    d = _read(celebdf_dir(data_root) / "drops.parquet")
    return int(d[d.level == "video"].video_id.nunique())


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def data_hashes(data_root) -> dict:
    """sha256 (hex) byte file của params.json và các manifest; file thiếu -> None."""
    f, c = ffpp_dir(data_root), celebdf_dir(data_root)
    return {"params_json_sha256": _sha(f / "params.json"),
            "manifest_sha256": {"raw": _sha(f / "manifest_raw.parquet"), "aug": _sha(f / "manifest_aug.parquet"),
                                "celebdf": _sha(c / "manifest.parquet")}}
