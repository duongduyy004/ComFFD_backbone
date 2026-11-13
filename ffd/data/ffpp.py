"""Đọc dữ liệu FF++ đã tiền xử lý TỪ ĐĨA, không import preprocessing (tách 13/11/2026).

Hợp đồng dữ liệu (do preprocessing sinh, xem detector/README.md mục "Hợp đồng dữ liệu"), dưới `ffpp_root`:
- ffpp_raw/manifest.parquet: cột RAW_COLS; ảnh PNG tại `path` (tương đối với ffpp_root). Mỗi video căn độc lập.
- ffpp_aug/manifest.parquet: cột AUG_COLS; ảnh JPEG tại `path`; kind ∈ {rep (nhãn fake), seam (nhãn real)};
  dòng `skipped != ""` bị bỏ. Chỉ video train.
Mỗi manifest phải có đúng một giá trị `config_hash` (hash tham số preprocessing của stage đó). Detector không tính
lại hash này; chỉ ghi vào run_info lúc train và so lại lúc eval (`manifest_hashes`).
Trạng thái: chưa chạy thử trên FF++ thật.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..guards import LeakageError, assert_subset
from ..io_utils import read_image

RAW_COLS = ("split", "method", "tid", "video_id", "frame_idx", "keep", "path", "config_hash")
AUG_COLS = ("kind", "tid", "frame_idx", "method", "regions", "skipped", "path", "config_hash")


class StaleArtifactError(ValueError):
    """Manifest không nhất quán (nhiều config_hash, thiếu cột): artifact tiền xử lý hỏng hoặc trộn lần chạy."""


def raw_root(ffpp_root) -> Path:
    return Path(ffpp_root) / "ffpp_raw"


def aug_root(ffpp_root) -> Path:
    return Path(ffpp_root) / "ffpp_aug"


def _read_manifest(path: Path, cols: tuple) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path} (ffpp_root có đúng thư mục output của preprocessing không?)")
    man = pd.read_parquet(path)
    miss = [c for c in cols if c not in man.columns]
    if miss:
        raise StaleArtifactError(f"{path} thiếu cột {miss}")
    hs = set(man.config_hash)
    if len(hs) > 1:
        raise StaleArtifactError(f"{path} trộn nhiều config_hash {sorted(hs)}; chạy lại stage đó với --force")
    return man


def raw_manifest(ffpp_root) -> pd.DataFrame:
    return _read_manifest(raw_root(ffpp_root) / "manifest.parquet", RAW_COLS)


def aug_manifest(ffpp_root) -> pd.DataFrame:
    return _read_manifest(aug_root(ffpp_root) / "manifest.parquet", AUG_COLS)


def _hash_of(man: pd.DataFrame):
    return str(man.config_hash.iloc[0]) if len(man) else None


def manifest_hashes(ffpp_root) -> dict:
    """{ffpp_raw_hash, ffpp_aug_hash}: config_hash của từng manifest; manifest không tồn tại -> None."""
    out = {}
    for k, root, cols in (("ffpp_raw_hash", raw_root, RAW_COLS), ("ffpp_aug_hash", aug_root, AUG_COLS)):
        p = root(ffpp_root) / "manifest.parquet"
        out[k] = _hash_of(_read_manifest(p, cols)) if p.exists() else None
    return out


class FFppRaw:
    """ffpp_raw/<subset>, chỉ frame keep. items = [(tid, frame_idx, method)] đã sắp xếp;
    phần tử i -> {"image": BGR uint8, "targets": {L0, method, tid, frame_idx, video_id}}."""

    def __init__(self, ffpp_root, subset: str):
        assert_subset(subset, {"train", "val", "test"}, "FFppRaw")
        self.root = Path(ffpp_root)
        man = raw_manifest(self.root)
        man = man[(man.split == subset) & man.keep.astype(bool)]
        self._path = {(str(t), int(f), str(m)): p
                      for t, f, m, p in man[["tid", "frame_idx", "method", "path"]].itertuples(index=False)}
        self.items = sorted(self._path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        t, f, s = self.items[i]
        img = read_image(self.root / self._path[(t, f, s)])
        return {"image": img, "targets": {"L0": 0 if s == "real" else 1, "method": s, "tid": t, "frame_idx": f,
                                          "video_id": f"{t}_{s}"}}


def expected_videos(ffpp_root, subset: str) -> dict:
    """{video_id: nhãn} của mọi video trong split, kể cả video không còn frame keep (detector chấm điểm 0)."""
    assert_subset(subset, {"train", "val", "test"}, "expected_videos")
    man = raw_manifest(ffpp_root)
    man = man[man.split == subset]
    return {str(v): (0 if m == "real" else 1)
            for v, m in man[["video_id", "method"]].drop_duplicates().itertuples(index=False)}


class FFppAug:
    """ffpp_aug/train: rep (nhãn fake) và seam (nhãn real) theo (tid, frame). Mọi tid phải là video train theo
    ffpp_raw/manifest (split == "train"); khác -> LeakageError."""

    def __init__(self, ffpp_root):
        self.root = Path(ffpp_root)
        man = aug_manifest(self.root)
        raw = raw_manifest(self.root)
        split = raw.drop_duplicates("tid").set_index("tid").split
        bad = sorted(t for t in set(man.tid) if split.get(t) != "train")
        if bad:
            raise LeakageError(f"ffpp_aug chứa video không thuộc train theo ffpp_raw/manifest (ví dụ {bad[0]!r})")
        ok = man[man.skipped == ""]
        self._lists: dict = {}
        for kind, t, f, path in ok[["kind", "tid", "frame_idx", "path"]].itertuples(index=False):
            self._lists.setdefault((kind, str(t), int(f)), []).append(path)
        self._meta = {r["path"]: r for r in ok.to_dict("records")}

    def rep(self, tid, f) -> list:
        return list(self._lists.get(("rep", str(tid), int(f)), []))

    def seam(self, tid, f) -> list:
        return list(self._lists.get(("seam", str(tid), int(f)), []))

    def read(self, path) -> np.ndarray:
        return read_image(self.root / path)

    def meta(self, path) -> dict:
        return dict(self._meta[path])
