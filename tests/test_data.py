"""Test hợp đồng dữ liệu v2, TrainList cố định, transform 224. Dữ liệu MINH HỌA (synth_ffpp)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from ffd import io_utils  # noqa: E402
from ffd.data.ffpp import aug_frames, data_hashes, dropped_videos, frames  # noqa: E402
from ffd.data.trainset import TrainList  # noqa: E402
from ffd.data.transforms import eval_transform, train_transform  # noqa: E402
from ffd.guards import LeakageError, allow_read  # noqa: E402


def _copy(data_root, dst):
    shutil.copytree(data_root, dst)
    return dst


def test_frames_ffpp_and_celebdf(data_root):
    tr = frames(data_root, "ffpp", "train")
    assert set(tr.columns) >= {"path", "label", "method", "video_id", "frame_idx"} and tr.path.map(Path.exists).all()
    assert set(frames(data_root, "celebdf", None).method) == {"real", "fake"}


def test_frames_video_id_unique_per_method(data_root):
    tr = frames(data_root, "ffpp", "train")
    assert (tr.video_id.str.split("/").str[0] == tr.method).all()
    assert (tr.src_video_id == tr.video_id.str.split("/").str[1]).all()
    assert (tr.groupby("video_id").method.nunique() == 1).all()


def test_frames_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest_raw"):
        frames(tmp_path, "ffpp", "train")


def test_dropped_videos(data_root):
    assert dropped_videos(data_root, "ffpp", "test") == 1
    assert dropped_videos(data_root, "ffpp", "val") == 0
    assert dropped_videos(data_root, "celebdf", None) == 1


def test_data_hashes(data_root, tmp_path):
    h = data_hashes(data_root)
    assert len(h["params_json_sha256"]) == 64 and all(len(v) == 64 for v in h["manifest_sha256"].values())
    root = _copy(data_root, tmp_path / "r")
    (root / "celebdf" / "manifest.parquet").unlink()
    assert data_hashes(root)["manifest_sha256"]["celebdf"] is None


def test_trainlist_sizes(data_root):
    raw = TrainList(data_root, "raw", 0, 0.5, 95)
    aug = TrainList(data_root, "raw_aug", 0, 0.5, 95)
    assert set(raw.items.source) == {"raw"} and len(aug.items) == len(raw.items) + len(aug_frames(data_root))
    assert set(aug.items.source) == {"raw", "rep", "seam"}
    x = raw[0]
    assert x["image"].shape == (3, 224, 224) and x["source"] == "raw" and isinstance(x["jpeg"], bool)
    # seam là real dù method mượn mask là fake
    assert (aug.items[aug.items.source == "seam"].label == 0).all()


def test_trainlist_deterministic_per_epoch(data_root):
    a = TrainList(data_root, "raw", 0, 0.5, 95)
    b = TrainList(data_root, "raw", 0, 0.5, 95)
    a.set_epoch(2)
    b.set_epoch(2)
    assert (a[3]["image"] == b[3]["image"]).all() and a[3]["jpeg"] == b[3]["jpeg"]


def test_trainlist_rejects_non_train_aug(data_root, tmp_path):
    root = _copy(data_root, tmp_path / "r")
    p = root / "ffpp" / "manifest_aug.parquet"
    m = pd.read_parquet(p)
    m.loc[0, "tid"] = "004"  # tid của val
    m.to_parquet(p, index=False)
    with pytest.raises(LeakageError):
        TrainList(root, "raw_aug", 0, 0.5, 95)


def test_transform_rejects_non_224():
    with pytest.raises(ValueError):
        eval_transform(np.zeros((336, 336, 3), np.uint8))


def test_train_transform_p_jpeg(monkeypatch):
    n = []
    real = io_utils.jpeg
    monkeypatch.setattr(io_utils, "jpeg", lambda im, q: (n.append(q), real(im, q))[1])
    img = np.zeros((224, 224, 3), np.uint8)
    for p, want in ((0.0, False), (1.0, True)):
        n.clear()
        x, j = train_transform(img, np.random.default_rng(0), p, 95)
        assert j is want and x.shape == (3, 224, 224) and len(n) == int(want)
    assert n == [95]


def test_aug_label_kind_mismatch(data_root, tmp_path):
    root = _copy(data_root, tmp_path / "r")
    p = root / "ffpp" / "manifest_aug.parquet"
    m = pd.read_parquet(p)
    m.loc[m.kind == "rep", "label"] = 0
    m.to_parquet(p, index=False)
    with pytest.raises(ValueError):
        aug_frames(root)


def test_guard_train_allows_aug_not_test():
    for w in ("ffpp_train", "ffpp_val", "ffpp_aug"):
        allow_read("train", w)
    for w in ("ffpp_test", "celebdf"):
        with pytest.raises(LeakageError):
            allow_read("train", w)
