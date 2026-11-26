"""Test hợp đồng dữ liệu (preprocessing v2: JPEG kích thước gốc), TrainList cố định, transform resize về 224.
Dữ liệu MINH HỌA (synth_ffpp)."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from ffd import io_utils  # noqa: E402
from ffd.data.ffpp import aug_frames, data_hashes, dropped_videos, frames  # noqa: E402
from ffd.data.trainset import TrainList  # noqa: E402
from ffd.data.transforms import eval_transform, to_size, train_transform  # noqa: E402
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


def test_synth_matches_v2_contract(data_root):
    """Dữ liệu tổng hợp đúng hợp đồng v2: ảnh .jpg không vuông, w/h trong manifest khớp file, mask cùng kích thước."""
    tr = frames(data_root, "ffpp", "train")
    assert (tr.path.map(lambda p: p.suffix) == ".jpg").all() and (tr.w != tr.h).any()
    for p, w, h in tr[["path", "w", "h"]].head(8).itertuples(index=False):
        assert io_utils.read_image(p).shape[:2] == (h, w)
    aug = aug_frames(data_root)
    for p, mp in aug[["path", "mask_path"]].head(8).itertuples(index=False):
        assert io_utils.read_image(p).shape[:2] == io_utils.read_image(mp).shape[:2]


def test_to_size_interp():
    rng = np.random.default_rng(0)
    same = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    assert to_size(same) is same  # dữ liệu v1 (224) đi qua nguyên vẹn
    big = rng.integers(0, 255, (300, 240, 3), dtype=np.uint8)
    assert np.array_equal(to_size(big), cv2.resize(big, (224, 224), interpolation=cv2.INTER_AREA))
    for hw in ((238, 178), (180, 136)):  # có chiều < 224 -> CUBIC
        im = rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
        assert np.array_equal(to_size(im), cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC))


def test_transforms_resize_any_size():
    img = np.random.default_rng(2).integers(0, 255, (238, 178, 3), dtype=np.uint8)
    assert torch.equal(eval_transform(img), eval_transform(to_size(img)))
    x, _, _ = train_transform(img, np.random.default_rng(0), 0.5, 95)
    assert x.shape == (3, 224, 224)
    with pytest.raises(ValueError):
        eval_transform(np.zeros((224, 224), np.uint8))  # thiếu kênh màu


def test_train_transform_p_jpeg(monkeypatch):
    n = []
    real = io_utils.jpeg
    monkeypatch.setattr(io_utils, "jpeg", lambda im, q: (n.append(q), real(im, q))[1])
    img = np.zeros((224, 224, 3), np.uint8)
    for p, want in ((0.0, False), (1.0, True)):
        n.clear()
        x, j, _ = train_transform(img, np.random.default_rng(0), p, 95)
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


# ---- 3 arm ----
def test_arm_items_and_stats(data_root):
    raw = TrainList(data_root, "raw", 0, 0.5, 95)
    assert list(raw.items.columns) == ["path", "label", "source", "video_id"] and raw.aug_stats == {}
    aug = TrainList(data_root, "raw_aug", 0, 0.5, 95)
    n_seam = int((aug.items.source == "seam").sum())
    assert aug.aug_stats == {"n_rep": int((aug.items.source == "rep").sum()), "n_seam": n_seam} and n_seam > 0
    for t in (raw, aug):
        assert set(t[0]) == set(t[len(t) - 1]) == {"image", "label", "source", "jpeg"}


def test_rep_arm_is_raw_aug_without_seam(data_root):
    """raw_rep: mọi rep của manifest_aug (không lọc, không cân bằng), bỏ hết seam."""
    t = TrainList(data_root, "raw_rep", 0, 0.5, 95)
    u = TrainList(data_root, "raw_aug", 0, 0.5, 95)
    assert set(t.items.source) == {"raw", "rep"} and list(t.items.columns) == ["path", "label", "source", "video_id"]
    assert list(t.items.path) == list(u.items[u.items.source != "seam"].path)
    assert t.aug_stats == {"n_seam_dropped": u.aug_stats["n_seam"], "n_rep": u.aug_stats["n_rep"], "n_seam": 0}
    assert set(t[len(t) - 1]) == {"image", "label", "source", "jpeg"}


def test_trainlist_rejects_old_arms(data_root):
    for old in ("raw_aug_filt", "raw_aug_filt_loc", "raw_rep_filt"):
        with pytest.raises(ValueError):
            TrainList(data_root, old, 0, 0.5, 95)


def test_train_transform_rng_unchanged():
    img = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for seed in range(6):
        ref = np.random.default_rng(seed)
        did = bool(ref.random() < 0.5)
        want = io_utils.jpeg(img, 95) if did else img
        flip = ref.random() < 0.5
        want = eval_transform(want[:, ::-1] if flip else want)
        x, j, f = train_transform(img, np.random.default_rng(seed), 0.5, 95)
        assert j is did and f is bool(flip) and torch.equal(x, want)
