"""Test hợp đồng dữ liệu v2, TrainList cố định, transform 224. Dữ liệu MINH HỌA (synth_ffpp)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

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


# ---- arm filt/loc, nhãn patch (Task 2) ----
G = 16


def _pool(mask_path, g=G):
    m = (io_utils.read_image(mask_path).max(axis=2) > 0).astype(np.float32)
    c = m.shape[0] // g
    return m.reshape(g, c, g, c).mean(axis=(1, 3))


def _loc(data_root, seed=0):
    return TrainList(data_root, "raw_aug_filt_loc", seed, 0.0, 95, grid=G)


def test_old_arms_unchanged(data_root):
    for arm in ("raw", "raw_aug"):
        t = TrainList(data_root, arm, 0, 0.5, 95)
        assert list(t.items.columns) == ["path", "label", "source", "video_id"] and t.aug_stats == {}
        assert set(t[0]) == {"image", "label", "source", "jpeg"}
        assert set(t[len(t) - 1]) == {"image", "label", "source", "jpeg"}


def test_filt_arm_applies_filter(data_root):
    t = TrainList(data_root, "raw_aug_filt", 0, 0.5, 95)
    assert "mask_path" in t.items.columns and set(t[0]) == {"image", "label", "source", "jpeg"}
    st = t.aug_stats
    assert st["tau_m"] == pytest.approx(0.05) and st["n_rep_kept"] > 0 and st["n_seam_after"] > 0
    aug = aug_frames(data_root).set_index("path")
    a = t.items[t.items.source != "raw"]
    assert len(a) == st["n_rep_after"] + st["n_seam_after"]
    assert (aug.loc[a.path, "r"] >= 0.75).all() and (aug.loc[a.path, "mask_frac_final"] >= st["tau_m"]).all()
    assert t.items[t.items.source != "rep"].mask_path.isna().all()


def test_rep_filt_arm_drops_seam_keeps_reps(data_root):
    t = TrainList(data_root, "raw_rep_filt", 0, 0.5, 95)
    u = TrainList(data_root, "raw_aug_filt", 0, 0.5, 95)
    assert set(t.items.source) == {"raw", "rep"}
    assert list(t.items[t.items.source == "rep"].path) == list(u.items[u.items.source == "rep"].path)
    assert list(t.items[t.items.source == "raw"].path) == list(u.items[u.items.source == "raw"].path)
    assert t.aug_stats == {**u.aug_stats, "n_seam_dropped": u.aug_stats["n_seam_after"]}
    assert t.aug_stats["n_seam_dropped"] > 0
    assert t.items[t.items.source == "rep"].mask_path.notna().all()
    assert set(t[len(t) - 1]) == {"image", "label", "source", "jpeg"}  # không có nhãn patch


def test_loc_item_targets(data_root):
    t = _loc(data_root)
    seen = set()
    for i in range(len(t)):
        it, row = t[i], t.items.iloc[i]
        assert it["patch_target"].shape == (G, G) and it["patch_target"].dtype == torch.float32
        if row.source == "rep":
            assert it["has_loc"] is True and (it["patch_target"] > 0).any()
            assert it["patch_target"].sum().item() == pytest.approx(
                (io_utils.read_image(row.mask_path).max(axis=2) > 0).mean() * G * G, abs=1e-3)
        elif row.source == "seam" or row.label == 0:
            assert it["has_loc"] is True and not it["patch_target"].any()
        else:
            assert it["has_loc"] is False and not it["patch_target"].any()
        seen.add((row.source, int(row.label), it["has_loc"]))
    assert seen == {("rep", 1, True), ("seam", 0, True), ("raw", 0, True), ("raw", 1, False)}


def test_loc_mask_flipped_with_image(data_root):
    t = _loc(data_root)
    i = int(t.items.index[t.items.source == "rep"][0])
    got = {}
    for seed in range(40):  # tìm seed cho cả hai hướng lật, đối chiếu ảnh để biết đã lật hay chưa
        tt = _loc(data_root, seed)
        it = tt[i]
        raw = eval_transform(io_utils.read_image(tt._paths[i]))
        flipped = not torch.equal(it["image"], raw)
        got[flipped] = (it["patch_target"], _pool(tt.items.mask_path[i]))
        if len(got) == 2:
            break
    assert set(got) == {True, False}
    assert not np.array_equal(got[True][1], np.fliplr(got[True][1]))  # mask bất đối xứng, test có nghĩa
    assert torch.equal(got[True][0], torch.from_numpy(np.fliplr(got[True][1]).copy()))
    assert torch.equal(got[False][0], torch.from_numpy(got[False][1]))


def test_loc_missing_mask_fails_early(data_root, tmp_path):
    root = _copy(data_root, tmp_path / "r")
    kept = _loc(root).items
    victim = Path(kept[kept.source == "rep"].mask_path.iloc[0])
    victim.unlink()
    with pytest.raises(FileNotFoundError, match=victim.name):
        _loc(root)
    TrainList(root, "raw_aug_filt", 0, 0.0, 95)  # arm không loc không đọc mask


def test_loc_grid_not_divisible(data_root):
    with pytest.raises(ValueError, match="lưới"):  # lỗi lúc dựng, không phải giữa epoch
        TrainList(data_root, "raw_aug_filt_loc", 0, 0.0, 95, grid=15)
    TrainList(data_root, "raw_aug_filt", 0, 0.0, 95, grid=15)  # arm không loc không đọc mask


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
