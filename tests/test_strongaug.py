"""Test arm raw_strongaug: suy giảm chất lượng cho cả hai lớp, arm cũ không đổi. Dữ liệu MINH HỌA (synth_ffpp)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from ffd import io_utils  # noqa: E402
from ffd.config import ALL_ARMS, ARMS, STRONG_ARMS, config_hash, load_train_config  # noqa: E402
from ffd.data.augfilter import arm_constants, check_arm_constants  # noqa: E402
from ffd.data.trainset import TrainList  # noqa: E402
from ffd.data.transforms import STRONG, eval_transform, strong_train_transform, train_transform  # noqa: E402


def _img(seed=1):
    return np.random.default_rng(seed).integers(0, 255, (224, 224, 3), dtype=np.uint8)


def test_arm_sets():
    assert ARMS == ("raw", "raw_aug", "raw_aug_filt", "raw_aug_filt_loc")  # main không đổi
    assert STRONG_ARMS == ("raw_strongaug",) and ALL_ARMS == ARMS + STRONG_ARMS


def test_config_accepts_arm_and_hash_differs(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(f"data_root: {tmp_path.as_posix()}\n", encoding="utf-8")
    a = load_train_config(p, {"arm": "raw"})
    b = load_train_config(p, {"arm": "raw_strongaug"})
    assert b.arm == "raw_strongaug" and config_hash(a) != config_hash(b)
    with pytest.raises(ValueError):
        load_train_config(p, {"arm": "raw_strongaug_typo"})


def test_arm_constants_strong():
    c = arm_constants("raw_strongaug")
    assert set(c) == {"strong"} and c["strong"]["jpeg_q_min"] == 40
    assert c["strong"]["resize_scale"] == [0.4, 1.0] and c["strong"]["blur_sigma"] == [0.1, 1.5]
    assert arm_constants("raw") == {} and arm_constants("raw_aug") == {}
    check_arm_constants("raw_strongaug", {"arm_constants": c}, "test")
    for bad in ({}, {"arm_constants": {"strong": {**c["strong"], "p_blur": 0.1}}}):
        with pytest.raises(ValueError):
            check_arm_constants("raw_strongaug", bad, "test")
    check_arm_constants("raw", {}, "test")  # arm cũ không kiểm


def test_strong_transform_shape_types_and_determinism():
    img = _img()
    x, j, f = strong_train_transform(img, np.random.default_rng(3), 0.5, 95)
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and isinstance(j, bool) and isinstance(f, bool)
    y, j2, f2 = strong_train_transform(img, np.random.default_rng(3), 0.5, 95)
    assert torch.equal(x, y) and (j, f) == (j2, f2)
    assert (img == _img()).all()  # đầu vào không bị sửa tại chỗ


def test_strong_transform_rng_chain_independent_of_p_jpeg():
    """Số mẫu ngẫu nhiên bốc mỗi ảnh cố định (9), nên rng sau ảnh không phụ thuộc p_jpeg."""
    img = _img()
    nxt = []
    for p in (0.0, 1.0):
        rng = np.random.default_rng(5)
        strong_train_transform(img, rng, p, 95)
        nxt.append(rng.random())
    assert nxt[0] == nxt[1]


def test_strong_jpeg_range_and_p_jpeg(monkeypatch):
    qs = []
    real = io_utils.jpeg
    monkeypatch.setattr(io_utils, "jpeg", lambda im, q: (qs.append(q), real(im, q))[1])
    img = _img()
    for s in range(300):
        _, j, _ = strong_train_transform(img, np.random.default_rng(s), 1.0, 95)
        assert j
    assert len(qs) == 300 and min(qs) >= STRONG["jpeg_q_min"] and max(qs) <= 95 and len(set(qs)) > 30
    qs.clear()
    for s in range(50):
        _, j, _ = strong_train_transform(img, np.random.default_rng(s), 0.0, 95)
        assert not j
    assert qs == []
    with pytest.raises(ValueError):
        strong_train_transform(img, np.random.default_rng(0), 1.0, STRONG["jpeg_q_min"] - 1)


def test_strong_degrades_more_than_raw_transform():
    """Trung bình, ảnh strong mất nhiều chi tiết tần số cao hơn ảnh arm raw (độ lệch Laplacian nhỏ hơn)."""
    img = _img()

    def lap(t):  # t đã chuẩn hóa; Laplacian trên kênh xám
        g = t.mean(0).numpy()
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    strong = np.mean([lap(strong_train_transform(img, np.random.default_rng(s), 0.5, 95)[0]) for s in range(60)])
    base = np.mean([lap(train_transform(img, np.random.default_rng(s), 0.5, 95)[0]) for s in range(60)])
    assert strong < 0.7 * base


def test_strong_rejects_non_224():
    with pytest.raises(ValueError):
        strong_train_transform(np.zeros((336, 336, 3), np.uint8), np.random.default_rng(0), 0.5, 95)


def test_eval_transform_unchanged_clean():
    img = _img()
    x = eval_transform(img)
    assert torch.equal(x, eval_transform(img.copy()))  # eval không suy giảm, tất định


def test_trainlist_strong_is_raw_only_same_items(data_root):
    raw = TrainList(data_root, "raw", 0, 0.5, 95)
    st = TrainList(data_root, "raw_strongaug", 0, 0.5, 95)
    assert st.items.equals(raw.items) and st.aug_stats == {}
    assert set(st.items.source) == {"raw"} and set(st.items.label) == {0, 1}
    assert set(st[0]) == {"image", "label", "source", "jpeg"} and st[0]["image"].shape == (3, 224, 224)


def test_trainlist_strong_differs_from_raw_images_but_deterministic(data_root):
    raw = TrainList(data_root, "raw", 0, 0.5, 95)
    a = TrainList(data_root, "raw_strongaug", 0, 0.5, 95)
    b = TrainList(data_root, "raw_strongaug", 0, 0.5, 95)
    a.set_epoch(2)
    b.set_epoch(2)
    assert torch.equal(a[3]["image"], b[3]["image"])
    assert any(not torch.equal(raw[i]["image"], a[i]["image"]) for i in range(len(a)))


def test_trainlist_strong_does_not_read_aug(data_root):
    """raw_strongaug chỉ đọc raw train: không gọi allow_read cho ffpp_aug."""
    from ffd import guards

    seen = []
    orig = guards.allow_read
    import ffd.data.trainset as ts

    ts.allow_read = lambda *a: (seen.append(a), orig(*a))[1]
    try:
        TrainList(data_root, "raw_strongaug", 0, 0.5, 95)
    finally:
        ts.allow_read = orig
    assert ("train", "ffpp_train") in seen and ("train", "ffpp_aug") not in seen


def test_train_transform_unchanged_for_old_arms(data_root):
    """Chuỗi rng và đầu ra của train_transform (arm raw/raw_aug) không đổi sau khi thêm strong."""
    img = _img()
    for seed in range(6):
        ref = np.random.default_rng(seed)
        did = bool(ref.random() < 0.5)
        want = io_utils.jpeg(img, 95) if did else img
        flip = ref.random() < 0.5
        want = eval_transform(want[:, ::-1] if flip else want)
        x, j, f = train_transform(img, np.random.default_rng(seed), 0.5, 95)
        assert j is did and f is bool(flip) and torch.equal(x, want)
