"""Test Dataset theo arm (raw, raw+rep) và transforms. Trạng thái: chưa chạy thử. Dữ liệu: FF++ tổng hợp (MINH HỌA)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compsynth_ff import composite as comp_mod  # noqa: E402
from compsynth_ff.dataset import AugCore  # noqa: E402
from ffd.data.arms import ArmTrainDataset, build_train_dataset  # noqa: E402
from ffd.data.transforms import eval_transform, train_transform  # noqa: E402


def count_calls(monkeypatch, target_mod, name="jpeg"):
    orig, n = getattr(target_mod, name), [0]

    def wrapped(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(target_mod, name, wrapped)
    return lambda: n[0]


def _sources(ds, parity, n=2000):
    idx = range(parity, min(len(ds), 2 * n), 2)
    return [ds[i]["targets"] for i in idx]


def test_arm_balance_and_sources(ffpp_synth):
    raw = ArmTrainDataset(ffpp_synth, "raw", 0.5, seed=0)
    assert len(raw) == 2 * len(raw.real_frames)
    ys = [raw[i]["targets"]["L0"] for i in range(min(len(raw), 200))]
    assert ys[0::2] == [0] * len(ys[0::2]) and ys[1::2] == [1] * len(ys[1::2])
    assert {t["source"] for t in _sources(raw, 0, 100) + _sources(raw, 1, 100)} == {"raw"}
    rep = ArmTrainDataset(ffpp_synth, "raw+rep", 0.5, seed=0)
    fk, rl = [], []
    for e in range(10):  # synth chỉ ~64 frame real train: gom nhiều epoch để đủ mẫu
        rep.set_epoch(e)
        fk += _sources(rep, 1, 300)
        rl += _sources(rep, 0, 300)
    assert len(fk) >= 400
    assert abs(np.mean([t["source"] == "rep" for t in fk]) - 0.5) < 0.08
    assert abs(np.mean([t["source"] == "seam" for t in rl]) - 0.5) < 0.08
    assert all(t["L0"] == 1 for t in fk) and all(t["L0"] == 0 for t in rl)
    x = rep[1]["image"]
    assert x.shape == (3, 336, 336) and x.dtype == torch.float32
    assert set(rep[1]["targets"]) == {"L0", "source", "method", "region", "fallback"}


def test_jpeg_once(monkeypatch, ffpp_synth):
    calls = count_calls(monkeypatch, comp_mod)
    raw = ArmTrainDataset(ffpp_synth, "raw", 0.5, 0)
    for i in (0, 1, 3):
        n0 = calls()
        raw[i]
        assert calls() - n0 == 1
    rep = ArmTrainDataset(ffpp_synth, "raw+rep", 1.0, 0)
    for i in range(0, 12):
        n0 = calls()
        t = rep[i]["targets"]
        assert calls() - n0 == (1 if t["fallback"] else 0), t


def test_fallback_when_no_aug(ffpp_synth, monkeypatch):
    monkeypatch.setattr(AugCore, "rep", lambda self, t, f: [])
    monkeypatch.setattr(AugCore, "seam", lambda self, t, f: [])
    ds = ArmTrainDataset(ffpp_synth, "raw+rep", 1.0, 0)
    for i in range(6):
        t = ds[i]["targets"]
        assert t["fallback"] is True and t["source"] == ("seam" if i % 2 == 0 else "rep")


def test_deterministic_per_epoch(ffpp_synth):
    a = ArmTrainDataset(ffpp_synth, "raw+rep", 0.5, 0)
    b = ArmTrainDataset(ffpp_synth, "raw+rep", 0.5, 0)
    assert torch.equal(a[7]["image"], b[7]["image"])
    a.set_epoch(1)
    assert any(not torch.equal(a[i]["image"], b[i]["image"]) for i in range(7, 15))


def test_train_only(ffpp_synth):
    from compsynth_ff.select_cells import load_frames

    ds = ArmTrainDataset(ffpp_synth, "raw+rep", 0.5, 0)
    fr = load_frames(ffpp_synth)
    sub = fr.drop_duplicates("tid").set_index("tid").subset
    assert {sub[t] for t, _ in ds.real_frames} == {"train"}


def test_no_method_frame_raises(ffpp_synth, monkeypatch):
    ds = ArmTrainDataset(ffpp_synth, "raw", 0.5, 0)
    monkeypatch.setattr(ds, "_methods", {})
    with pytest.raises(RuntimeError):
        ds[1]


def test_transforms():
    img = np.full((336, 336, 3), 128, np.uint8)
    img[:, :, 0] = 0  # B=0, G=R=128
    out = train_transform(img, np.random.default_rng(0), already_jpeg=True)
    assert out.shape == (3, 336, 336) and out.dtype == torch.float32
    ev = eval_transform(img)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    exp = (torch.tensor([128.0, 128.0, 0.0]).view(3, 1, 1) / 255 - mean) / std  # đã đổi BGR->RGB
    assert torch.allclose(ev, exp.expand(3, 336, 336), atol=1e-4)


def test_flip_half():
    img = np.zeros((336, 336, 3), np.uint8)
    img[:, :100] = 255
    fl = [bool(train_transform(img, np.random.default_rng(i), already_jpeg=True)[0, 0, 0] > 0) for i in range(400)]
    assert abs(np.mean(fl) - 0.5) < 0.08


def test_build_train_dataset_arm(ffpp_synth, tmp_path):
    from ffd.config import TrainConfig

    cfg = TrainConfig(ffpp_config=tmp_path / "a.yaml", runs_root=tmp_path, mode="raw+rep", seed=3,
                      total_steps=1, eval_every=1, ckpt_every=1, rho=0.25)
    ds = build_train_dataset(cfg, ff_cfg=ffpp_synth)
    assert isinstance(ds, ArmTrainDataset) and ds.rho == 0.25 and ds.seed == 3 and ds.arm == "raw+rep"


def test_wrong_size_raises():
    with pytest.raises(ValueError):
        eval_transform(np.zeros((300, 300, 3), np.uint8))


def test_aug_only_where_both_kinds_exist(ffpp_synth, monkeypatch):
    """Final I1: frame chỉ có rep (seam bị skip) → mục fake chọn rep cũng phải fallback, để tỉ lệ ảnh tăng cường
    thực tế bằng nhau ở hai lớp. Ngược lại, frame chỉ có seam → mục real chọn seam fallback."""
    monkeypatch.setattr(AugCore, "seam", lambda self, t, f: [])
    ds = ArmTrainDataset(ffpp_synth, "raw+rep", 1.0, 0)
    for i in range(1, 12, 2):
        t = ds[i]["targets"]
        assert t["source"] == "rep" and t["fallback"] is True, t
    monkeypatch.undo()
    monkeypatch.setattr(AugCore, "rep", lambda self, t, f: [])
    ds = ArmTrainDataset(ffpp_synth, "raw+rep", 1.0, 0)
    for i in range(0, 12, 2):
        t = ds[i]["targets"]
        assert t["source"] == "seam" and t["fallback"] is True, t
