"""Test Dataset theo arm, transforms, compffd. Trạng thái: chưa chạy thử. Dữ liệu: FF++ tổng hợp (MINH HỌA)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compsynth_ff import composite as comp_mod  # noqa: E402
from compsynth_ff.dataset import TrainCore  # noqa: E402
from ffd.data import arms  # noqa: E402
from ffd.data.arms import ArmTrainDataset, build_train_dataset  # noqa: E402
from ffd.data.transforms import eval_transform, train_transform  # noqa: E402


def count_calls(monkeypatch, target_mod, name="jpeg"):
    orig, n = getattr(target_mod, name), [0]

    def wrapped(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(target_mod, name, wrapped)
    return lambda: n[0]


def test_arm_dataset_balance_and_train_only(ffpp_synth):
    ds = ArmTrainDataset(ffpp_synth, "raw+sb", 0.5, seed=0)
    assert len(ds) == 2 * len(ds.core.real_frames)
    ys = [ds[i]["targets"]["L0"] for i in range(len(ds))]
    assert sum(ys) == len(ds) // 2 and ys[0::2] == [0] * (len(ds) // 2)
    assert {ds[i]["targets"]["source"] for i in range(1, len(ds), 2)} <= {"raw", "sb"}
    assert {ds[i]["targets"]["source"] for i in range(0, len(ds), 2)} == {"real"}
    x = ds[1]["image"]
    assert x.shape == (3, 336, 336) and x.dtype == torch.float32


@pytest.mark.parametrize("arm,rho", [("raw+comp", 1.0), ("raw+comp", 0.0), ("raw+sb", 1.0), ("raw", 0.5)])
def test_jpeg_applied_once_per_item(monkeypatch, ffpp_synth, arm, rho):
    # đếm tại comp_mod.jpeg: composite() và transforms đều tra tên này lúc gọi
    calls = count_calls(monkeypatch, comp_mod)
    ds = ArmTrainDataset(ffpp_synth, arm, rho, 0)
    for i in (0, 1, 3):
        n0 = calls()
        ds[i]
        assert calls() - n0 == 1, (arm, rho, i)


def test_deterministic_per_epoch(ffpp_synth):
    a = ArmTrainDataset(ffpp_synth, "raw+comp", 0.5, 0)
    b = ArmTrainDataset(ffpp_synth, "raw+comp", 0.5, 0)
    assert torch.equal(a[7]["image"], b[7]["image"])
    a.set_epoch(1)
    assert not torch.equal(a[7]["image"], b[7]["image"])


def test_arm_sources_by_rho(ffpp_synth):
    ds1 = ArmTrainDataset(ffpp_synth, "raw+comp", 1.0, 0)
    assert {ds1[i]["targets"]["source"] for i in range(1, len(ds1), 2)} == {"comp"}
    ds0 = ArmTrainDataset(ffpp_synth, "raw+sb", 0.0, 0)
    assert {ds0[i]["targets"]["source"] for i in range(1, len(ds0), 2)} == {"raw"}


def test_comp_and_sb_use_same_sigma_q_sampler(monkeypatch, ffpp_synth):
    seen = {}
    orig = arms.sample_sigma_q
    for arm in ("raw+comp", "raw+sb"):
        calls = []

        def spy(rng, ed, cfg, _c=calls):
            _c.append((ed, tuple(cfg.u_range), tuple(cfg.jpeg_q)))
            return orig(rng, ed, cfg)

        monkeypatch.setattr(arms, "sample_sigma_q", spy)
        ds = ArmTrainDataset(ffpp_synth, arm, 1.0, 0)
        for i in range(1, 9, 2):
            ds[i]
        seen[arm] = calls
    assert len(seen["raw+comp"]) == len(seen["raw+sb"]) == 4
    assert set(seen["raw+comp"]) == set(seen["raw+sb"])


def test_sb_items_have_region_and_borrowed_method(ffpp_synth):
    ds = ArmTrainDataset(ffpp_synth, "raw+sb", 1.0, 0)
    for i in range(1, min(len(ds), 40), 2):
        t = ds[i]["targets"]
        assert t["source"] == "sb" and t["region"] and t["method"]


def test_no_valid_frame_raises(ffpp_synth, monkeypatch):
    ds = ArmTrainDataset(ffpp_synth, "raw+comp", 1.0, 0)
    monkeypatch.setattr(ds, "_frame_info", lambda t, f: ((), {}))
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


def test_compffd_mode(ffpp_synth, tmp_path):
    core = TrainCore(ffpp_synth)
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps({"seen": sorted(map(list, core.allowed)), "unseen": []}), encoding="utf-8")
    from ffd.config import TrainConfig

    cfg = TrainConfig(ffpp_config=tmp_path / "ff.yaml", runs_root=tmp_path, mode="compffd", seed=0,
                      total_steps=1, eval_every=1, ckpt_every=1, split=str(sp))
    ds = build_train_dataset(cfg, ff_cfg=ffpp_synth)
    assert len(ds) == 2 * len(core.real_frames)
    it = ds[1]
    assert it["image"].shape == (3, 336, 336) and it["targets"]["L0"] == 1
    assert it["targets"]["source"] == "compffd" and ds[0]["targets"]["L0"] == 0
    ds.set_epoch(1)


def test_build_train_dataset_arm(ffpp_synth, tmp_path):
    from ffd.config import TrainConfig

    cfg = TrainConfig(ffpp_config=tmp_path / "a.yaml", runs_root=tmp_path, mode="raw+sb", seed=3,
                      total_steps=1, eval_every=1, ckpt_every=1, rho=0.25)
    ds = build_train_dataset(cfg, ff_cfg=ffpp_synth)
    assert isinstance(ds, ArmTrainDataset) and ds.rho == 0.25 and ds.seed == 3


def test_wrong_size_raises():
    with pytest.raises(ValueError):
        eval_transform(np.zeros((300, 300, 3), np.uint8))
    with pytest.raises(ValueError):
        train_transform(np.zeros((300, 300, 3), np.uint8), np.random.default_rng(0), already_jpeg=True)
