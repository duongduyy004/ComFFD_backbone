"""Test probe trên cache đặc trưng tổng hợp (MINH HỌA). CPU."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from ffd import probe as P
from ffd.evaluate import _SCORE_COLS
from ffd.features import load_features
from synth_feats import make_feat_dir


def test_slots_for():
    assert P.slots_for("probe_last", 24, None) == (24,)
    assert P.slots_for("probe_multi", 24, None) == (7, 11, 15, 19, 23)
    assert P.slots_for("probe_gate", 24, None) == tuple(range(24))
    assert P.slots_for("probe_layer", 24, 12) == (11,)
    with pytest.raises(ValueError):
        P.slots_for("probe_layer", 24, 25)


def test_gate_alpha_init_uniform():
    h = P.ProbeHead(24, 8, gate=True)
    assert np.allclose(h.alpha, 1 / 24)


@pytest.mark.parametrize("gate", [False, True])
def test_linear_init_zero(gate):  # logistic regression lồi: khởi tạo 0 -> kết quả không phụ thuộc init ngẫu nhiên
    h = P.ProbeHead(5, 8, gate=gate)
    assert float(h.fc.weight.detach().abs().sum()) == 0.0 and float(h.fc.bias.detach().abs().sum()) == 0.0


def test_run_writes_scores_same_order(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    run = P.run_probe(fd, tmp_path / "ck", "probe_multi", seed=0)
    assert run.name == "probe_multi_s0"
    for ds in ("val", "ffpp_test", "celebdf"):
        s = pd.read_parquet(run / f"scores_{ds}.parquet")
        _, idx, _ = load_features(fd, ds)
        assert list(s.columns) == _SCORE_COLS and list(s.video_id) == list(idx.video_id)
        assert list(s.frame_idx) == list(idx.frame_idx) and np.isfinite(s.logit).all()
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    assert info["wd_selected"] in P.WD_GRID and info["slots"] == [7, 11, 15, 19, 23]
    log = pd.read_csv(run / "train_log.csv")
    assert set(log.wd) == set(P.WD_GRID)
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert {"val", "ffpp_test", "celebdf"} <= set(res) and "auc_video" in res["celebdf"]


def test_signal_found(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BATCH", 32)  # tập tổng hợp nhỏ: 5 bước/epoch thay vì 1 để kịp hội tụ
    run = P.run_probe(make_feat_dir(tmp_path / "f"), tmp_path / "ck", "probe_multi")
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["val"]["auc"] > 0.9


def test_deterministic_same_seed(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    a = P.run_probe(fd, tmp_path / "a", "probe_gate", seed=0)
    b = P.run_probe(fd, tmp_path / "b", "probe_gate", seed=0)
    assert pd.read_parquet(a / "scores_celebdf.parquet").logit.equals(pd.read_parquet(b / "scores_celebdf.parquet").logit)


def test_gate_writes_alpha(tmp_path):
    run = P.run_probe(make_feat_dir(tmp_path / "f"), tmp_path / "ck", "probe_gate")
    alpha = json.loads((run / "run_info.json").read_text(encoding="utf-8"))["alpha"]
    assert len(alpha) == 24 and abs(sum(alpha) - 1) < 1e-6


def test_test_sets_opened_only_after_selection(tmp_path, monkeypatch):
    calls, orig = [], P.load_features
    monkeypatch.setattr(P, "load_features", lambda fd, ds: calls.append(ds) or orig(fd, ds))
    monkeypatch.setattr(P, "_select_done_hook", lambda: calls.append("SELECTED"))
    P.run_probe(make_feat_dir(tmp_path / "f"), tmp_path / "ck", "probe_last")
    i = calls.index("SELECTED")
    assert "ffpp_test" not in calls[:i] and "celebdf" not in calls[:i]
    assert {"ffpp_test", "celebdf"} <= set(calls[i:])


def test_zero_variance_dim_finite(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    for ds in ("train", "val", "ffpp_test", "celebdf"):
        x = np.load(fd / ds / "cls.npy")
        x[:, :, 0] = 3.0
        np.save(fd / ds / "cls.npy", x)
    for arm in ("probe_multi", "probe_gate"):
        run = P.run_probe(fd, tmp_path / "ck", arm)
        assert np.isfinite(pd.read_parquet(run / "scores_celebdf.parquet").logit).all()


def test_incomplete_cache_refuses(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    (fd / "train" / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="chưa hoàn tất"):
        P.run_probe(fd, tmp_path / "ck", "probe_last")


def test_rerun_mismatch_refuses(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    run = P.run_probe(fd, tmp_path / "ck", "probe_last")
    p = run / "run_info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    info["constants"]["lr"] = 0.5
    p.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="hằng số"):
        P.run_probe(fd, tmp_path / "ck", "probe_last")


def test_existing_result_skipped_unless_force(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    run = P.run_probe(fd, tmp_path / "ck", "probe_last")
    mt = (run / "result.json").stat().st_mtime_ns
    P.run_probe(fd, tmp_path / "ck", "probe_last")
    assert (run / "result.json").stat().st_mtime_ns == mt
    P.run_probe(fd, tmp_path / "ck", "probe_last", force=True)
    assert (run / "result.json").stat().st_mtime_ns != mt


def test_layer_all_writes_table(tmp_path, monkeypatch):
    fd = make_feat_dir(tmp_path / "f")
    monkeypatch.setattr(P, "feat_dir_of", lambda cfg, root: fd)
    # 24 run: bỏ bootstrap 2000 lần của frame_block và giảm epoch cho nhanh (test chỉ kiểm bảng)
    monkeypatch.setattr(P, "frame_block", lambda df, n: {"auc_video": 0.5})
    monkeypatch.setattr(P, "MAX_EPOCHS", 2)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"data_root": str(tmp_path), "ckpt_root": str(tmp_path / "ck")}), encoding="utf-8")
    assert P.main(["--config", str(cfg), "--arm", "probe_layer", "--layer", "all", "--device", "cpu"]) == 0
    md = (tmp_path / "ck" / "probe_layers.md").read_text(encoding="utf-8")
    rows = [l for l in md.splitlines() if l.startswith("| ") and l[2:3].isdigit()]
    assert len(rows) == 24 and "chỉ mô tả" in md


def test_slot_source_preloads_small_selection():
    cls = np.random.default_rng(0).normal(size=(50, 25, 8)).astype(np.float16)
    src, sl = P._slot_source(cls, (7, 11), max_bytes=10**9)
    assert isinstance(src, np.ndarray) and src.shape == (50, 2, 8) and sl is None
    assert np.array_equal(src, cls[:, [7, 11]])
    src2, sl2 = P._slot_source(cls, (7, 11), max_bytes=10)
    assert src2 is cls and sl2 == [7, 11]


def test_norm_stats_matches_numpy_multichunk():
    rng = np.random.default_rng(0)
    cls = (300 + rng.normal(scale=0.5, size=(3000, 25, 8))).astype(np.float16)
    mean, std = P._norm_stats(cls, tuple(range(24)), chunk=256)
    ref = cls[:, :24].astype(np.float64)
    assert np.allclose(mean, ref.mean(0), atol=1e-4) and np.allclose(std, ref.std(0), rtol=1e-3)


def test_inconsistent_caches_refused(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    p = fd / "celebdf" / "meta.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["transform_id"] = "khac"
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="không nhất quán"):
        P.run_probe(fd, tmp_path / "ck", "probe_last")
    assert not (tmp_path / "ck" / "probe_last_s0").exists()


def test_nonfinite_cache_clear_error(tmp_path):
    fd = make_feat_dir(tmp_path / "f")
    x = np.load(fd / "train" / "cls.npy")
    x[0, 24, 0] = np.inf
    np.save(fd / "train" / "cls.npy", x)
    with pytest.raises(ValueError, match="không hữu hạn"):
        P.run_probe(fd, tmp_path / "ck", "probe_last")
