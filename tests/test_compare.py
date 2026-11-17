"""Test so sánh hai arm; dữ liệu MINH HỌA (logit tổng hợp, không train model)."""
import json

import numpy as np
import pandas as pd
import pytest

from ffd.compare import NOTE, PAIRS, compare_pair, compare_runs


def _scores(ids, seed, shift):
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(len(ids))])
    return pd.DataFrame({
        "video_id": [f"m/v{i}" for i in ids], "frame_idx": [0] * len(ids),
        "label": y, "method": "m", "logit": y * shift + rng.normal(0, 1, len(ids)),
    })


def _write_run(d, arm, celeb, shift):
    d.mkdir(parents=True)
    res = {"arm": arm, "total_steps": 100 if arm == "raw" else 150,
           "epochs_run": 10 if arm == "raw" else 12, "best_epoch": 5 if arm == "raw" else 7}
    seed = {"raw": 1, "raw_aug": 2, "raw_aug_filt": 3, "raw_aug_filt_loc": 4}[arm]
    for name in ("ffpp_test", "celebdf"):
        if name == "celebdf" and not celeb:
            res[name] = {"skipped": "không có data"}
            continue
        res[name] = {"auc": 0.5}
        _scores(range(80), seed, shift).to_parquet(d / f"scores_{name}.parquet")
    (d / "result.json").write_text(json.dumps(res))


@pytest.fixture
def two_runs(tmp_path):
    _write_run(tmp_path / "raw", "raw", True, 1.0)
    _write_run(tmp_path / "raw_aug", "raw_aug", True, 1.5)
    return tmp_path


@pytest.fixture
def two_runs_no_celeb(tmp_path):
    _write_run(tmp_path / "raw", "raw", False, 1.0)
    _write_run(tmp_path / "raw_aug", "raw_aug", True, 1.5)
    return tmp_path


def test_compare_json(two_runs):
    c = json.loads(compare_runs(two_runs).read_text())
    assert c["note"] == NOTE
    keys = {"auc_raw", "auc_raw_aug", "diff", "diff_ci95", "delong_p"}
    assert set(c["ffpp_test"]) == keys and set(c["celebdf"]) == keys
    assert c["ffpp_test"]["diff"] == pytest.approx(c["ffpp_test"]["auc_raw_aug"] - c["ffpp_test"]["auc_raw"])
    assert c["total_steps"] == {"raw": 100, "raw_aug": 150}
    assert c["epochs_run"] == {"raw": 10, "raw_aug": 12} and c["best_epoch"] == {"raw": 5, "raw_aug": 7}


def test_compare_rejects_mismatched_frames(two_runs):
    p = two_runs / "raw_aug" / "scores_ffpp_test.parquet"
    pd.read_parquet(p).iloc[1:].to_parquet(p)
    with pytest.raises(ValueError):
        compare_runs(two_runs)


def test_compare_rejects_label_mismatch(two_runs):
    p = two_runs / "raw_aug" / "scores_ffpp_test.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "label"] = 1 - df.loc[0, "label"]
    df.to_parquet(p)
    with pytest.raises(ValueError, match="label"):
        compare_runs(two_runs)


def test_compare_without_celebdf(two_runs_no_celeb):
    c = json.loads(compare_runs(two_runs_no_celeb).read_text())
    assert "skipped" in c["celebdf"] and "diff" in c["ffpp_test"]


@pytest.fixture
def runs4(tmp_path):
    for arm, shift in (("raw", 1.0), ("raw_aug", 1.5), ("raw_aug_filt", 2.0), ("raw_aug_filt_loc", 2.5)):
        _write_run(tmp_path / arm, arm, True, shift)
    return tmp_path


def test_pairs_constant():
    assert PAIRS == (("raw", "raw_aug_filt"), ("raw", "raw_aug_filt_loc"), ("raw_aug", "raw_aug_filt"),
                     ("raw_aug_filt", "raw_aug_filt_loc"))


def test_compare_pair_files(runs4):
    old = compare_runs(runs4).read_text()
    for a, b in PAIRS:
        p = compare_pair(runs4, a, b)
        assert p == runs4 / f"compare_{a}__{b}.json"
        c = json.loads(p.read_text())
        assert c["note"] == NOTE
        for ds in ("ffpp_test", "celebdf"):
            assert set(c[ds]) == {f"auc_{a}", f"auc_{b}", "diff", "diff_ci95", "delong_p"}
            assert c[ds]["diff"] == pytest.approx(c[ds][f"auc_{b}"] - c[ds][f"auc_{a}"])
        for k in ("total_steps", "epochs_run", "best_epoch"):
            assert set(c[k]) == {a, b}
    assert len(list(runs4.glob("compare_*__*.json"))) == 4
    assert compare_runs(runs4).read_text() == old  # compare.json không đổi
    assert set(json.loads(old)["ffpp_test"]) == {"auc_raw", "auc_raw_aug", "diff", "diff_ci95", "delong_p"}


def test_compare_pair_matches_compare_runs(runs4):
    """Cùng (raw, raw_aug): compare_pair cho đúng số của compare.json."""
    c = json.loads(compare_runs(runs4).read_text())
    q = json.loads(compare_pair(runs4, "raw", "raw_aug").read_text())
    assert q == c


def test_cli_compare_skips_missing_pairs(tmp_path, capsys):
    """Lệnh compare: ghi mọi cặp có đủ hai run, cặp thiếu run chỉ in ghi chú (không lỗi)."""
    from ffd.cli import build_parser

    for arm, shift in (("raw", 1.0), ("raw_aug", 1.5), ("raw_aug_filt", 2.0)):
        _write_run(tmp_path / arm, arm, True, shift)
    a = build_parser().parse_args(["compare", "--ckpt-root", str(tmp_path)])
    a.fn(a)
    out = capsys.readouterr().out
    assert (tmp_path / "compare.json").exists()
    assert (tmp_path / "compare_raw__raw_aug_filt.json").exists()
    assert (tmp_path / "compare_raw_aug__raw_aug_filt.json").exists()
    assert not (tmp_path / "compare_raw__raw_aug_filt_loc.json").exists()
    assert "Bỏ qua cặp raw vs raw_aug_filt_loc" in out and "Bỏ qua cặp raw_aug_filt vs raw_aug_filt_loc" in out
