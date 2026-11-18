"""Test ffd.realfp (tỉ lệ real bị đoán fake ở ngưỡng FF++ val, bootstrap theo video).
Dữ liệu: điểm tự sinh trong test (MINH HỌA), không phải kết quả thật."""
import json

import numpy as np
import pandas as pd
import pytest

from ffd import realfp as R


def _df(videos):
    """videos: list (video_id, label, [logits]) -> DataFrame cột như scores_*.parquet."""
    rows = [(v, i, lab, "real" if lab == 0 else "Deepfakes", float(x))
            for v, lab, xs in videos for i, x in enumerate(xs)]
    return pd.DataFrame(rows, columns=["video_id", "frame_idx", "label", "method", "logit"])


def _write_run(root, arm, val, ffpp_test, celebdf, skip=()):
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    for ds, df in (("val", val), ("ffpp_test", ffpp_test), ("celebdf", celebdf)):
        if ds not in skip:
            df.to_parquet(d / f"scores_{ds}.parquet", index=False)


def _sets(rng, shift, celeb_ids):
    """Một run giả: val, ffpp_test, celebdf; `shift` cộng vào logit real (dương = real bị đoán fake nhiều hơn)."""
    def mk(ids_real, ids_fake):
        return _df([(v, 0, list(rng.normal(-2 + shift, 1, 5))) for v in ids_real] +
                   [(v, 1, list(rng.normal(3, 1, 5))) for v in ids_fake])
    return (mk([f"r{i}" for i in range(10)], [f"f{i}" for i in range(10)]),
            mk([f"tr{i}" for i in range(10)], [f"tf{i}" for i in range(10)]),
            mk(celeb_ids, [f"Celeb-synthesis__{i:03d}" for i in range(10)]))


CELEB = [f"Celeb-real__{i:03d}" for i in range(6)] + [f"YouTube-real__{i:03d}" for i in range(6)]


def _two_runs(root, celeb_ids=CELEB):
    _write_run(root, "x", *_sets(np.random.default_rng(1), 0.0, celeb_ids))
    _write_run(root, "y", *_sets(np.random.default_rng(2), 3.0, celeb_ids))


def test_val_threshold_tpr95():
    df = _df([(f"f{i}", 1, [float(i)]) for i in range(100)] + [("r0", 0, [-5.0])])
    t = R.val_threshold(df)
    assert t == 5.0 and (df[df.label == 1].logit >= t).mean() == 0.95


def test_val_threshold_video_level():
    # 10 video fake -> chỉ số floor(0.05 * 10) = 0: video 'fa' (trung bình 5); 11 frame -> floor(0.55) = 0: frame 0.0
    df = _df([("fa", 1, [0.0, 10.0])] + [(f"f{i}", 1, [100.0]) for i in range(9)] + [("r", 0, [0.0])])
    assert R.val_threshold(df, level="video") == 5.0 and R.val_threshold(df, level="frame") == 0.0


def test_real_fp_rate_counts_ties_as_fake():
    assert R.real_fp_rate(_df([("r", 0, [0.0, 1.0, 2.0]), ("f", 1, [9.0])]), 1.0) == pytest.approx(2 / 3)


def test_video_table():
    v = R.video_table(_df([("a", 0, [1.0, 3.0]), ("b", 1, [5.0])]))
    assert list(v.index) == ["a", "b"] and list(v.s) == [2.0, 5.0] and list(v.n) == [2, 1] and list(v.label) == [0, 1]


def test_boot_identical_arms_zero():
    df = _df([("r1", 0, [-1.0, 0.5]), ("r2", 0, [2.0]), ("f", 1, [5.0])])
    out = R.boot_real_fp_diff(df, df, 0.0, 0.0, B=200)
    assert out["diff"] == 0 and out["ci95"] == [0.0, 0.0]


def test_boot_frame_weighted_and_shift():
    a = _df([("r1", 0, [-1.0]), ("r2", 0, [1.0] * 9), ("f", 1, [5.0])])
    b = _df([("r1", 0, [-1.0]), ("r2", 0, [-1.0] * 9), ("f", 1, [5.0])])
    out = R.boot_real_fp_diff(a, b, 0.0, 0.0, B=500)
    assert out["rate_a"] == pytest.approx(0.9) and out["rate_b"] == 0.0 and out["diff"] == pytest.approx(-0.9)
    assert out["ci95"][1] <= 0


def test_boot_video_level():
    a = _df([("r1", 0, [-1.0]), ("r2", 0, [1.0] * 9), ("f", 1, [5.0])])
    out = R.boot_real_fp_diff(a, a, 0.0, 0.0, B=50, level="video")
    assert out["rate_a"] == 0.5  # 1/2 video real có trung bình >= 0


def test_boot_auc_video_diff():
    a = _df([("r1", 0, [0.0]), ("r2", 0, [2.0]), ("f1", 1, [1.0]), ("f2", 1, [3.0])])
    b = _df([("r1", 0, [0.0]), ("r2", 0, [0.5]), ("f1", 1, [1.0]), ("f2", 1, [3.0])])
    out = R.boot_auc_video_diff(a, b, B=200)
    assert out["auc_a"] == 0.75 and out["auc_b"] == 1.0 and out["diff"] == 0.25 and len(out["ci95"]) == 2


def test_verdict():
    assert R.verdict([-0.1, -0.01]) == "seam_hurts_real"
    assert R.verdict([-0.1, 0.02]) == "inconclusive"
    assert R.verdict([0.01, 0.1]) == "seam_helps_real"


def test_mismatched_video_sets_raise():
    a = _df([("r1", 0, [0.0]), ("r2", 0, [1.0]), ("f", 1, [5.0])])
    b = _df([("r1", 0, [0.0]), ("f", 1, [5.0])])
    with pytest.raises(ValueError, match="video"):
        R.boot_real_fp_diff(a, b, 0.0, 0.0, B=10)
    with pytest.raises(ValueError, match="video"):
        R.boot_auc_video_diff(a, b, B=10)


def test_missing_val_scores_raises(tmp_path):
    rng = np.random.default_rng(0)
    _write_run(tmp_path, "x", *_sets(rng, 0.0, CELEB))
    _write_run(tmp_path, "y", *_sets(rng, 0.0, CELEB), skip=("val",))
    with pytest.raises(FileNotFoundError, match="scores_val.parquet"):
        R.compare_real_fp(tmp_path, "x", "y", B=10)


def test_compare_writes_outputs(tmp_path):
    _two_runs(tmp_path)
    j = R.compare_real_fp(tmp_path, "x", "y", B=200)
    assert (tmp_path / "realfp_x__y.json").is_file() and (tmp_path / "realfp_x__y.md").is_file()
    assert json.loads((tmp_path / "realfp_x__y.json").read_text(encoding="utf-8")) == j
    assert set(j["thresholds"]) == {"x", "y"} and set(j["thresholds"]["x"]) == {"frame", "video"}
    p = j["primary"]
    assert set(p) >= {"rate_a", "rate_b", "diff", "ci95", "verdict"}
    assert p["verdict"] == "seam_helps_real" and p["diff"] > 0  # y có real dịch lên +3 -> bị đoán fake nhiều hơn
    s = j["secondary"]
    assert set(s) >= {"celebdf_video_fp", "ffpp_test_real_fp", "celebdf_auc_video", "ffpp_test_auc_video",
                      "celebdf_real_fp_logit0", "by_source", "inhomain_drop_gt_0_01"}
    assert set(s["by_source"]) == {"Celeb-real", "YouTube-real"} and set(s["celebdf_real_fp_logit0"]) == {"x", "y"}
    assert j["B"] == 200 and j["seed"] == 0 and j["a"] == "x" and j["b"] == "y"


def test_compare_without_source_prefix(tmp_path):
    _two_runs(tmp_path, celeb_ids=[f"real{i}" for i in range(8)])
    j = R.compare_real_fp(tmp_path, "x", "y", B=50)
    assert "by_source" not in j["secondary"]
    assert "không tách được theo nguồn" in (tmp_path / "realfp_x__y.md").read_text(encoding="utf-8")


def test_main_cli_ckpt_root(tmp_path):
    _two_runs(tmp_path)
    assert R.main(["--ckpt-root", str(tmp_path), "--a", "x", "--b", "y", "--B", "50"]) == 0
    assert (tmp_path / "realfp_x__y.json").is_file()


def test_main_cli_config(tmp_path):
    ck = tmp_path / "ck"
    _two_runs(ck)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"data_root: {tmp_path.as_posix()}\nckpt_root: {ck.as_posix()}\n", encoding="utf-8")
    assert R.main(["--config", str(cfg), "--a", "x", "--b", "y", "--B", "50"]) == 0
    assert (ck / "realfp_x__y.json").is_file()


def test_verdict_warns_for_non_default_pair(tmp_path):
    _two_runs(tmp_path)
    R.compare_real_fp(tmp_path, "x", "y", B=20)
    assert "chỉ có nghĩa cho cặp raw_aug_filt → raw_rep_filt" in (tmp_path / "realfp_x__y.md").read_text(encoding="utf-8")
    _write_run(tmp_path, "raw_aug_filt", *_sets(np.random.default_rng(1), 0.0, CELEB))
    _write_run(tmp_path, "raw_rep_filt", *_sets(np.random.default_rng(2), 0.0, CELEB))
    R.compare_real_fp(tmp_path, "raw_aug_filt", "raw_rep_filt", B=20)
    md = (tmp_path / "realfp_raw_aug_filt__raw_rep_filt.md").read_text(encoding="utf-8")
    assert "chỉ có nghĩa cho cặp" not in md
