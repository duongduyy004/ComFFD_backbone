"""Test so sánh mức video (ffd.videoboot) trên điểm tổng hợp (MINH HỌA)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from ffd import videoboot as V


def _scores(rng, mode: str, n_real=20, n_fake=30, n_frames=3) -> pd.DataFrame:
    rows = []
    for i, label in enumerate([0] * n_real + [1] * n_fake):
        for f in range(n_frames):
            if mode == "perfect":
                z = 5.0 + rng.normal() if label else -5.0 + rng.normal()
            else:
                z = rng.normal() + 0.3 * label
            rows.append({"video_id": f"v{i:03d}", "frame_idx": f, "label": label,
                         "method": "fake" if label else "real", "logit": z})
    return pd.DataFrame(rows)


def _write(root, arm, mode, seed=0):
    rng = np.random.default_rng(seed)
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    for ds in ("ffpp_test", "celebdf"):
        _scores(rng, mode).to_parquet(d / f"scores_{ds}.parquet", index=False)


def test_identical_arms_zero_ci(tmp_path):
    _write(tmp_path, "a", "noise")
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    p = V.compare_video(tmp_path, "a", ["b"], B=200)["celebdf"]
    q = p["pairs"]["a__b"]
    assert q["auc_diff"] == 0 and q["auc_ci95"] == [0.0, 0.0] and q["fpr_ci95"] == [0.0, 0.0]
    assert p["corr_real"]["b"] == pytest.approx(1.0)


def test_better_arm_positive(tmp_path):
    _write(tmp_path, "a", "noise")
    _write(tmp_path, "b", "perfect", seed=1)
    j = V.compare_video(tmp_path, "a", ["b"], B=300)
    q = j["celebdf"]["pairs"]["a__b"]
    assert q["auc_ci95"][0] > 0 and q["fpr_ci95"][1] < 0
    assert j["celebdf"]["auc"]["b"] == 1.0 and j["celebdf"]["n_real"] == 20 and j["celebdf"]["n_fake"] == 30


def test_fpr_at_tpr95_definition():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.95, 0.5, 0.6, 0.7, 0.8])
    assert V.fpr_at_tpr95(y, s) == 0.5  # TPR 1.0 cần ngưỡng 0.5 -> 2/4 real vượt


def test_mismatched_videos_raise(tmp_path):
    _write(tmp_path, "a", "noise")
    _write(tmp_path, "b", "noise", seed=1)
    p = tmp_path / "b" / "scores_celebdf.parquet"
    df = pd.read_parquet(p)
    df[df.video_id != "v000"].to_parquet(p, index=False)
    with pytest.raises(ValueError, match="video"):
        V.compare_video(tmp_path, "a", ["b"], B=10)


def test_missing_scores_raise(tmp_path):
    _write(tmp_path, "a", "noise")
    _write(tmp_path, "b", "noise", seed=1)
    (tmp_path / "b" / "scores_celebdf.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="scores_celebdf.parquet"):
        V.compare_video(tmp_path, "a", ["b"], B=10)


def test_writes_named_outputs_not_summary(tmp_path):
    for i, arm in enumerate(("a", "b", "c")):
        _write(tmp_path, arm, "noise", seed=i)
    j = V.compare_video(tmp_path, "a", ["b", "c"], B=100, name="vb_test")
    assert (tmp_path / "vb_test.json").is_file() and (tmp_path / "vb_test.md").is_file()
    assert not (tmp_path / "summary.md").exists() and set(j["celebdf"]["pairs"]) == {"a__b", "a__c"}
    assert j["B"] == 100 and j["seed"] == 0 and "Exploratory" in j["note"]
    assert json.loads((tmp_path / "vb_test.json").read_text(encoding="utf-8"))["B"] == 100


def test_default_name(tmp_path):
    _write(tmp_path, "a", "noise")
    _write(tmp_path, "b", "noise", seed=1)
    V.compare_video(tmp_path, "a", ["b"], B=10)
    assert (tmp_path / "videoboot_a.json").is_file()


def test_main_cli(tmp_path):
    _write(tmp_path, "raw", "noise")
    _write(tmp_path, "p", "noise", seed=1)
    assert V.main(["--ckpt-root", str(tmp_path), "--arms", "p", "--B", "20"]) == 0
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"data_root": str(tmp_path), "ckpt_root": str(tmp_path)}), encoding="utf-8")
    assert V.main(["--config", str(cfg), "--arms", "p", "--B", "20", "--name", "x"]) == 0
    assert (tmp_path / "x.md").is_file()
