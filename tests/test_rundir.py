"""Test run dir. Trạng thái: chưa chạy thử."""
import json
from pathlib import Path

import pytest

from ffd.config import TrainConfig, config_hash
from ffd.rundir import prepare_run_dir, run_dir, write_run_info


def cfg(**kw):
    d = dict(ffpp_root=Path("x"), runs_root=Path("runs"), mode="raw", seed=0,
             total_steps=10, eval_every=5, ckpt_every=5)
    d.update(kw)
    return TrainConfig(**d)


def test_run_dir_layout(tmp_path):
    assert run_dir(cfg(mode="raw+rep", seed=2, runs_root=tmp_path)) == tmp_path / "aug/raw+rep/seed2"


def test_refuse_finished_run(tmp_path):
    d = prepare_run_dir(c := cfg(runs_root=tmp_path), resume=False, force=False)
    (d / "best.pt").write_bytes(b"x"); (d / "done.flag").touch()
    with pytest.raises(FileExistsError):
        prepare_run_dir(c, resume=False, force=False)
    assert prepare_run_dir(c, resume=False, force=True) == d


def test_resume_hash_mismatch(tmp_path):
    c = cfg(runs_root=tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    write_run_info(d, c, {})
    assert prepare_run_dir(c, resume=True, force=False) == d
    with pytest.raises(ValueError):
        prepare_run_dir(cfg(runs_root=tmp_path, rho=0.7), resume=True, force=False)


def test_write_run_info(tmp_path):
    sp = tmp_path / "split.json"; sp.write_text("{}")
    c = cfg(runs_root=tmp_path)
    write_run_info(tmp_path / "r", c, {"split_file": sp, "note": "x"})
    j = json.loads((tmp_path / "r/run_info.json").read_text(encoding="utf-8"))
    assert j["config_hash"] == config_hash(c) and j["seed"] == 0 and j["mode"] == "raw"
    assert j["split_hash"] and j["prereg_hash"] is None and "torch" in j["versions"] and j["note"] == "x"


def test_run_info_atomic_roundtrip(tmp_path):
    d = tmp_path / "r"
    write_run_info(d, cfg(runs_root=tmp_path), {})
    assert json.loads((d / "run_info.json").read_text(encoding="utf-8"))["mode"] == "raw"
    assert [p.name for p in d.iterdir()] == ["run_info.json"]


def test_force_removes_stale_artifacts_but_resume_keeps(tmp_path):
    c = cfg(runs_root=tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    stale = ["done.flag", "best.pt", "last.pt", "val_curve.jsonl", "train_log.jsonl", "error.json"]
    for n in stale:
        (d / n).write_text("x")
    write_run_info(d, c, {})
    (d / "done.flag").unlink()
    assert prepare_run_dir(c, resume=True, force=False) == d
    assert all((d / n).exists() for n in stale if n != "done.flag")
    (d / "done.flag").write_text("x")
    prepare_run_dir(c, resume=False, force=True)
    assert not any((d / n).exists() for n in stale)
    assert (d / "run_info.json").exists()


def test_stale_artifacts_without_resume_or_force_refused(tmp_path):
    c = cfg(runs_root=tmp_path)
    for n in ("last.pt", "best.pt", "val_curve.jsonl", "train_log.jsonl", "error.json"):
        d = prepare_run_dir(c, resume=False, force=False)
        (d / n).write_text("x")
        with pytest.raises(FileExistsError):
            prepare_run_dir(c, resume=False, force=False)
        assert prepare_run_dir(c, resume=True, force=False) == d
        prepare_run_dir(c, resume=False, force=True)
        assert not (d / n).exists()


def test_run_info_ffpp_hashes(tmp_path, ffpp_synth):
    """run_info ghi config_hash của ffpp_raw/ và ffpp_aug/manifest (preprocessing ghi, detector không tính lại);
    manifest không có -> None."""
    from synth_ffpp import AUG_HASH, RAW_HASH

    write_run_info(tmp_path / "r", cfg(runs_root=tmp_path, ffpp_root=ffpp_synth), {"model_commit": "abc"})
    j = json.loads((tmp_path / "r/run_info.json").read_text(encoding="utf-8"))
    assert j["ffpp_raw_hash"] == RAW_HASH and j["ffpp_aug_hash"] == AUG_HASH
    assert j["model_commit"] == "abc" and j["model_revision"] is None
    write_run_info(tmp_path / "r2", cfg(runs_root=tmp_path, ffpp_root=tmp_path / "none"), {})
    j2 = json.loads((tmp_path / "r2/run_info.json").read_text(encoding="utf-8"))
    assert j2["ffpp_raw_hash"] is None and j2["ffpp_aug_hash"] is None


def test_manifest_with_mixed_hash_rejected(tmp_path, ffpp_synth):
    import shutil

    import pandas as pd

    from ffd.data.ffpp import StaleArtifactError, manifest_hashes

    root = tmp_path / "o"
    shutil.copytree(ffpp_synth / "ffpp_raw", root / "ffpp_raw", ignore=shutil.ignore_patterns("*.png"))
    mp = root / "ffpp_raw" / "manifest.parquet"
    man = pd.read_parquet(mp)
    man.loc[0, "config_hash"] = "khac"
    man.to_parquet(mp, index=False)
    with pytest.raises(StaleArtifactError, match="config_hash"):
        manifest_hashes(root)
    man.drop(columns="keep").to_parquet(mp, index=False)
    with pytest.raises(StaleArtifactError, match="keep"):
        manifest_hashes(root)


def test_force_removes_scores_and_eval_info(tmp_path):
    """I6: scores_*.parquet và eval_info.json là kết quả cũ: không force thì từ chối, force thì xóa."""
    c = cfg(runs_root=tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    names = ["scores_ffpp_test.parquet", "scores_cdf2.parquet", "eval_info.json"]
    for n in names:
        (d / n).write_text("x")
    with pytest.raises(FileExistsError):
        prepare_run_dir(c, resume=False, force=False)
    prepare_run_dir(c, resume=False, force=True)
    assert not any((d / n).exists() for n in names)
