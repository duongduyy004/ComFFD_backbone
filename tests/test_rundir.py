"""Test run dir. Trạng thái: chưa chạy thử."""
import json
from pathlib import Path

import pytest

from ffd.config import TrainConfig, config_hash
from ffd.rundir import prepare_run_dir, run_dir, write_run_info


def cfg(**kw):
    d = dict(ffpp_config=Path("x.yaml"), runs_root=Path("runs"), mode="raw", seed=0,
             total_steps=10, eval_every=5, ckpt_every=5)
    d.update(kw)
    return TrainConfig(**d)


def test_run_dir_layout(tmp_path):
    assert run_dir(cfg(mode="raw+sb", seed=2, runs_root=tmp_path)) == tmp_path / "aug/raw+sb/seed2"


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


def _ff_yaml(path, data_root, out_root, **kw):
    import yaml

    d = {"data_root": str(data_root), "out_root": str(out_root), "splits_dir": f"{out_root}/sp", **kw}
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    return path


def test_run_info_ffpp_hashes(tmp_path):
    """R1: ffpp_config_hash băm Config đã parse (nạp từ file khi không truyền ff_cfg); không có file -> None."""
    from compsynth_ff.config import load_config, stage_hash
    from ffd.rundir import ffpp_config_hash

    ff = _ff_yaml(tmp_path / "ff.yaml", "F:/data", "F:/out")
    write_run_info(tmp_path / "r", cfg(runs_root=tmp_path, ffpp_config=ff), {"model_commit": "abc"})
    j = json.loads((tmp_path / "r/run_info.json").read_text(encoding="utf-8"))
    ffc = load_config(ff)
    assert j["ffpp_config_hash"] == ffpp_config_hash(ffc) and j["ffpp_stage_hash"] == stage_hash(ffc, "render")
    assert j["model_commit"] == "abc" and j["model_revision"] is None
    write_run_info(tmp_path / "r2", cfg(runs_root=tmp_path, ffpp_config=tmp_path / "none.yaml"), {})
    j2 = json.loads((tmp_path / "r2/run_info.json").read_text(encoding="utf-8"))
    assert j2["ffpp_config_hash"] is None and j2["ffpp_stage_hash"] is None


def test_ffpp_config_hash_ignores_paths_not_params(tmp_path):
    """R1: cùng tham số, khác đường dẫn/ổ đĩa/comment/thứ tự khóa -> cùng hash; đổi tham số -> khác hash."""
    from compsynth_ff.config import load_config
    from ffd.rundir import ffpp_config_hash

    a = load_config(_ff_yaml(tmp_path / "a.yaml", "F:/DeepFakeData/FF", "F:/DeepFakeData/out", crf=23))
    b_path = tmp_path / "b.yaml"
    b_path.write_text("# may GPU khac\ncrf: 23\nout_root: /mnt/data/out\ndata_root: /mnt/data/FF\n"
                      "splits_dir: /home/u/sp\n", encoding="utf-8")
    b = load_config(b_path)
    c = load_config(_ff_yaml(tmp_path / "c.yaml", "F:/DeepFakeData/FF", "F:/DeepFakeData/out", crf=24))
    assert ffpp_config_hash(a) == ffpp_config_hash(b) != ffpp_config_hash(c)


def test_force_removes_scores_and_eval_info(tmp_path):
    """I6: scores_*.parquet và eval_info.json là kết quả cũ: không force thì từ chối, force thì xóa."""
    c = cfg(runs_root=tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    names = ["scores_ffpp_test.parquet", "scores_compsynth_test.parquet", "scores_cdf2.parquet", "eval_info.json"]
    for n in names:
        (d / n).write_text("x")
    with pytest.raises(FileExistsError):
        prepare_run_dir(c, resume=False, force=False)
    prepare_run_dir(c, resume=False, force=True)
    assert not any((d / n).exists() for n in names)
