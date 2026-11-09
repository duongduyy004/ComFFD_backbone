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
    assert run_dir(cfg(mode="compffd", split="FF-03", seed=0, runs_root=tmp_path)) == tmp_path / "compffd/FF-03/seed0"


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
