"""Test thư mục run checkpoints/clip/<run>. Trạng thái: chưa chạy thử trên GPU."""
import json
from pathlib import Path

import pytest

from ffd.config import TrainConfig, config_hash
from ffd.rundir import prepare_run_dir, run_dir, write_run_info

RESULTS = ["best.pt", "last.pt", "train_log.csv", "result.json", "scores_ffpp_test.parquet",
           "scores_celebdf.parquet", "run_info.json", "done.flag"]


def cfg(tmp_path, **kw):
    d = dict(data_root=Path("x"), ckpt_root=tmp_path / "ck", arm="raw", seed=0)
    d.update(kw)
    return TrainConfig(**d)


def test_run_dir_layout(tmp_path):
    assert run_dir(cfg(tmp_path)) == tmp_path / "ck" / "raw"
    assert run_dir(cfg(tmp_path, arm="raw_aug")) == tmp_path / "ck" / "raw_aug"
    assert run_dir(cfg(tmp_path, run_name="lr1e-4")) == tmp_path / "ck" / "lr1e-4"


def test_refuse_finished_run(tmp_path):
    c = cfg(tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    (d / "result.json").write_text("{}")
    for resume in (False, True):
        with pytest.raises(FileExistsError):
            prepare_run_dir(c, resume=resume, force=False)
    assert prepare_run_dir(c, resume=False, force=True) == d
    assert not (d / "result.json").exists()


@pytest.mark.parametrize("name", [n for n in RESULTS if n != "result.json"])
def test_leftovers_need_resume_or_force(tmp_path, name):
    c = cfg(tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    (d / name).write_text("x")
    with pytest.raises(FileExistsError):
        prepare_run_dir(c, resume=False, force=False)
    prepare_run_dir(c, resume=False, force=True)
    assert not (d / name).exists()


def test_force_removes_all_results(tmp_path):
    c = cfg(tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    for n in RESULTS + ["error.json", "keep.txt"]:
        (d / n).write_text("x")
    prepare_run_dir(c, resume=False, force=True)
    assert sorted(p.name for p in d.iterdir()) == ["keep.txt"]


def test_resume_keeps_files_and_checks_hash(tmp_path):
    c = cfg(tmp_path)
    d = prepare_run_dir(c, resume=False, force=False)
    write_run_info(d, c, {})
    (d / "last.pt").write_text("x")
    assert prepare_run_dir(c, resume=True, force=False) == d
    assert (d / "last.pt").exists()
    with pytest.raises(ValueError, match="config_hash"):
        prepare_run_dir(cfg(tmp_path, lr=1e-4), resume=True, force=False)


def test_write_run_info(tmp_path, data_root):
    c = cfg(tmp_path, data_root=data_root)
    write_run_info(tmp_path / "r", c, {"model_commit": "abc", "note": "x"})
    j = json.loads((tmp_path / "r/run_info.json").read_text(encoding="utf-8"))
    assert j["config_hash"] == config_hash(c) and j["model_commit"] == "abc" and j["note"] == "x"
    assert j["config"]["data_root"] == str(data_root) and j["config"]["arm"] == "raw"
    assert j["config"]["jpeg_q"] == 95 and "torch" in j["versions"] and "gpu" in j
    assert j["data_hashes"]["params_json_sha256"] and j["data_hashes"]["manifest_sha256"]["raw"]
    assert [p.name for p in (tmp_path / "r").iterdir()] == ["run_info.json"]
