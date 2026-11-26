"""Test CLI v2 (parse lệnh, kiểm tham số trước khi chạy). Trạng thái: chưa chạy thử trên GPU."""
import subprocess
import sys

import pytest

from ffd.cli import build_parser, parse_sets


def run_cli(args):
    return subprocess.run([sys.executable, "-m", "ffd.cli", *args], capture_output=True, text=True)


def test_cli_parses_commands():
    p = build_parser()
    for argv in (["main", "--config", "c.yaml", "--set", "lr=1e-4"],
                 ["train", "--config", "c.yaml", "--resume"], ["train", "--config", "c.yaml", "--force"],
                 ["eval", "--run", "r"], ["eval", "--run", "r", "--data-root", "d"],
                 ["eval", "--run", "r", "--datasets", "val", "--force", "--no-progress"],
                 ["smoke", "--config", "c.yaml", "--steps", "5"]):
        a = p.parse_args(argv)
        assert a.cmd == argv[0] and callable(a.fn)
    with pytest.raises(SystemExit):
        p.parse_args(["report", "--runs-root", "r"])  # lệnh cũ đã bỏ
    for old in (["compare", "--ckpt-root", "ck"], ["main-v2", "--config", "c.yaml"]):
        with pytest.raises(SystemExit):
            p.parse_args(old)


def test_no_pilot_command():
    """Lệnh pilot đã bỏ (16/11): parser từ chối, không còn trong danh sách lệnh."""
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["pilot", "--config", "c.yaml"])
    assert "pilot" not in p.format_help()
    r = run_cli(["pilot", "--config", "c.yaml"])
    assert r.returncode != 0 and "invalid choice" in r.stderr


def test_parse_set():
    assert parse_sets(["lr=1e-4", "seed=2", "arm=raw_aug", "num_workers=0", "grad_checkpointing=false",
                       "tap_layers=[8,16]"]) == \
        {"lr": 1e-4, "seed": 2, "arm": "raw_aug", "num_workers": 0, "grad_checkpointing": False, "tap_layers": [8, 16]}
    with pytest.raises(ValueError):
        parse_sets(["noequals"])


def _cfg(tmp_path):
    c = tmp_path / "c.yaml"
    c.write_text(f"data_root: {tmp_path.as_posix()}/data\nckpt_root: {tmp_path.as_posix()}/ck\n", encoding="utf-8")
    return c


def test_train_resume_and_force_rejected(tmp_path):
    r = run_cli(["train", "--config", str(_cfg(tmp_path)), "--resume", "--force"])
    assert r.returncode != 0 and "--resume" in r.stderr and "Traceback" not in r.stderr
    assert not (tmp_path / "ck").exists()


def test_bad_config_key_clean_exit(tmp_path):
    r = run_cli(["main", "--config", str(_cfg(tmp_path)), "--set", "lrr=1"])
    assert r.returncode != 0 and "lrr" in r.stderr and "Traceback" not in r.stderr


def test_malformed_set_clean_exit(tmp_path):
    r = run_cli(["train", "--config", str(_cfg(tmp_path)), "--set", "x=["])
    assert r.returncode != 0 and "Traceback" not in r.stderr


def test_main_without_celebdf_clean_exit(tmp_path):
    r = run_cli(["main", "--config", str(_cfg(tmp_path))])
    assert r.returncode != 0 and "celebdf" in r.stderr and "Traceback" not in r.stderr
    assert not (tmp_path / "ck").exists()


def test_smoke_steps_lt_2():
    pytest.importorskip("torch")
    from ffd.smoke import smoke

    with pytest.raises(ValueError):
        smoke(None, 1)


def test_smoke_dump_samples(data_root, tmp_path):
    """16 ảnh mỗi loại, trải đều theo manifest_aug, chép nguyên file (dữ liệu tổng hợp MINH HỌA)."""
    pytest.importorskip("torch")
    from ffd.smoke import N_SAMPLES, dump_samples

    for kind in ("rep", "seam"):
        assert dump_samples(data_root, kind, tmp_path) == N_SAMPLES
    assert len(list(tmp_path.glob("rep_*.jpg"))) == len(list(tmp_path.glob("seam_*.jpg"))) == 16


def test_eval_refuses_overwrite_without_force(tmp_path):
    run = tmp_path / "raw"
    run.mkdir()
    (run / "result.json").write_text("{}")
    r = run_cli(["eval", "--run", str(run)])
    assert r.returncode != 0 and "--force" in r.stderr and "Traceback" not in r.stderr
    assert (run / "result.json").read_text() == "{}"


def test_eval_pilot_run_only_val(tmp_path):
    """Run pilot cũ trên đĩa (run_name pilot/...) không được chấm ffpp_test/celebdf, kể cả với --force."""
    import json

    run = tmp_path / "pilot" / "lr1e-4_p0.5"
    run.mkdir(parents=True)
    (run / "run_info.json").write_text(json.dumps({"config": {"run_name": "pilot/lr1e-4_p0.5"}}))
    for ds in (["--datasets", "val,ffpp_test"], [], ["--datasets", "celebdf", "--force"]):
        r = run_cli(["eval", "--run", str(run), *ds])
        assert r.returncode != 0 and "pilot" in r.stderr and "Traceback" not in r.stderr
    r = run_cli(["eval", "--run", str(run), "--datasets", "bogus"])
    assert r.returncode != 0 and "bogus" in r.stderr
