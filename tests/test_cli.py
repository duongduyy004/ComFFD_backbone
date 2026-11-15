"""Test CLI v2 (parse lệnh, kiểm tham số trước khi chạy). Trạng thái: chưa chạy thử trên GPU."""
import subprocess
import sys

import pytest

from ffd.cli import build_parser, parse_sets


def run_cli(args):
    return subprocess.run([sys.executable, "-m", "ffd.cli", *args], capture_output=True, text=True)


def test_cli_parses_commands():
    p = build_parser()
    for argv in (["pilot", "--config", "c.yaml", "--set", "lr=1e-4"], ["main", "--config", "c.yaml"],
                 ["train", "--config", "c.yaml", "--resume"], ["train", "--config", "c.yaml", "--force"],
                 ["eval", "--run", "r"], ["eval", "--run", "r", "--data-root", "d"],
                 ["compare", "--ckpt-root", "ck"], ["smoke", "--config", "c.yaml", "--steps", "5"]):
        a = p.parse_args(argv)
        assert a.cmd == argv[0] and callable(a.fn)
    with pytest.raises(SystemExit):
        p.parse_args(["report", "--runs-root", "r"])  # lệnh cũ đã bỏ


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
    r = run_cli(["pilot", "--config", str(_cfg(tmp_path)), "--set", "lrr=1"])
    assert r.returncode != 0 and "lrr" in r.stderr and "Traceback" not in r.stderr


def test_malformed_set_clean_exit(tmp_path):
    r = run_cli(["train", "--config", str(_cfg(tmp_path)), "--set", "x=["])
    assert r.returncode != 0 and "Traceback" not in r.stderr


def test_main_without_pilot_clean_exit(tmp_path):
    r = run_cli(["main", "--config", str(_cfg(tmp_path))])
    assert r.returncode != 0 and "pilot" in r.stderr and "Traceback" not in r.stderr


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
    assert len(list(tmp_path.glob("rep_*.png"))) == len(list(tmp_path.glob("seam_*.png"))) == 16
