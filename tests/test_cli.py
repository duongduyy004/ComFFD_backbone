"""Test CLI (không cần torch/GPU). Trạng thái: chưa chạy thử."""
import subprocess
import sys

import pytest


def run_cli(args):
    return subprocess.run([sys.executable, "-m", "ffd.cli", *args], capture_output=True, text=True)


def test_cli_rejects_unknown_dataset():
    r = run_cli(["eval", "--run", "x", "--datasets", "ffpp_val"])
    assert r.returncode != 0 and "ffpp_val" in r.stderr


def test_cli_prep_cross_only_cdf2():
    assert run_cli(["prep-cross", "--dataset", "dfdc", "--root", "r", "--out", "o", "--ffpp-config", "c"]).returncode != 0


def test_cli_bad_cross_root():
    r = run_cli(["eval", "--run", "x", "--datasets", "cdf2", "--cross-root", "nopath"])
    assert r.returncode != 0 and "cross-root" in r.stderr


def test_cli_set_locked_key_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    pre = tmp_path / "p.yaml"
    pre.write_text("lr: 1.0e-4\ntotal_steps: 10\neval_every: 5\n", encoding="utf-8")
    cfg.write_text(f"ffpp_config: x.yaml\nruns_root: {tmp_path.as_posix()}/runs\nmode: raw\nseed: 0\n"
                   f"total_steps: 1\neval_every: 1\nckpt_every: 1\nprereg: {pre.as_posix()}\n", encoding="utf-8")
    r = run_cli(["train", "--config", str(cfg), "--set", "lr=0.1"])
    assert r.returncode != 0 and "lr" in r.stderr


def test_parse_set():
    from ffd.cli import parse_sets

    assert parse_sets(["lr=1e-4", "seed=2", "mode=raw+sb", "num_workers=0"]) == \
        {"lr": 1e-4, "seed": 2, "mode": "raw+sb", "num_workers": 0}
    with pytest.raises(ValueError):
        parse_sets(["noequals"])


def test_make_detector_without_facexlib_clear_error(monkeypatch):
    from ffd import cli

    monkeypatch.setitem(sys.modules, "facexlib", None)  # import -> ImportError
    with pytest.raises(SystemExit) as e:
        cli._make_detector()
    assert "facexlib" in str(e.value)
