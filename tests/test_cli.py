"""Test CLI (không cần torch/GPU). Trạng thái: chưa chạy thử."""
import subprocess
import sys

import pytest
from prereg_util import write_prereg


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
    write_prereg(pre)
    cfg.write_text(f"ffpp_config: x.yaml\nruns_root: {tmp_path.as_posix()}/runs\nmode: raw\nseed: 0\n"
                   f"total_steps: 1\neval_every: 1\nckpt_every: 1\nprereg: {pre.as_posix()}\n", encoding="utf-8")
    r = run_cli(["train", "--config", str(cfg), "--set", "lr=0.1"])
    assert r.returncode != 0 and "lr" in r.stderr


def test_parse_set():
    from ffd.cli import parse_sets

    assert parse_sets(["lr=1e-4", "seed=2", "mode=raw+rep", "num_workers=0"]) == \
        {"lr": 1e-4, "seed": 2, "mode": "raw+rep", "num_workers": 0}
    with pytest.raises(ValueError):
        parse_sets(["noequals"])


def test_make_detector_without_facexlib_clear_error(monkeypatch):
    from ffd import cli

    monkeypatch.setitem(sys.modules, "facexlib", None)  # import -> ImportError
    with pytest.raises(SystemExit) as e:
        cli._make_detector()
    assert "facexlib" in str(e.value)


def _cfg(tmp_path, prereg=None):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"ffpp_config: x.yaml\nruns_root: {tmp_path.as_posix()}/runs\nmode: raw\nseed: 0\n"
                   f"total_steps: 1\neval_every: 1\nckpt_every: 1\n" + (f"prereg: {prereg}\n" if prereg else ""),
                   encoding="utf-8")
    return cfg


def test_train_without_prereg_refused(tmp_path):
    r = run_cli(["train", "--config", str(_cfg(tmp_path))])
    assert r.returncode != 0 and "prereg" in r.stderr


def test_train_set_prereg_null_refused(tmp_path):
    pre = tmp_path / "p.yaml"
    write_prereg(pre)  # đủ khóa, model_revision: null
    r = run_cli(["train", "--config", str(_cfg(tmp_path, pre.as_posix())), "--set", "prereg=null"])
    assert r.returncode != 0 and "prereg" in r.stderr


def test_pilot_runs_root_is_sibling(tmp_path):
    from ffd.cli import _load_cfg
    from ffd.rundir import run_dir

    cfg = _load_cfg(str(_cfg(tmp_path)), [], pilot=True)
    assert run_dir(cfg) == tmp_path / "runs_pilot" / "aug" / "raw" / "seed0"
    assert cfg.prereg is None


def test_malformed_set_clean_exit(tmp_path):
    r = run_cli(["train", "--config", str(_cfg(tmp_path)), "--pilot", "--set", "x=["])
    assert r.returncode != 0 and "Traceback" not in r.stderr


def test_parse_set_bool_list_and_typo(tmp_path):
    from ffd.cli import _load_cfg, parse_sets

    assert parse_sets(["grad_checkpointing=false", "tap_layers=[8,16]"]) == \
        {"grad_checkpointing": False, "tap_layers": [8, 16]}
    with pytest.raises(SystemExit) as e:
        _load_cfg(str(_cfg(tmp_path)), ["lrr=1"], pilot=True)
    assert "lrr" in str(e.value)


def test_smoke_steps_lt_2():
    pytest.importorskip("torch")
    from ffd.smoke import smoke

    with pytest.raises(ValueError):
        smoke(None, 1)


def test_train_without_model_revision_refused(tmp_path):
    """I2: run thật (không --pilot) có prereg nhưng model_revision None -> từ chối, nhắc ghim trong prereg."""
    pre = tmp_path / "p.yaml"
    write_prereg(pre)  # đủ khóa, model_revision: null
    r = run_cli(["train", "--config", str(_cfg(tmp_path, pre.as_posix()))])
    assert r.returncode != 0 and "model_revision" in r.stderr and "Traceback" not in r.stderr
