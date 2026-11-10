"""Test config. Trạng thái: chưa chạy thử."""
from pathlib import Path

import pytest
import yaml
from compsynth_ff.composite import DEFAULT_T

from ffd.config import config_hash, file_hash, load_train_config
from prereg_util import full_prereg

BASE = dict(ffpp_config="ffpp.yaml", runs_root="runs", mode="raw", seed=0,
            total_steps=100, eval_every=10, ckpt_every=50)


def write_yaml(path, **kw):
    path = Path(path)
    if path.suffix != ".yaml":
        path = path / "cfg.yaml"
    if path.name == "pre.yaml":  # R3: prereg luôn đủ khóa; giữ cả giá trị None (vd model_revision)
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in full_prereg(**kw).items()}
    else:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in {**BASE, **kw}.items() if v is not None}
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    return path


def test_defaults_and_types(tmp_path):
    c = load_train_config(write_yaml(tmp_path))
    assert c.selfblend == DEFAULT_T and c.lora_targets == ("q_proj", "v_proj")
    assert isinstance(c.runs_root, Path) and c.prereg is None and not hasattr(c, "split")


def test_compffd_mode_rejected(tmp_path):
    with pytest.raises(ValueError):  # spec augmentation 11/11: bỏ chế độ compffd (E0)
        load_train_config(write_yaml(tmp_path, mode="compffd"))


def test_bad_mode(tmp_path):
    with pytest.raises(ValueError):
        load_train_config(write_yaml(tmp_path, mode="bogus"))


def test_prereg_locks_keys(tmp_path):
    pre = write_yaml(tmp_path / "pre.yaml", rho=0.3, lr=2e-4, total_steps=100, eval_every=10)
    cfg_path = write_yaml(tmp_path, prereg=pre)
    with pytest.raises(ValueError):
        load_train_config(cfg_path, overrides={"lr": 1e-3})
    c = load_train_config(cfg_path, overrides={"seed": 3})
    assert c.rho == 0.3 and c.seed == 3 and c.prereg == pre


def test_non_locked_override_ok_without_prereg(tmp_path):
    c = load_train_config(write_yaml(tmp_path), overrides={"lr": 1e-3})
    assert c.lr == 1e-3


def test_config_hash_ignores_runs_root_and_workers(tmp_path):
    a = load_train_config(write_yaml(tmp_path))
    b = load_train_config(write_yaml(tmp_path), overrides={"runs_root": Path("other"), "num_workers": 0})
    c = load_train_config(write_yaml(tmp_path), overrides={"seed": 1})
    assert config_hash(a) == config_hash(b) != config_hash(c)
    assert len(config_hash(a)) == 12


def test_file_hash(tmp_path):
    p = tmp_path / "f"; p.write_bytes(b"abc")
    assert len(file_hash(p)) == 12 and file_hash(p) == file_hash(p)


def test_example_yaml_valid():
    p = Path(__file__).resolve().parents[1] / "config.example.yaml"
    assert load_train_config(p).mode == "raw"


def test_config_hash_ignores_path_spelling(tmp_path):
    a = load_train_config(write_yaml(tmp_path), overrides={"ffpp_config": Path("a/b.yaml")})
    b = load_train_config(write_yaml(tmp_path), overrides={"ffpp_config": Path("C:/other/root/b.yaml")})
    c = load_train_config(write_yaml(tmp_path), overrides={"ffpp_config": Path("/mnt/x/b.yaml")})
    import dataclasses
    d = dataclasses.replace(a, prereg=Path("p/q.yaml"))
    e = dataclasses.replace(a, prereg=Path("/vol/prereg.yaml"))
    assert config_hash(a) == config_hash(b) == config_hash(c) == config_hash(d) == config_hash(e)


def test_model_revision_hashed_and_locked(tmp_path):
    """I2: model_revision vào config_hash và bị prereg khóa."""
    a = load_train_config(write_yaml(tmp_path))
    b = load_train_config(write_yaml(tmp_path), overrides={"model_revision": "abc"})
    assert a.model_revision is None and b.model_revision == "abc" and config_hash(a) != config_hash(b)
    pre = write_yaml(tmp_path / "pre.yaml", model_revision="abc", lr=2e-4)
    c = load_train_config(write_yaml(tmp_path, prereg=pre))
    assert c.model_revision == "abc"
    with pytest.raises(ValueError, match="model_revision"):
        load_train_config(write_yaml(tmp_path, prereg=pre), overrides={"model_revision": "def"})


def test_prereg_locks_all_result_keys_and_rejects_unknown(tmp_path):
    """I4: grad_accum, model_name, model_revision, head, tap_layers, grad_checkpointing bị khóa; khóa lạ trong prereg -> lỗi."""
    from ffd.config import PREREG_KEYS

    for k in ("grad_accum", "model_name", "model_revision", "head", "tap_layers", "grad_checkpointing", "rho",
              "batch_size"):
        assert k in PREREG_KEYS
    pre = write_yaml(tmp_path / "pre.yaml", grad_accum=2, tap_layers=[8], grad_checkpointing=False, lora_r=8)
    cfg_path = write_yaml(tmp_path, prereg=pre)
    c = load_train_config(cfg_path)
    assert c.grad_accum == 2 and c.tap_layers == (8,) and c.grad_checkpointing is False and c.lora_r == 8
    for k, v in (("grad_accum", 4), ("head", "binary"), ("tap_layers", [16]), ("model_name", "x")):
        with pytest.raises(ValueError, match=k):
            load_train_config(cfg_path, overrides={k: v})
    for bad in ({"seed": 1}, {"foo": 1, "num_workers": 2}):
        pre2 = write_yaml(tmp_path / "pre.yaml", lr=2e-4, **bad)
        with pytest.raises(ValueError, match=sorted(bad)[0]):
            load_train_config(write_yaml(tmp_path, prereg=pre2))


def test_prereg_missing_key_rejected(tmp_path):
    """R3: prereg thiếu bất kỳ khóa PREREG_KEYS/lora_* nào -> ValueError nêu tên khóa thiếu."""
    from ffd.config import PREREG_REQUIRED

    for k in ("total_steps", "lora_r", "model_revision", "selfblend"):
        d = full_prereg()
        del d[k]
        pre = tmp_path / "p.yaml"
        pre.write_text(yaml.safe_dump(d), encoding="utf-8")
        with pytest.raises(ValueError, match=k):
            load_train_config(write_yaml(tmp_path, prereg=pre))
    assert set(PREREG_REQUIRED) == set(full_prereg())  # file mẫu ghi đúng và đủ khóa bắt buộc


def test_prereg_example_loads(tmp_path):
    """File mẫu configs/prereg_aug.example.yaml nạp được nguyên trạng."""
    from prereg_util import PREREG_EXAMPLE

    c = load_train_config(write_yaml(tmp_path, prereg=PREREG_EXAMPLE))
    assert c.lora_targets == ("q_proj", "v_proj") and c.model_revision is None
