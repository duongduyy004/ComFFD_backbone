"""Test config v2. Trạng thái: chưa chạy thử trên GPU (test CPU)."""
from pathlib import Path

import pytest

from ffd.config import ARMS, config_hash, file_hash, load_train_config, run_name_of


def wr(tmp_path, text="data_root: D:/x\n"):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults(tmp_path):
    c = load_train_config(wr(tmp_path))
    assert (c.model_name, c.max_epochs, c.patience, c.p_jpeg, c.jpeg_q, c.arm) == \
        ("openai/clip-vit-base-patch16", 15, 5, 0.5, 95, "raw")
    assert run_name_of(c) == "raw" and ARMS == ("raw", "raw_aug", "raw_rep")
    assert isinstance(c.data_root, Path) and isinstance(c.ckpt_root, Path)
    assert c.ckpt_root == Path("checkpoints/clip") and c.lora_targets == ("q_proj", "v_proj")
    assert c.tap_layers == () and c.batch_size == 32 and c.seed == 0
    assert c.lr == 1e-4 and c.grad_checkpointing is False
    assert c.model_class == "CLIPVisionModel" and len(c.image_mean) == 3 and len(c.image_std) == 3


def test_backbone_keys_validated(tmp_path):
    c = load_train_config(wr(tmp_path, "data_root: x\nimage_mean: [0.5, 0.5, 0.5]\nimage_std: [0.5, 0.5, 0.5]\n"))
    assert c.image_mean == (0.5, 0.5, 0.5) and isinstance(c.image_std, tuple)
    for k, v in (("image_mean", [0.5, 0.5]), ("image_std", [0.5, 0.0, 0.5])):
        with pytest.raises(ValueError, match=k):
            load_train_config(wr(tmp_path), {k: v})
    a, b = load_train_config(wr(tmp_path)), load_train_config(wr(tmp_path), {"model_class": "Dinov2Model"})
    assert config_hash(a) != config_hash(b)


def test_types_and_run_name(tmp_path):
    c = load_train_config(wr(tmp_path, "data_root: x\nlora_targets: [q_proj]\ntap_layers: [8]\nrun_name: foo\n"))
    assert c.lora_targets == ("q_proj",) and c.tap_layers == (8,) and run_name_of(c) == "foo"
    assert run_name_of(load_train_config(wr(tmp_path), {"arm": "raw_aug"})) == "raw_aug"


def test_rejects_old_keys(tmp_path):
    for k in ("prereg: null", "rho: 0.5", "total_steps: 10", "mode: raw", "selfblend: {}",
              "grad_accum: 1", "eval_every: 1", "ckpt_every: 1", "runs_root: r", "ffpp_root: r"):
        with pytest.raises(ValueError):
            load_train_config(wr(tmp_path, f"data_root: x\n{k}\n"))


def test_missing_data_root(tmp_path):
    with pytest.raises(ValueError):
        load_train_config(wr(tmp_path, "seed: 1\n"))


def test_validation(tmp_path):
    for k, v in (("arm", "bogus"), ("p_jpeg", 1.5), ("p_jpeg", -0.1), ("patience", 0), ("max_epochs", 0)):
        with pytest.raises(ValueError):
            load_train_config(wr(tmp_path), {k: v})


def test_arms(tmp_path):
    from ffd.config import AUG_ARMS
    assert AUG_ARMS == ("raw_aug", "raw_rep")
    for arm in ARMS:
        assert load_train_config(wr(tmp_path), {"arm": arm}).arm == arm
    for old in ("raw_aug_filt", "raw_aug_filt_loc", "raw_rep_filt"):  # arm cũ đã gỡ (còn ở tag detector_v1)
        with pytest.raises(ValueError):
            load_train_config(wr(tmp_path), {"arm": old})


def test_hash_ignores_paths_and_workers(tmp_path):
    a = load_train_config(wr(tmp_path))
    b = load_train_config(wr(tmp_path), {"data_root": "E:/y", "ckpt_root": "z", "num_workers": 0, "run_name": "n"})
    c = load_train_config(wr(tmp_path), {"lr": 2e-4})  # khác mặc định 1e-4
    assert config_hash(a) == config_hash(b) != config_hash(c) and len(config_hash(a)) == 12


def test_hash_ignores_grad_checkpointing(tmp_path):
    """grad_checkpointing chỉ đổi bộ nhớ/tốc độ, không đổi phép toán -> cùng config_hash."""
    a = load_train_config(wr(tmp_path), {"grad_checkpointing": True})
    b = load_train_config(wr(tmp_path), {"grad_checkpointing": False})
    assert config_hash(a) == config_hash(b)


def test_file_hash(tmp_path):
    p = tmp_path / "f"; p.write_bytes(b"abc")
    assert len(file_hash(p)) == 12 and file_hash(p) == file_hash(p)


def test_backbone_folders_valid():
    """Mỗi backbones/<tên>/ có config.yaml hợp lệ (ckpt_root = thư mục checkpoints của chính nó), README.md, checkpoints/."""
    root = Path(__file__).resolve().parents[1]
    dirs = sorted(d for d in (root / "backbones").iterdir() if d.is_dir())
    assert {d.name for d in dirs} >= {"clip_vit_b16", "clip_vit_l14"}
    hashes = set()
    for d in dirs:
        assert (d / "README.md").is_file() and (d / "checkpoints").is_dir(), d
        c = load_train_config(d / "config.yaml")
        assert c.ckpt_root == Path("backbones") / d.name / "checkpoints", d
        assert c.arm == "raw" and (c.lr, c.p_jpeg, c.max_epochs) == (1e-4, 0.5, 15)
        hashes.add(config_hash(c))
    assert len(hashes) == len(dirs)  # backbone khác -> config_hash khác
    b16 = load_train_config(root / "backbones/clip_vit_b16/config.yaml")
    assert b16.model_name == "openai/clip-vit-base-patch16" and b16.model_class == "CLIPVisionModel"
