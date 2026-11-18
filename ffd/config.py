"""TrainConfig v2, nạp YAML, hash config. Không còn prereg (exploratory, 1 run/arm).

Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ARMS = ("raw", "raw_aug", "raw_aug_filt", "raw_aug_filt_loc")
# Arm hợp lệ nhưng không chạy trong `main` (chạy riêng bằng train + eval). raw_rep_filt: spec v5.
EXTRA_ARMS = ("raw_rep_filt",)
ALL_ARMS = ARMS + EXTRA_ARMS
FILTER_ARMS = ("raw_aug_filt", "raw_aug_filt_loc", "raw_rep_filt")
DROP_SEAM_ARMS = ("raw_rep_filt",)  # filt rồi bỏ mọi ảnh seam
LOC_ARMS = ("raw_aug_filt_loc",)
_PATH_FIELDS = ("data_root", "ckpt_root")
_TUPLE_FIELDS = ("lora_targets", "tap_layers")
# Không băm đường dẫn/worker/tên run (không ảnh hưởng kết quả). grad_checkpointing: không đổi phép toán,
# chỉ đổi bộ nhớ/tốc độ (16/11).
_HASH_SKIP = ("data_root", "ckpt_root", "num_workers", "grad_checkpointing", "run_name")


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path  # out_root của preprocessing (chứa manifest và ảnh)
    ckpt_root: Path = Path("checkpoints/clip")
    arm: Literal["raw", "raw_aug", "raw_aug_filt", "raw_aug_filt_loc", "raw_rep_filt"] = "raw"
    run_name: str | None = None  # None -> dùng arm
    seed: int = 0
    model_name: str = "openai/clip-vit-large-patch14"
    model_revision: str | None = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple = ("q_proj", "v_proj")
    tap_layers: tuple = ()
    head: str = "binary"
    grad_checkpointing: bool = False  # tắt nhanh hơn ~21%, VRAM ~8.8 GB (smoke 16/11); ngoài hash
    batch_size: int = 32
    lr: float = 1e-4  # theo run thử lr1e-4_p0.5 (16/11); khớp config.example.yaml
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    max_epochs: int = 30
    patience: int = 5
    p_jpeg: float = 0.5
    jpeg_q: int = 95
    num_workers: int = 4


def load_train_config(path, overrides: dict | None = None) -> TrainConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    d.update(overrides or {})
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"Khóa config không hợp lệ: {sorted(unknown)}")
    for k in _PATH_FIELDS:
        if d.get(k) is not None:
            d[k] = Path(d[k])
    for k in _TUPLE_FIELDS:
        if k in d:
            d[k] = tuple(d[k])
    try:
        cfg = TrainConfig(**d)
    except TypeError as e:
        raise ValueError(f"Thiếu khóa bắt buộc: {e}") from e
    if cfg.arm not in ALL_ARMS:
        raise ValueError(f"arm phải thuộc {ALL_ARMS}, nhận {cfg.arm!r}")
    if not 0.0 <= cfg.p_jpeg <= 1.0:
        raise ValueError(f"p_jpeg phải nằm trong [0, 1], nhận {cfg.p_jpeg}")
    if cfg.patience < 1:
        raise ValueError(f"patience phải >= 1, nhận {cfg.patience}")
    if cfg.max_epochs < 1:
        raise ValueError(f"max_epochs phải >= 1, nhận {cfg.max_epochs}")
    return cfg


def run_name_of(cfg: TrainConfig) -> str:
    return cfg.run_name or cfg.arm


def config_hash(cfg: TrainConfig) -> str:
    d = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _HASH_SKIP}
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
