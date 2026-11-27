"""TrainConfig, nạp YAML, hash config. Không còn prereg (exploratory, 1 run/arm).

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

# R0 raw (đối chứng), R1 raw_aug (+ mọi rep/seam), R2 raw_rep (raw_aug bỏ seam). Không lọc, không cân bằng lại:
# preprocessing đã kiểm rep thấy được, seam sạch và cân seam/rep theo target (CompFFD/ke_hoach_run_v2_2026-11-26.md).
ARMS = ("raw", "raw_aug", "raw_rep")
AUG_ARMS = ("raw_aug", "raw_rep")  # đọc manifest_aug
_PATH_FIELDS = ("data_root", "ckpt_root")
_TUPLE_FIELDS = ("lora_targets", "tap_layers", "image_mean", "image_std")
# Không băm đường dẫn/worker/tên run (không ảnh hưởng kết quả). grad_checkpointing: không đổi phép toán,
# chỉ đổi bộ nhớ/tốc độ (16/11).
_HASH_SKIP = ("data_root", "ckpt_root", "num_workers", "grad_checkpointing", "run_name")


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path  # out_root của preprocessing (chứa manifest và ảnh)
    ckpt_root: Path = Path("checkpoints/clip")
    arm: Literal["raw", "raw_aug", "raw_rep"] = "raw"
    run_name: str | None = None  # None -> dùng arm
    seed: int = 0
    # Backbone: đổi bằng config (backbones/<tên>/config.yaml), không sửa code. Xem ffd/backbone.py.
    model_class: str = "CLIPVisionModel"  # class trong transformers
    model_name: str = "openai/clip-vit-base-patch16"
    model_revision: str | None = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple = ("q_proj", "v_proj")
    image_mean: tuple = (0.48145466, 0.4578275, 0.40821073)  # chuẩn hóa ảnh của backbone (RGB [0, 1]); mặc định CLIP
    image_std: tuple = (0.26862954, 0.26130258, 0.27577711)
    tap_layers: tuple = ()
    head: str = "binary"
    grad_checkpointing: bool = False  # không đổi phép toán, chỉ đổi bộ nhớ/tốc độ; ngoài hash
    batch_size: int = 32
    lr: float = 1e-4  # theo run thử lr1e-4_p0.5 (16/11); khớp backbones/*/config.yaml
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    max_epochs: int = 15  # ViT-L/v1: best epoch 4–8, dừng ở 8–13 -> cosine 30 epoch không kịp giảm lr (27/11)
    patience: int = 5
    p_jpeg: float = 0.5
    jpeg_q: int = 95
    num_workers: int = 8  # ViT-B nhanh hơn ViT-L ~3 lần mỗi bước -> đọc ảnh dễ thành nút cổ chai


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
    if cfg.arm not in ARMS:
        raise ValueError(f"arm phải thuộc {ARMS}, nhận {cfg.arm!r}")
    if not 0.0 <= cfg.p_jpeg <= 1.0:
        raise ValueError(f"p_jpeg phải nằm trong [0, 1], nhận {cfg.p_jpeg}")
    if cfg.patience < 1:
        raise ValueError(f"patience phải >= 1, nhận {cfg.patience}")
    for k in ("image_mean", "image_std"):
        v = getattr(cfg, k)
        if len(v) != 3 or (k == "image_std" and min(v) <= 0):
            raise ValueError(f"{k} phải có 3 giá trị (RGB){', đều > 0' if k == 'image_std' else ''}, nhận {v}")
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
