"""TrainConfig, nạp YAML, hash config, khóa tham số theo prereg.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from compsynth_ff.composite import DEFAULT_T

MODES = ("raw", "raw+comp", "raw+sb", "compffd")
# Khóa lấy từ prereg (cùng mọi lora_*) và không được override từ CLI.
PREREG_KEYS = ("rho", "lr", "weight_decay", "warmup_frac", "total_steps", "eval_every",
               "batch_size", "selfblend")
_PATH_FIELDS = ("ffpp_config", "runs_root", "prereg")
_HASH_SKIP = ("runs_root", "num_workers")


def _is_locked(k: str) -> bool:
    return k in PREREG_KEYS or k.startswith("lora_")


@dataclass(frozen=True)
class TrainConfig:
    ffpp_config: Path
    runs_root: Path
    mode: Literal["raw", "raw+comp", "raw+sb", "compffd"]
    seed: int
    total_steps: int
    eval_every: int
    ckpt_every: int
    split: str | None = None
    rho: float = 0.5
    model_name: str = "openai/clip-vit-large-patch14-336"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple = ("q_proj", "v_proj")
    tap_layers: tuple = ()
    head: str = "binary"
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_accum: int = 1
    num_workers: int = 4
    selfblend: dict = field(default_factory=lambda: dict(DEFAULT_T))
    prereg: Path | None = None


def _read_yaml(path) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return d or {}


def load_train_config(path, overrides: dict | None = None) -> TrainConfig:
    d = _read_yaml(path)
    overrides = dict(overrides or {})
    prereg = overrides.get("prereg", d.get("prereg"))
    if prereg:
        locked = sorted(k for k in overrides if _is_locked(k))
        if locked:
            raise ValueError(f"Khóa bị prereg khóa, không được override: {locked}")
        for k, v in _read_yaml(prereg).items():
            if _is_locked(k):
                d[k] = v
        d["prereg"] = prereg
    d.update(overrides)
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"Khóa config không hợp lệ: {sorted(unknown)}")
    for k in _PATH_FIELDS:
        if d.get(k) is not None:
            d[k] = Path(d[k])
    for k in ("lora_targets", "tap_layers"):
        if k in d:
            d[k] = tuple(d[k])
    if "selfblend" in d:
        d["selfblend"] = {**DEFAULT_T, **d["selfblend"]}
    try:
        cfg = TrainConfig(**d)
    except TypeError as e:
        raise ValueError(f"Thiếu khóa bắt buộc: {e}") from e
    if cfg.mode not in MODES:
        raise ValueError(f"mode phải thuộc {MODES}, nhận {cfg.mode!r}")
    if cfg.mode == "compffd" and not cfg.split:
        raise ValueError("mode=compffd bắt buộc có split")
    return cfg


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
