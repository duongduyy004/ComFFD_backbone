"""Thư mục run: bố cục, chống ghi đè run đã xong, run_info.json.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from compsynth_ff.io_utils import write_json_atomic

from .config import TrainConfig, config_hash, file_hash

FFPP_STAGE = "render"  # stage cuối của compsynth_ff (gồm cả trường extract + select)


def run_dir(cfg: TrainConfig) -> Path:
    if cfg.mode == "compffd":
        return Path(cfg.runs_root) / "compffd" / cfg.split / f"seed{cfg.seed}"
    return Path(cfg.runs_root) / "aug" / cfg.mode / f"seed{cfg.seed}"


# Kết quả của một lần chạy (cả điểm eval: scores_*.parquet, eval_info.json): --force xóa hết, không resume/force thì từ chối.
STALE = ("done.flag", "best.pt", "last.pt", "val_curve.jsonl", "train_log.jsonl", "error.json",
         "scores_ffpp_test.parquet", "scores_compsynth_test.parquet", "scores_cdf2.parquet", "eval_info.json")


def prepare_run_dir(cfg: TrainConfig, *, resume: bool, force: bool) -> Path:
    d = run_dir(cfg)
    if (d / "best.pt").exists() and (d / "done.flag").exists() and not force:
        raise FileExistsError(f"Run đã xong: {d} (dùng force để chạy lại)")
    if not resume and not force:
        left = [n for n in STALE if (d / n).exists()]
        if left:
            raise FileExistsError(f"Thư mục run còn kết quả cũ {left} trong {d}; dùng resume hoặc force")
    info = d / "run_info.json"
    if resume and info.exists():
        old = json.loads(info.read_text(encoding="utf-8")).get("config_hash")
        if old != config_hash(cfg):
            raise ValueError(f"config_hash khác run cũ ({old} != {config_hash(cfg)}); không resume")
    d.mkdir(parents=True, exist_ok=True)
    if force and not resume:  # chạy lại từ đầu: xóa kết quả cũ để không lẫn vào run mới
        for n in STALE:
            (d / n).unlink(missing_ok=True)
    return d


def _versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "peft"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def _gpu():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


FFPP_PATH_FIELDS = ("data_root", "out_root", "splits_dir")  # khác nhau giữa máy/ổ đĩa: không băm


def ffpp_config_hash(ff_cfg) -> str:
    """R1: hash các trường ĐÃ PARSE của Config FF++, bỏ trường đường dẫn. Đổi máy/ổ đĩa/thư mục không đổi hash;
    đổi bất kỳ tham số nào (kể cả trường mới thêm sau này) thì đổi hash."""
    import hashlib

    d = {k: v for k, v in dataclasses.asdict(ff_cfg).items() if k not in FFPP_PATH_FIELDS}
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def ffpp_hashes(cfg: TrainConfig, ff_cfg) -> dict:
    """Định danh artifact tiền xử lý: `ffpp_config_hash(ff_cfg)` và stage_hash(ff_cfg, "render").
    ff_cfg None -> nạp từ cfg.ffpp_config nếu file tồn tại; không có thì cả hai là None."""
    from compsynth_ff.config import load_config, stage_hash

    if ff_cfg is None and cfg.ffpp_config is not None and Path(cfg.ffpp_config).is_file():
        ff_cfg = load_config(Path(cfg.ffpp_config))
    if ff_cfg is None:
        return {"ffpp_config_hash": None, "ffpp_stage_hash": None}
    return {"ffpp_config_hash": ffpp_config_hash(ff_cfg), "ffpp_stage_hash": stage_hash(ff_cfg, FFPP_STAGE)}


def write_run_info(dir, cfg: TrainConfig, extra: dict, ff_cfg=None) -> None:
    """Ghi run_info.json. extra có thể chứa model_commit (commit HF đã resolve), split_file (đường dẫn để băm), ..."""
    extra = dict(extra or {})
    split_file = extra.pop("split_file", None)
    info = {
        "config_hash": config_hash(cfg),
        "prereg_hash": file_hash(cfg.prereg) if cfg.prereg else None,
        "split_hash": file_hash(split_file) if split_file else None,
        "versions": _versions(),
        "model_name": cfg.model_name,
        "model_revision": cfg.model_revision,
        "model_commit": extra.pop("model_commit", None),
        **ffpp_hashes(cfg, ff_cfg),
        "gpu": _gpu(),
        "seed": cfg.seed,
        "mode": cfg.mode,
        "config": json.loads(json.dumps(dataclasses.asdict(cfg), default=str)),  # evaluate dựng lại Detector từ đây
        **extra,
    }
    write_json_atomic(info, Path(dir) / "run_info.json")
