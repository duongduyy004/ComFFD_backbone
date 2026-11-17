"""Chạy chính: train + eval 4 arm (`ARMS`) -> compare.json + compare_<a>__<b>.json (PAIRS), summary.md.

Không còn pilot (bỏ 16/11, quyết định người dùng): `lr`, `p_jpeg` và mọi trường khác lấy thẳng từ config
(config.example.yaml: lr 1e-4, p_jpeg 0.5 theo run thử 16/11). Thiếu `<data_root>/celebdf/manifest.parquet`
-> FileNotFoundError (không chạy main khi thiếu Celeb-DF). Trước khi train, in ra stdout một khối config hiệu lực
(`EFFECTIVE_KEYS`) để phát hiện config.yaml chép từ mẫu cũ (vd lr 2e-4, grad_checkpointing true). Train + eval (val, ffpp_test, celebdf) từng arm
(`run_name = arm`), rồi compare_runs + compare_pair cho mọi cặp đủ hai run, rồi in bảng tổng hợp và ghi `<ckpt_root>/summary.md`. Seed lấy từ config
(mặc định 0; 1 run/arm, exploratory).

Quy tắc chạy tiếp một run (`ensure_run`): có result.json đủ mọi dataset yêu cầu -> xong, bỏ qua; result.json thiếu
dataset yêu cầu, hoặc celebdf là {"skipped": ...} mà nay đã có dữ liệu -> chỉ eval lại (không train);
có done.flag -> chỉ eval; có last.pt -> train(resume=True); không có gì -> train mới (còn file dở từ lần chết trước epoch 1 thì force, vì không có tiến độ
để giữ). Không bao giờ truyền resume và force cùng lúc. run_info.json có config_hash khác config hiện tại -> ValueError;
arm filt mà run_info thiếu/khác `arm_constants` hiện tại -> ValueError (cả đường bỏ qua, eval và resume).
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from . import tables
from .compare import PAIRS, compare_pair, compare_runs
from .config import ARMS, TrainConfig, config_hash
from .data.augfilter import check_arm_constants
from .data.ffpp import celebdf_dir
from .evaluate import DATASETS, evaluate_run
from .rundir import run_dir
from .train import train

MAIN_DATASETS = ("val", "ffpp_test", "celebdf")
# Khóa in ra đầu `main` (config hiệu lực sau --set).
EFFECTIVE_KEYS = ("data_root", "ckpt_root", "lr", "p_jpeg", "batch_size", "max_epochs", "patience", "seed",
                  "grad_checkpointing", "model_name", "model_revision", "num_workers")


def effective_settings(cfg: TrainConfig) -> str:
    """Khối chữ gọn: một dòng `khóa: giá trị` cho mỗi khóa trong EFFECTIVE_KEYS."""
    w = max(map(len, EFFECTIVE_KEYS))
    return "\n".join(f"  {k.ljust(w)} : {getattr(cfg, k)}" for k in EFFECTIVE_KEYS)


def _missing(res: dict, datasets, data_root) -> list[str]:
    """Dataset yêu cầu mà result.json chưa có, hoặc celebdf bị skipped trong khi nay đã có dữ liệu."""
    has_celeb = (celebdf_dir(data_root) / "manifest.parquet").exists()
    out = []
    for ds in datasets:
        v = res.get(ds)
        if v is None or (isinstance(v, dict) and "skipped" in v and ds == "celebdf" and has_celeb):
            out.append(ds)
    return out


def ensure_run(cfg: TrainConfig, datasets, *, vision_config=None, allow_cpu: bool = False, device=None) -> dict:
    """Đưa một run tới trạng thái có result.json (theo quy tắc ở docstring module); trả nội dung result.json."""
    d = run_dir(cfg)
    info = d / "run_info.json"
    if info.exists():
        rec = json.loads(info.read_text(encoding="utf-8"))
        old = rec.get("config_hash")
        if old != config_hash(cfg):
            raise ValueError(f"{d}: config_hash của run cũ ({old}) khác config hiện tại ({config_hash(cfg)}); "
                             "xóa thư mục run hoặc đổi ckpt_root")
        check_arm_constants(cfg.arm, rec, "xóa thư mục run hoặc đổi ckpt_root")
    res = d / "result.json"
    if res.exists():
        old = json.loads(res.read_text(encoding="utf-8"))
        if not _missing(old, datasets, cfg.data_root):
            return old
        # Chỉ eval lại (không train): chấm lại mọi dataset yêu cầu + dataset đã có trong result.json cũ.
        want = set(datasets) | {k for k in DATASETS if k in old}
        res = evaluate_run(d, datasets=tuple(k for k in DATASETS if k in want), vision_config=vision_config,
                           device=device)
        return json.loads(Path(res).read_text(encoding="utf-8"))
    if not (d / "done.flag").exists():
        resume = (d / "last.pt").exists()
        force = not resume and d.exists() and any(d.iterdir())
        train(cfg, resume=resume, force=force, vision_config=vision_config, allow_cpu=allow_cpu, device=device)
    res = evaluate_run(d, datasets=tuple(datasets), vision_config=vision_config, device=device)
    return json.loads(Path(res).read_text(encoding="utf-8"))


def run_main(base_cfg: TrainConfig, *, vision_config=None, allow_cpu: bool = False, device=None) -> Path:
    """Train + eval 4 arm với đúng config (lr, p_jpeg...) -> compare.json + compare_<a>__<b>.json; in bảng tổng hợp,
    ghi summary.md. Trả đường dẫn compare.json."""
    root = Path(base_cfg.ckpt_root)
    celeb = celebdf_dir(base_cfg.data_root) / "manifest.parquet"
    if not celeb.exists():
        raise FileNotFoundError(f"Thiếu {celeb}: main bắt buộc có Celeb-DF; chạy tiền xử lý Celeb-DF trước `main`")
    print(f"Config hiệu lực của main ({', '.join(ARMS)}):\n{effective_settings(base_cfg)}", flush=True)
    results = {}
    for arm in ARMS:
        cfg = dataclasses.replace(base_cfg, arm=arm, run_name=arm)
        results[arm] = ensure_run(cfg, MAIN_DATASETS, vision_config=vision_config, allow_cpu=allow_cpu, device=device)
    out = compare_runs(root)
    comp = json.loads(Path(out).read_text(encoding="utf-8"))
    pairs = {(a, b): json.loads(Path(compare_pair(root, a, b)).read_text(encoding="utf-8")) for a, b in PAIRS}
    tables.write_text_atomic(tables.summary_markdown(results, comp, pairs), root / "summary.md")
    print(f"Tổng hợp main ({root}):\n{tables.summary_text(results, comp, pairs)}", flush=True)
    return out
