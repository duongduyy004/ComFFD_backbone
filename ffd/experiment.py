"""Pilot (chọn lr, p_jpeg trên FF++ val) và chạy chính (raw, raw_aug -> compare.json).

Pilot: chỉ arm `raw`, chỉ chấm val. Bước 1: lr ∈ {1e-4, 2e-4, 5e-4} với p_jpeg 0.5, chọn lr có val.auc cao nhất
(hòa tuyệt đối -> lr nhỏ hơn). Bước 2: p_jpeg ∈ {0, 1} với lr đã chọn; p_jpeg = giá trị nhỏ nhất trong {0, 0.5, 1}
có val.auc >= max - 0.005. Kết quả: `<ckpt_root>/pilot/selected.json` = {"lr", "p_jpeg", "val_auc": {run: auc}}.
Chạy chính: đọc selected.json, train + eval (val, ffpp_test, celebdf) arm raw và raw_aug (`run_name = arm`), rồi
compare_runs. Seed lấy từ config (mặc định 0; 1 run/arm, exploratory).

Quy tắc chạy tiếp một run (`ensure_run`): có result.json -> xong, bỏ qua; có done.flag -> chỉ eval; có last.pt ->
train(resume=True); không có gì -> train mới (còn file dở từ lần chết trước epoch 1 thì force, vì không có tiến độ
để giữ). Không bao giờ truyền resume và force cùng lúc. run_info.json có config_hash khác config hiện tại -> ValueError.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .compare import compare_runs
from .config import ARMS, TrainConfig, config_hash
from .evaluate import evaluate_run
from .io_utils import write_json_atomic
from .rundir import run_dir
from .train import train

PILOT_LRS = (1e-4, 2e-4, 5e-4)
PILOT_P0 = 0.5
PILOT_PS = (0.0, 1.0)
TIE = 0.005
MAIN_DATASETS = ("val", "ffpp_test", "celebdf")


def fmt_lr(lr: float) -> str:
    """1e-4 -> "1e-4", 1.5e-4 -> "1.5e-4" (bỏ số 0 đệm ở số mũ)."""
    m, e = f"{lr:.6e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def pilot_name(lr: float, p: float) -> str:
    return f"pilot/lr{fmt_lr(lr)}_p{p:g}"


def ensure_run(cfg: TrainConfig, datasets, *, vision_config=None, allow_cpu: bool = False, device=None) -> dict:
    """Đưa một run tới trạng thái có result.json (theo quy tắc ở docstring module); trả nội dung result.json."""
    d = run_dir(cfg)
    info = d / "run_info.json"
    if info.exists():
        old = json.loads(info.read_text(encoding="utf-8")).get("config_hash")
        if old != config_hash(cfg):
            raise ValueError(f"{d}: config_hash của run cũ ({old}) khác config hiện tại ({config_hash(cfg)}); "
                             "xóa thư mục run hoặc đổi ckpt_root")
    res = d / "result.json"
    if not res.exists():
        if not (d / "done.flag").exists():
            resume = (d / "last.pt").exists()
            force = not resume and d.exists() and any(d.iterdir())
            train(cfg, resume=resume, force=force, vision_config=vision_config, allow_cpu=allow_cpu, device=device)
        res = evaluate_run(d, datasets=tuple(datasets), vision_config=vision_config, device=device)
    return json.loads(Path(res).read_text(encoding="utf-8"))


def run_pilot(base_cfg: TrainConfig, *, vision_config=None, allow_cpu: bool = False, device=None) -> Path:
    kw = dict(vision_config=vision_config, allow_cpu=allow_cpu, device=device)
    aucs: dict[str, float] = {}

    def one(lr: float, p: float) -> float:
        name = pilot_name(lr, p)
        cfg = dataclasses.replace(base_cfg, arm="raw", run_name=name, lr=lr, p_jpeg=p)
        aucs[name] = float(ensure_run(cfg, ("val",), **kw)["val"]["auc"])
        return aucs[name]

    by_lr = {lr: one(lr, PILOT_P0) for lr in PILOT_LRS}
    lr = max(PILOT_LRS, key=lambda x: by_lr[x])  # hòa tuyệt đối -> lr đứng trước (nhỏ hơn)
    by_p = {PILOT_P0: by_lr[lr], **{p: one(lr, p) for p in PILOT_PS}}
    top = max(by_p.values())
    p_jpeg = min(p for p, a in by_p.items() if a >= top - TIE - 1e-12)  # hòa trong 0.005 -> p nhỏ hơn
    out = Path(base_cfg.ckpt_root) / "pilot" / "selected.json"
    write_json_atomic({"lr": lr, "p_jpeg": float(p_jpeg), "val_auc": aucs}, out)
    return out


def run_main(base_cfg: TrainConfig, *, vision_config=None, allow_cpu: bool = False, device=None) -> Path:
    root = Path(base_cfg.ckpt_root)
    sel_path = root / "pilot" / "selected.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"Thiếu {sel_path}: chạy `python -m ffd.cli pilot --config ...` trước `main`")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    for arm in ARMS:
        cfg = dataclasses.replace(base_cfg, arm=arm, run_name=arm, lr=float(sel["lr"]), p_jpeg=float(sel["p_jpeg"]))
        ensure_run(cfg, MAIN_DATASETS, vision_config=vision_config, allow_cpu=allow_cpu, device=device)
    return compare_runs(root)
