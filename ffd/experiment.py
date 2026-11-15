"""Pilot (chọn lr, p_jpeg trên FF++ val) và chạy chính (raw, raw_aug -> compare.json).

Pilot: chỉ arm `raw`, chỉ chấm val. Bước 1: lr ∈ {1e-4, 2e-4, 5e-4} với p_jpeg 0.5, chọn lr có val.auc cao nhất
(hòa tuyệt đối -> lr nhỏ hơn). Bước 2: p_jpeg ∈ {0, 1} với lr đã chọn; p_jpeg = giá trị nhỏ nhất trong {0, 0.5, 1}
có val.auc >= max - 0.005. Kết quả: `<ckpt_root>/pilot/selected.json` = {"lr", "p_jpeg", "val_auc": {run: auc}, "base_config"}.
`base_config` = các trường TrainConfig được băm (như config_hash) trừ lr, p_jpeg, arm, run_name (seed vẫn giữ).
Chạy chính: đọc selected.json; base_config hiện tại khác lúc pilot -> ValueError; thiếu
`<data_root>/celebdf/manifest.parquet` -> FileNotFoundError (không chạy main khi thiếu Celeb-DF). Rồi train + eval
(val, ffpp_test, celebdf) arm raw và raw_aug (`run_name = arm`), rồi compare_runs. Seed lấy từ config (mặc định 0;
1 run/arm, exploratory).

Quy tắc chạy tiếp một run (`ensure_run`): có result.json đủ mọi dataset yêu cầu -> xong, bỏ qua; result.json thiếu
dataset yêu cầu, hoặc celebdf là {"skipped": ...} mà nay đã có dữ liệu -> chỉ eval lại (không train); có done.flag -> chỉ eval; có last.pt ->
train(resume=True); không có gì -> train mới (còn file dở từ lần chết trước epoch 1 thì force, vì không có tiến độ
để giữ). Không bao giờ truyền resume và force cùng lúc. run_info.json có config_hash khác config hiện tại -> ValueError.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .compare import compare_runs
from .config import _HASH_SKIP, ARMS, TrainConfig, config_hash
from .data.ffpp import celebdf_dir
from .evaluate import DATASETS, evaluate_run
from .io_utils import write_json_atomic
from .rundir import run_dir
from .train import train

PILOT_LRS = (1e-4, 2e-4, 5e-4)
PILOT_P0 = 0.5
PILOT_PS = (0.0, 1.0)
TIE = 0.005
MAIN_DATASETS = ("val", "ffpp_test", "celebdf")
# Trường được phép khác giữa pilot và main (pilot dò lr/p_jpeg; arm/run_name do pilot/main tự đặt).
_BASE_SKIP = ("lr", "p_jpeg", "arm", "run_name")


def fmt_lr(lr: float) -> str:
    """1e-4 -> "1e-4", 1.5e-4 -> "1.5e-4" (bỏ số 0 đệm ở số mũ)."""
    m, e = f"{lr:.6e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def pilot_name(lr: float, p: float) -> str:
    return f"pilot/lr{fmt_lr(lr)}_p{p:g}"


def base_config(cfg: TrainConfig) -> dict:
    """Các trường được băm của cfg (như config_hash) trừ lr, p_jpeg, arm, run_name; dạng JSON (tuple -> list)."""
    d = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _HASH_SKIP and k not in _BASE_SKIP}
    return json.loads(json.dumps(d, sort_keys=True, default=str))


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
        old = json.loads(info.read_text(encoding="utf-8")).get("config_hash")
        if old != config_hash(cfg):
            raise ValueError(f"{d}: config_hash của run cũ ({old}) khác config hiện tại ({config_hash(cfg)}); "
                             "xóa thư mục run hoặc đổi ckpt_root")
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
    write_json_atomic({"lr": lr, "p_jpeg": float(p_jpeg), "val_auc": aucs, "base_config": base_config(base_cfg)}, out)
    return out


def run_main(base_cfg: TrainConfig, *, vision_config=None, allow_cpu: bool = False, device=None) -> Path:
    root = Path(base_cfg.ckpt_root)
    sel_path = root / "pilot" / "selected.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"Thiếu {sel_path}: chạy `python -m ffd.cli pilot --config ...` trước `main`")
    celeb = celebdf_dir(base_cfg.data_root) / "manifest.parquet"
    if not celeb.exists():
        raise FileNotFoundError(f"Thiếu {celeb}: main bắt buộc có Celeb-DF; chạy tiền xử lý Celeb-DF trước `main`")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    if "base_config" not in sel:
        raise ValueError(f"{sel_path} không có base_config (pilot cũ); chạy lại pilot")
    now = base_config(base_cfg)
    diff = sorted(k for k in set(now) | set(sel["base_config"]) if now.get(k) != sel["base_config"].get(k))
    if diff:
        raise ValueError(f"Config khác lúc pilot ở các trường {diff}; giữ nguyên config giữa pilot và main")
    for arm in ARMS:
        cfg = dataclasses.replace(base_cfg, arm=arm, run_name=arm, lr=float(sel["lr"]), p_jpeg=float(sel["p_jpeg"]))
        ensure_run(cfg, MAIN_DATASETS, vision_config=vision_config, allow_cpu=allow_cpu, device=device)
    return compare_runs(root)
