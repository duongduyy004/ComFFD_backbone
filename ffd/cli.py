"""CLI: python -m ffd.cli {train,eval,report,smoke}.

Import nặng (torch, model) để trễ trong từng lệnh, nên kiểm tra tham số lỗi thất bại nhanh.
`train` KHÔNG truyền allow_cpu (train thật cần CUDA); `--set k=v` đi qua load_train_config nên khóa prereg bị từ chối.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVAL_DATASETS = ("ffpp_test",)


def _scalar(v: str, yaml):
    for f in (int, float):  # YAML 1.1 đọc '1e-4' thành chuỗi, nên thử số trước
        try:
            return f(v)
        except ValueError:
            pass
    return yaml.safe_load(v)


def parse_sets(items) -> dict:
    """['lr=1e-4', 'mode=raw'] -> {'lr': 1e-4, 'mode': 'raw'}; giá trị parse bằng YAML (số, bool, chuỗi, list)."""
    import yaml

    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise ValueError(f"--set phải dạng k=v, nhận {it!r}")
        out[k.strip()] = _scalar(v, yaml)
    return out


def _load_cfg(config, sets, pilot: bool = False):
    import dataclasses

    import yaml

    from .config import load_train_config

    try:
        cfg = load_train_config(config, parse_sets(sets))
    except (ValueError, yaml.YAMLError) as e:
        raise SystemExit(f"Lỗi config: {e}") from e
    if pilot:  # thư mục anh em, tách khỏi runs_root thật để không đè run seed 0 sau này
        cfg = dataclasses.replace(cfg, runs_root=Path(str(cfg.runs_root).rstrip("/\\") + "_pilot"))
    return cfg


def _cmd_train(a):
    from .train import train

    cfg = _load_cfg(a.config, a.set, pilot=a.pilot)
    if not a.pilot and not cfg.prereg:
        raise SystemExit("Từ chối train: config không có prereg (thiếu, null hoặc rỗng). Run thật cần "
                         "--set prereg=configs/prereg_aug.yaml; pilot thì thêm --pilot.")
    if not a.pilot and cfg.model_revision is None:
        raise SystemExit("Từ chối train: model_revision chưa ghim (None). Đặt `model_revision: <commit>` trong prereg; "
                         "`ffd.cli smoke` in ra commit đã resolve của model_name.")
    d = train(cfg, resume=a.resume, force=a.force)  # không allow_cpu
    print(d)


def _cmd_eval(a):
    from .evaluate import evaluate_run

    ds = [x for x in a.datasets.split(",") if x]
    bad = [x for x in ds if x not in EVAL_DATASETS]
    if bad or not ds:
        raise SystemExit(f"--datasets không hợp lệ {bad}; chỉ chấp nhận {list(EVAL_DATASETS)}")
    try:
        print(evaluate_run(a.run, ds, ffpp_root=a.ffpp_root, prereg=a.prereg))
    except ValueError as e:
        raise SystemExit(f"Lỗi eval: {e}") from e


def _cmd_report(a):
    from .report import build_report

    print(build_report(a.runs_root))


def _cmd_smoke(a):
    from .smoke import smoke

    cfg = _load_cfg(a.config, a.set)
    import json

    try:
        p = smoke(cfg, a.steps)
    except ValueError as e:
        raise SystemExit(f"Lỗi smoke: {e}") from e
    print(p)
    print(f"model_commit (ghi vào prereg là model_revision): {json.loads(p.read_text(encoding='utf-8'))['model_commit']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffd.cli")
    sp = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-progress", action="store_true", help="tắt thanh tiến trình (stderr)")

    t = sp.add_parser("train", parents=[common])
    t.add_argument("--config", required=True)
    t.add_argument("--set", action="append", default=[], metavar="k=v")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--force", action="store_true")
    t.add_argument("--pilot", action="store_true",
                   help="ghi vào <runs_root>_pilot, cho phép thiếu prereg; kết quả pilot không vào report")
    t.set_defaults(fn=_cmd_train)

    e = sp.add_parser("eval", parents=[common])
    e.add_argument("--run", required=True)
    e.add_argument("--datasets", required=True, help=",".join(EVAL_DATASETS))
    e.add_argument("--ffpp-root", default=None,
                   help="thư mục output preprocessing trên máy eval (khác máy train); manifest phải trùng hash lúc train")
    e.add_argument("--prereg", default=None, help="đường dẫn prereg trên máy eval; nội dung phải trùng lúc train")
    e.set_defaults(fn=_cmd_eval)

    r = sp.add_parser("report", parents=[common])
    r.add_argument("--runs-root", required=True)
    r.set_defaults(fn=_cmd_report)

    s = sp.add_parser("smoke", parents=[common])
    s.add_argument("--config", required=True)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--set", action="append", default=[], metavar="k=v")
    s.set_defaults(fn=_cmd_smoke)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from . import progress

    progress.ENABLED = not a.no_progress
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
