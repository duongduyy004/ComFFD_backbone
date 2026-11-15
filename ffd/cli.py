"""CLI: python -m ffd.cli {pilot,main,train,eval,compare,smoke}.

Thứ tự dùng: smoke -> pilot -> main. Import nặng (torch, model) để trễ trong từng lệnh, nên lỗi tham số thất bại nhanh.
CLI không truyền allow_cpu (train thật cần CUDA). `--set k=v` đi qua load_train_config (khóa lạ bị từ chối).
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import argparse
import sys


def _scalar(v: str, yaml):
    for f in (int, float):  # YAML 1.1 đọc '1e-4' thành chuỗi, nên thử số trước
        try:
            return f(v)
        except ValueError:
            pass
    return yaml.safe_load(v)


def parse_sets(items) -> dict:
    """['lr=1e-4', 'arm=raw'] -> {'lr': 1e-4, 'arm': 'raw'}; giá trị parse bằng YAML (số, bool, chuỗi, list)."""
    import yaml

    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise ValueError(f"--set phải dạng k=v, nhận {it!r}")
        out[k.strip()] = _scalar(v, yaml)
    return out


def _load_cfg(config, sets):
    import yaml

    from .config import load_train_config

    try:
        return load_train_config(config, parse_sets(sets))
    except (ValueError, yaml.YAMLError, OSError) as e:
        raise SystemExit(f"Lỗi config: {e}") from e


def _cmd_pilot(a):
    from . import experiment

    p = experiment.run_pilot(_load_cfg(a.config, a.set))
    print(p)
    print(p.read_text(encoding="utf-8"))


def _cmd_main(a):
    from . import experiment

    cfg = _load_cfg(a.config, a.set)
    try:
        print(experiment.run_main(cfg))
    except FileNotFoundError as e:
        raise SystemExit(f"Lỗi main: {e}") from e


def _cmd_train(a):
    from .train import train

    cfg = _load_cfg(a.config, a.set)
    print(train(cfg, resume=a.resume, force=a.force))  # không allow_cpu


def _cmd_eval(a):
    from .evaluate import evaluate_run

    try:
        print(evaluate_run(a.run, data_root=a.data_root))
    except ValueError as e:
        raise SystemExit(f"Lỗi eval: {e}") from e


def _cmd_compare(a):
    from .compare import compare_runs

    try:
        print(compare_runs(a.ckpt_root))
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Lỗi compare: {e}") from e


def _cmd_smoke(a):
    import json

    from .smoke import smoke

    cfg = _load_cfg(a.config, a.set)
    try:
        p = smoke(cfg, a.steps)
    except ValueError as e:
        raise SystemExit(f"Lỗi smoke: {e}") from e
    print(p)
    print(p.read_text(encoding="utf-8"))
    print(f"model_commit (ghim vào config là model_revision): {json.loads(p.read_text(encoding='utf-8'))['model_commit']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffd.cli")
    sp = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-progress", action="store_true", help="tắt thanh tiến trình (stderr)")

    def with_config(name, fn, help_):
        s = sp.add_parser(name, parents=[common], help=help_)
        s.add_argument("--config", required=True)
        s.add_argument("--set", action="append", default=[], metavar="k=v")
        s.set_defaults(fn=fn)
        return s

    with_config("pilot", _cmd_pilot, "5 run arm raw, chỉ val -> <ckpt_root>/pilot/selected.json")
    with_config("main", _cmd_main, "train + eval raw, raw_aug với lr/p_jpeg đã chọn -> compare.json")
    t = with_config("train", _cmd_train, "train một run (<ckpt_root>/<run_name>)")
    t.add_argument("--resume", action="store_true", help="chạy tiếp từ last.pt")
    t.add_argument("--force", action="store_true", help="xóa kết quả cũ của run rồi chạy lại")
    s = with_config("smoke", _cmd_smoke, "vài bước train ViT-L thật trên GPU: giây/bước, VRAM đỉnh, ảnh mẫu")
    s.add_argument("--steps", type=int, default=50)

    e = sp.add_parser("eval", parents=[common], help="chấm best.pt của một run -> result.json")
    e.add_argument("--run", required=True)
    e.add_argument("--data-root", default=None, help="data_root trên máy eval (khác máy train); dữ liệu phải trùng hash")
    e.set_defaults(fn=_cmd_eval)

    c = sp.add_parser("compare", parents=[common], help="so sánh <ckpt_root>/raw và raw_aug -> compare.json")
    c.add_argument("--ckpt-root", required=True)
    c.set_defaults(fn=_cmd_compare)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "train" and a.resume and a.force:
        raise SystemExit("Không dùng --resume và --force cùng lúc (resume giữ tiến độ, force xóa hết).")
    from . import progress

    progress.ENABLED = not a.no_progress
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
