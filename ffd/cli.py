"""CLI: python -m ffd.cli {main,train,eval,smoke,diagnose}.

Thứ tự dùng: smoke -> main (3 arm raw, raw_aug, raw_rep; lr, p_jpeg lấy từ config).
Import nặng (torch, model) để trễ trong từng lệnh, nên lỗi tham số thất bại nhanh.
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


def _cmd_main(a):
    from . import experiment

    cfg = _load_cfg(a.config, a.set)
    try:
        print(experiment.run_main(cfg, B=a.B))
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Lỗi main: {e}") from e


def _cmd_train(a):
    from .train import train

    cfg = _load_cfg(a.config, a.set)
    print(train(cfg, resume=a.resume, force=a.force))  # không allow_cpu


EVAL_DATASETS = ("val", "ffpp_test", "celebdf")
_HELD_OUT = ("ffpp_test", "celebdf")


def _cmd_eval(a):
    import json
    from pathlib import Path

    run = Path(a.run)
    datasets = tuple(x.strip() for x in a.datasets.split(",") if x.strip())
    bad = [x for x in datasets if x not in EVAL_DATASETS]
    if bad or not datasets:
        raise SystemExit(f"Lỗi eval: --datasets không hợp lệ {bad}; chỉ {list(EVAL_DATASETS)} (không rỗng)")
    if (run / "result.json").exists() and not a.force:
        raise SystemExit(f"Lỗi eval: {run / 'result.json'} đã có; dùng --force để chấm lại và ghi đè")
    info_p = run / "run_info.json"
    if info_p.exists():  # thiếu run_info -> evaluate_run báo lỗi
        info = json.loads(info_p.read_text(encoding="utf-8"))
        name = (info.get("config") or {}).get("run_name") or info.get("run_name") or ""
        held = [x for x in datasets if x in _HELD_OUT]
        if str(name).startswith("pilot/") and held:  # run pilot cũ (trước 16/11) trên đĩa: vẫn chỉ val
            raise SystemExit(f"Lỗi eval: run pilot ({name}) chỉ được chấm val, không chấm {held} (chống leakage); "
                             "dùng --datasets val")
    from .evaluate import evaluate_run

    try:
        print(evaluate_run(run, data_root=a.data_root, datasets=datasets))
    except ValueError as e:
        raise SystemExit(f"Lỗi eval: {e}") from e


def _cmd_diagnose(a):
    from .diagnose import diagnose

    try:
        print(diagnose(a.ckpt_root, data_root=a.data_root, n_per_kind=a.n, batch_size=a.batch_size,
                       num_workers=a.num_workers))
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Lỗi diagnose: {e}") from e


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

    m = with_config("main", _cmd_main, "train + eval raw, raw_aug, raw_rep (arm đã có result.json thì bỏ qua) "
                    "-> videoboot, realfp, summary.{json,md} theo tiêu chí ghi trước")
    m.add_argument("--B", type=int, default=4000, help="số vòng bootstrap")
    t = with_config("train", _cmd_train, "train một run (<ckpt_root>/<run_name>)")
    t.add_argument("--resume", action="store_true", help="chạy tiếp từ last.pt")
    t.add_argument("--force", action="store_true", help="xóa kết quả cũ của run rồi chạy lại")
    s = with_config("smoke", _cmd_smoke, "vài bước train ViT-L thật trên GPU: giây/bước, VRAM đỉnh, ảnh mẫu")
    s.add_argument("--steps", type=int, default=50)

    e = sp.add_parser("eval", parents=[common], help="chấm best.pt của một run -> result.json")
    e.add_argument("--run", required=True)
    e.add_argument("--data-root", default=None, help="data_root trên máy eval (khác máy train); dữ liệu phải trùng hash")
    e.add_argument("--datasets", default=",".join(EVAL_DATASETS),
                   help="danh sách cách nhau bởi dấu phẩy (val,ffpp_test,celebdf); run pilot cũ chỉ được val")
    e.add_argument("--force", action="store_true", help="ghi đè result.json đã có")
    e.set_defaults(fn=_cmd_eval)

    g = sp.add_parser("diagnose", parents=[common], help="chẩn đoán raw_aug so với raw -> <ckpt_root>/diagnose/")
    g.add_argument("--ckpt-root", required=True)
    g.add_argument("--data-root", default=None, help="data_root trên máy này (mặc định: theo run_info)")
    g.add_argument("--n", type=int, default=2000, help="số ảnh rep và seam lấy mẫu mỗi loại")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--num-workers", type=int, default=8)
    g.set_defaults(fn=_cmd_diagnose)
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
