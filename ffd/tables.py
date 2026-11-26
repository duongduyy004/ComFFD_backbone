"""Bảng kết quả dạng chữ: cột cố định (in stdout) và Markdown (result.md). Không import torch.

Số chỉ số in 4 chữ số thập phân; dataset skipped -> một dòng `skipped: <lý do>`.
Trạng thái: chưa chạy thử trên dữ liệu thật.
"""
from __future__ import annotations

from pathlib import Path

from .io_utils import atomic_write_bytes

RESULT_HEAD = ["dataset", "n_frames", "ACC", "AUC", "PREC", "RECALL", "F1", "AUC_video"]
_KEYS = ["acc", "auc", "precision", "recall", "f1"]  # theo thứ tự ACC, AUC, PREC, RECALL, F1
DATASETS = ("val", "ffpp_test", "celebdf")


def f4(v) -> str:
    return "" if v is None else f"{float(v):.4f}"


def _skipped(v) -> str | None:
    return v["skipped"] if isinstance(v, dict) and "skipped" in v else None


def fixed(head: list[str], rows: list) -> str:
    """Bảng ASCII cột cố định. Dòng là list ô, hoặc chuỗi (in nguyên một dòng, vd `celebdf  skipped: ...`)."""
    cells = [r for r in rows if not isinstance(r, str)]
    w = [max(len(str(x)) for x in col) for col in zip(head, *cells)]
    line = lambda r: "  ".join(str(x).ljust(w[0]) if i == 0 else str(x).rjust(w[i]) for i, x in enumerate(r))
    out = [line(head), "  ".join("-" * x for x in w)]
    out += [r if isinstance(r, str) else line(r) for r in rows]
    return "\n".join(out)


def md(head: list[str], rows: list) -> str:
    """Bảng Markdown; dòng chuỗi -> ô đầu là chuỗi đó, các ô còn lại trống."""
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        r = [r] + [""] * (len(head) - 1) if isinstance(r, str) else r
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- một run (result.json)
def result_rows(res: dict, *, for_md: bool = False) -> list:
    rows = []
    for ds in DATASETS:
        if ds not in res:
            continue
        v = res[ds]
        sk = _skipped(v)
        if sk is not None:
            rows.append(f"{ds}: skipped: {sk}" if for_md else f"{ds}  skipped: {sk}")
            continue
        rows.append([ds, int(v["n_frames"]), *(f4(v[k]) for k in _KEYS), f4(v.get("auc_video"))])
    return rows


def result_text(res: dict) -> str:
    return fixed(RESULT_HEAD, result_rows(res))


def result_markdown(res: dict, run_name: str) -> str:
    parts = [f"# Kết quả run `{run_name}` (arm `{res.get('arm')}`)", "",
             f"best_epoch: {res.get('best_epoch')} / epochs_run: {res.get('epochs_run')}", "",
             md(RESULT_HEAD, result_rows(res, for_md=True))]
    per = (res.get("ffpp_test") or {}).get("auc_per_method")
    if per:
        parts += ["", "## ffpp_test: AUC theo method", "",
                  md(["method", "AUC"], [[m, f4(a)] for m, a in sorted(per.items())])]
    return "\n".join(parts) + "\n"


def write_text_atomic(text: str, path: Path) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))
