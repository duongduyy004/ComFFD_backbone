"""So sánh mức video giữa một arm gốc và nhiều arm (spec v6, mục 4): AUC video và FPR@TPR95 video, hiệu arm − gốc,
CI95 bootstrap phân tầng theo lớp. Chưa chạy thử trên dữ liệu thật của arm probe.

Điểm video = logit trung bình mỗi video. Bootstrap: B vòng, mỗi dataset một `np.random.default_rng(seed)`; mỗi vòng
bốc `rng.choice(real)` rồi `rng.choice(fake)` (cùng thứ tự với `scripts/video_bootstrap.py`, nên CI cặp trùng script
cũ), mọi arm tính trên cùng mẫu. FPR@TPR95 = `fpr[searchsorted(tpr, 0.95)]` của `roc_curve` (như script cũ).
`corr_real` = tương quan Pearson điểm video real giữa gốc và arm (thấp -> hai mô hình sai trên video khác nhau).
Ghi `<ckpt_root>/<name>.json` và `.md` (mặc định `videoboot_<base>`); không đụng `summary.md`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .io_utils import write_json_atomic
from .tables import write_text_atomic

DATASETS = ("ffpp_test", "celebdf")
NOTE = ("Exploratory: arm gốc 1 seed; CI bootstrap theo video chỉ phản ánh lấy mẫu video test, không phản ánh dao "
        "động giữa seed; chưa hiệu chỉnh đa so sánh.")


def video_scores(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("video_id", sort=True).agg(label=("label", "first"), s=("logit", "mean"))


def fpr_at_tpr95(y, s) -> float:
    f, t, _ = roc_curve(y, s)
    return float(f[np.searchsorted(t, 0.95)])


def _load(root: Path, arms: list[str], ds: str) -> dict[str, pd.DataFrame]:
    out = {}
    for a in arms:
        p = root / a / f"scores_{ds}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"thiếu {p}")
        out[a] = video_scores(pd.read_parquet(p))
    ref = out[arms[0]]
    for a in arms[1:]:
        if not out[a].index.equals(ref.index) or not out[a].label.equals(ref.label):
            raise ValueError(f"{ds}: tập video (hoặc nhãn) của {a} khác {arms[0]}")
    return out


def _ci(x) -> list[float]:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def compare_video(ckpt_root, base: str, arms: list[str], B: int = 4000, seed: int = 0,
                  name: str | None = None) -> dict:
    root = Path(ckpt_root)
    arms = list(arms)
    names = [base] + arms
    out: dict = {"base": base, "arms": arms, "B": int(B), "seed": int(seed), "note": NOTE}
    for ds in DATASETS:  # đọc và kiểm hết trước khi bootstrap
        out[ds] = _load(root, names, ds)
    for ds in DATASETS:
        V = out[ds]
        y = V[base].label.to_numpy(int)
        S = {a: V[a].s.to_numpy(float) for a in names}
        r, k = np.where(y == 0)[0], np.where(y == 1)[0]
        rng = np.random.default_rng(seed)
        auc_b = {a: np.empty(B) for a in names}
        fpr_b = {a: np.empty(B) for a in names}
        for i in range(B):
            ii = np.r_[rng.choice(r, len(r)), rng.choice(k, len(k))]
            for a in names:
                auc_b[a][i] = roc_auc_score(y[ii], S[a][ii])
                fpr_b[a][i] = fpr_at_tpr95(y[ii], S[a][ii])
        auc = {a: float(roc_auc_score(y, S[a])) for a in names}
        fpr = {a: fpr_at_tpr95(y, S[a]) for a in names}
        out[ds] = {
            "n_real": int(len(r)), "n_fake": int(len(k)), "auc": auc, "fpr_at_tpr95": fpr,
            "pairs": {f"{base}__{a}": {"auc_diff": auc[a] - auc[base], "auc_ci95": _ci(auc_b[a] - auc_b[base]),
                                       "fpr_diff": fpr[a] - fpr[base], "fpr_ci95": _ci(fpr_b[a] - fpr_b[base])}
                      for a in arms},
            "corr_real": {a: float(np.corrcoef(S[base][r], S[a][r])[0, 1]) for a in arms},
        }
    name = name or f"videoboot_{base}"
    write_json_atomic(out, root / f"{name}.json")
    write_text_atomic(_markdown(out), root / f"{name}.md")
    return out


def _markdown(j: dict) -> str:
    base, names = j["base"], [j["base"]] + j["arms"]
    L = [f"# So sánh mức video: gốc `{base}`", "", f"Tạo bởi `ffd.videoboot`, B = {j['B']}, seed {j['seed']}. {j['note']}"]
    for ds in DATASETS:
        v = j[ds]
        L += ["", f"## {ds} ({v['n_real']} video real, {v['n_fake']} video fake)", "",
              "| arm | AUC video | FPR@TPR95 |", "|---|---|---|"]
        L += [f"| {a} | {v['auc'][a]:.4f} | {v['fpr_at_tpr95'][a]:.3f} |" for a in names]
        L += ["", "| arm − gốc | ΔAUC [CI95] | ΔFPR@TPR95 [CI95] | tương quan real |", "|---|---|---|---|"]
        for a in j["arms"]:
            q = v["pairs"][f"{base}__{a}"]
            L.append(f"| {a} − {base} | {q['auc_diff']:+.4f} [{q['auc_ci95'][0]:+.4f}, {q['auc_ci95'][1]:+.4f}] | "
                     f"{q['fpr_diff']:+.3f} [{q['fpr_ci95'][0]:+.3f}, {q['fpr_ci95'][1]:+.3f}] | "
                     f"{v['corr_real'][a]:.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ffd.videoboot", description="So sánh mức video với arm gốc (spec v6)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="config.yaml của train; lấy ckpt_root từ đây")
    g.add_argument("--ckpt-root", help="thư mục chứa các run (thay cho --config)")
    ap.add_argument("--base", default="raw")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    if a.config:
        from .config import load_train_config

        root = load_train_config(a.config).ckpt_root
    else:
        root = Path(a.ckpt_root)
    compare_video(root, a.base, a.arms, B=a.B, name=a.name)
    print((Path(root) / f"{a.name or 'videoboot_' + a.base}.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
