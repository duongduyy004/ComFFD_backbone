"""So sánh arm theo video (bootstrap phân tầng theo lớp) -> compare_<a>__<b>.json (frame + video) và summary.md.
Dùng: py -I detector/scripts/video_bootstrap.py <ckpt_root>   (cần pandas, numpy, scikit-learn; chạy bằng venv của detector)
AUC video = AUC của logit trung bình mỗi video. CI95 = paired bootstrap 4000 lần trên video, seed 0.
Exploratory: 1 seed/arm; CI chỉ phản ánh lấy mẫu video test, không phản ánh dao động giữa seed. Chưa hiệu chỉnh đa so sánh."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffd.compare import PAIRS, compare_pair

ROOT = Path(sys.argv[1])
ARMS = ["raw", "raw_aug", "raw_aug_filt", "raw_aug_filt_loc", "raw_strongaug"]
PAIRS_ALL = list(PAIRS) + [("raw", "raw_strongaug")]
DS = ("ffpp_test", "celebdf"); B = 4000
NOTE = ("Exploratory: 1 seed/arm, không prereg; CI bootstrap theo video (chỉ phản ánh lấy mẫu video test, "
        "không phản ánh dao động giữa seed); chưa hiệu chỉnh đa so sánh.")

def vid(ds):
    out = {}
    for a in ARMS:
        df = pd.read_parquet(ROOT / a / f"scores_{ds}.parquet")
        out[a] = df.groupby("video_id").agg(label=("label", "first"), s=("logit", "mean"))
    idx = out["raw"].index
    assert all(out[a].index.equals(idx) for a in ARMS), "tập video khác nhau giữa các arm"
    return out

def fpr_at_tpr95(y, s):
    f, t, _ = roc_curve(y, s); return float(f[np.searchsorted(t, 0.95)])

res = {a: json.loads((ROOT / a / "result.json").read_text(encoding="utf-8")) for a in ARMS}
video = {}
for ds in DS:
    V = vid(ds); y = V["raw"].label.values; S = {a: V[a].s.values for a in ARMS}
    rng = np.random.default_rng(0); r, k = np.where(y == 0)[0], np.where(y == 1)[0]
    boot = {a: np.empty(B) for a in ARMS}
    for i in range(B):
        ii = np.r_[rng.choice(r, len(r)), rng.choice(k, len(k))]
        for a in ARMS: boot[a][i] = roc_auc_score(y[ii], S[a][ii])
    video[ds] = dict(n_real=int(len(r)), n_fake=int(len(k)),
        auc={a: float(roc_auc_score(y, S[a])) for a in ARMS},
        fpr_at_tpr95={a: fpr_at_tpr95(y, S[a]) for a in ARMS},
        pairs={})
    for a, b in PAIRS_ALL:
        d = boot[b] - boot[a]
        video[ds]["pairs"][f"{a}__{b}"] = dict(diff=float(roc_auc_score(y, S[b]) - roc_auc_score(y, S[a])),
                                                diff_ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])])
for a, b in PAIRS_ALL:
    p = compare_pair(ROOT, a, b); j = json.loads(p.read_text(encoding="utf-8"))
    j["video_level"] = {ds: dict(auc_a=video[ds]["auc"][a], auc_b=video[ds]["auc"][b], **video[ds]["pairs"][f"{a}__{b}"],
                                 n_videos=video[ds]["n_real"] + video[ds]["n_fake"]) for ds in DS}
    j["note"] = NOTE
    p.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")

L = ["# Tổng hợp 5 arm (best.pt, ngưỡng logit >= 0)", "",
     "Tạo bởi `detector/scripts/video_bootstrap.py`. " + NOTE, "",
     "`raw_aug_filt_loc` **bị cắt ở epoch 8, chưa early stop** (xem `raw_aug_filt_loc/done.flag`); không có val. "
     "`raw_strongaug` không có val và code arm đã bị revert khỏi repo (commit 7053fe8).", "",
     "## Chỉ số theo frame", "", "| arm | dataset | ACC | AUC frame | PREC | RECALL | F1 | AUC video |", "|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    for ds in ("val", "ffpp_test", "celebdf"):
        x = res[a].get(ds)
        if not isinstance(x, dict) or "acc" not in str(x).lower() and "ACC" not in x: continue
        g = lambda *ks: next((x[k] for k in ks if k in x), float("nan"))
        L.append(f"| {a} | {ds} | {g('acc','ACC'):.4f} | {g('auc','AUC'):.4f} | {g('precision','PREC'):.4f} | "
                 f"{g('recall','RECALL'):.4f} | {g('f1','F1'):.4f} | {g('auc_video','AUC_video'):.4f} |")
for ds in DS:
    v = video[ds]
    L += ["", f"## {ds}: AUC video ({v['n_real']} real, {v['n_fake']} fake)", "", "| arm | AUC video | FPR@TPR95 |", "|---|---|---|"]
    L += [f"| {a} | {v['auc'][a]:.4f} | {v['fpr_at_tpr95'][a]:.3f} |" for a in ARMS]
    L += ["", "| cặp (b − a) | diff AUC video | CI95 bootstrap theo video |", "|---|---|---|"]
    for a, b in PAIRS_ALL:
        q = v["pairs"][f"{a}__{b}"]; L.append(f"| {b} − {a} | {q['diff']:+.4f} | [{q['diff_ci95'][0]:+.4f}, {q['diff_ci95'][1]:+.4f}] |")
(ROOT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
