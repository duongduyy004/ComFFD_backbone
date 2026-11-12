"""Tổng hợp mọi `scores_{dataset}.parquet` dưới runs_root thành bảng báo cáo.

Đầu ra: runs_root/report/{main.md, delong.csv, summary.parquet}.
Quy tắc trung thực:
- Điểm mức video tính TRƯỚC (`stats.video_scores`: trung bình logit theo video_id), mọi AUC/CI/DeLong dùng bảng video.
  Video có cờ no_usable_frame/decode_error (logit 0.0) vẫn nằm trong AUC và được đếm riêng trong chú thích.
- AUC video tính trên bảng video. CI: `bootstrap_ci` n=1000, alpha=0.05, seed 0, resample theo CỤM video nguồn
  `tid` cho ffpp_test ("{tid}_{src}"), các video cùng tid (real + mọi fake) cùng vào hoặc cùng ra; dataset khác: cụm = video.
- AUC frame: trên mọi dòng frame thật (bỏ dòng cờ frame_idx=-1), báo cạnh AUC video trong main.md.
- DeLong ghép cặp, gom cụm theo cùng cụm với CI (`stats.delong_paired_clustered`, Obuchowski 1997): các arm
  so từng cặp trong CÙNG dataset và CÙNG seed, qua
  `stats.paired_frames` trên video_id (chỉ giao các video chung; `n_dropped` = số video chỉ có ở một bên, ghi vào
  delong.csv và main.md).
- Đủ kết quả: mỗi arm kỳ vọng (EXPECTED_ARMS, kể cả arm vắng hoàn toàn) phải có đủ
  seed {0,1,2} x mọi dataset đã gặp ở đâu đó; thiếu thì KHÔNG raise mà liệt kê
  đầu main.md (mục CẢNH BÁO) và đánh dấu ô bị ảnh hưởng. CI trong main.md luôn lấy từ seed 0 (thiếu -> "CI: thiếu seed 0").
- Cùng (dataset, mode, split, seed) xuất hiện ở hai file scores -> ValueError; video có nhãn không nhất quán -> ValueError.
- run_info["config"] của các run arm (aug/) chỉ được khác nhau ở ARM_FREE_KEYS; khóa khác lệch -> liệt kê dưới CẢNH BÁO.
- Ô main.md: "AUC video trung bình các seed ± sd (ddof=1) [CI seed 0]"; một seed -> sd "n/a".
- Mỗi scores_*.parquet phải có best_step bằng step của best.pt HIỆN TẠI trong cùng thư mục run; lệch, thiếu best.pt
  hoặc nhiều giá trị -> liệt kê dưới CẢNH BÁO (điểm cũ không còn khớp checkpoint).
Mô-đun không import torch ở cấp module (chỉ import trễ để đọc step trong best.pt). Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from compsynth_ff.io_utils import atomic_write_bytes, write_parquet_atomic

from .stats import auc, bootstrap_ci, delong_paired_clustered, paired_frames, video_scores

REQUIRED = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag",
            "best_step"]
N_BOOT, ALPHA = 1000, 0.05
REQUIRED_SEEDS = (0, 1, 2)
EXPECTED_ARMS = ("raw", "raw+rep")
TID_CLUSTERED = ("ffpp_test",)


def _cluster(dataset: str, video_id: str) -> str:
    """Cụm bootstrap: tid video nguồn cho ffpp_test ("{tid}_{src}"); dataset khác -> video_id."""
    if dataset == "ffpp_test":
        return video_id.rsplit("_", 1)[0]
    return video_id


def _arm_label(mode: str, split: str) -> str:
    return mode


def _video_table(g: pd.DataFrame, key) -> pd.DataFrame:
    nl = g.groupby("video_id").label.nunique()
    if (nl > 1).any():
        raise ValueError(f"{key}: video {nl[nl > 1].index[0]!r} có nhãn không nhất quán")
    v = video_scores(g)
    meta = g.groupby("video_id").agg(cell=("cell", "first"),
                                     nuf=("flag", lambda s: (s == "no_usable_frame").any()),
                                     dec=("flag", lambda s: (s == "decode_error").any())).reset_index()
    v = v.merge(meta, on="video_id")
    v["cluster"] = [_cluster(key[0], x) for x in v.video_id]
    return v


def _frame_auc(g: pd.DataFrame) -> float:
    f = g[g.flag == ""]
    return auc(f.label, f.logit) if f.label.nunique() == 2 else float("nan")


def _fmt(x, p=4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def _best_pt_step(run: Path):
    """Step trong run/best.pt; None nếu không có file."""
    p = run / "best.pt"
    if not p.exists():
        return None
    import torch  # trễ: chỉ cần khi có best.pt

    return int(torch.load(p, map_location="cpu", weights_only=False)["step"])


def _load(runs_root: Path):
    files = sorted(p for p in runs_root.rglob("scores_*.parquet") if "report" not in p.relative_to(runs_root).parts)
    if not files:
        raise FileNotFoundError(f"Không có scores_*.parquet dưới {runs_root}")
    seen, tables, fauc, stale, best_cache, frames = {}, {}, {}, [], {}, {}
    for p in files:
        df = pd.read_parquet(p)
        miss = set(REQUIRED) - set(df.columns)
        if miss:
            raise ValueError(f"{p} thiếu cột {sorted(miss)}")
        df["split"] = df["split"].fillna("").astype(str)
        run = p.parent
        if run not in best_cache:
            best_cache[run] = _best_pt_step(run)
        steps, now = sorted({int(x) for x in df.best_step}), best_cache[run]
        if now is None or steps != [now]:
            stale.append((p.relative_to(runs_root).as_posix(), steps, now))
        for key, g in df.groupby(["dataset", "mode", "split", "seed"], sort=True):
            if key in seen:
                raise ValueError(f"Chạy trùng {key}: có ở {seen[key]} và {p}")
            seen[key] = p
            tables[key] = _video_table(g, key)
            fauc[key] = _frame_auc(g)
    return tables, fauc, stale, frames


def _runs_without_prereg(runs_root: Path) -> list:
    """Run dưới aug/ có run_info.json nhưng prereg_hash rỗng (đường dẫn tương đối, posix)."""
    out = []
    for top in ("aug",):
        for p in sorted((runs_root / top).rglob("run_info.json")):
            if not json.loads(p.read_text(encoding="utf-8")).get("prereg_hash"):
                out.append(p.parent.relative_to(runs_root).as_posix())
    return out


# Khóa config được phép khác nhau giữa các run arm (aug/); mọi khóa khác phải giống hệt.
ARM_FREE_KEYS = ("mode", "seed", "runs_root", "num_workers")


def _arm_config_diffs(runs_root: Path) -> list:
    """[(khóa, {run: giá trị})] cho các khóa trong run_info["config"] khác nhau giữa các run dưới aug/."""
    cfgs = {}
    for p in sorted((runs_root / "aug").rglob("run_info.json")):
        c = json.loads(p.read_text(encoding="utf-8")).get("config")
        if c is not None:
            cfgs[p.parent.relative_to(runs_root).as_posix()] = c
    keys = sorted({k for c in cfgs.values() for k in c} - set(ARM_FREE_KEYS))
    out = []
    for k in keys:
        vals = {r: json.dumps(c.get(k, "<thiếu>"), sort_keys=True, default=str) for r, c in cfgs.items()}
        if len(set(vals.values())) > 1:
            out.append((k, vals))
    return out


def build_report(runs_root) -> Path:
    runs_root = Path(runs_root)
    tables, fauc, stale, frames = _load(runs_root)
    out = runs_root / "report"

    # ---- summary
    rows = []
    for (ds, mode, split, seed), v in sorted(tables.items()):
        try:
            a, lo, hi = bootstrap_ci(v.label, v.score, v.cluster, n=N_BOOT, seed=0, alpha=ALPHA)
        except ValueError as e:
            raise ValueError(f"{(ds, mode, split, seed)}: không tính được AUC/CI ({e})") from e
        rows.append(dict(dataset=ds, mode=mode, split=split, seed=int(seed), auc=a, ci_lo=lo, ci_hi=hi,
                         auc_frame=fauc[(ds, mode, split, seed)], n_clusters=int(v.cluster.nunique()),
                         n_videos=len(v), n_no_usable_frame=int(v.nuf.sum()), n_decode_error=int(v.dec.sum())))
    summ = pd.DataFrame(rows)

    # ---- delong (chỉ arm, cùng dataset + cùng seed)
    dl = []
    arms = {k: v for k, v in tables.items() if k[1] != "compffd"}
    cnt = {}
    for k in arms:
        cnt.setdefault((k[0], k[3], k[1]), []).append(k)
    for k3, ks in cnt.items():
        if len(ks) != 1:
            raise ValueError(f"(dataset, seed, mode) = {k3} có {len(ks)} run (split khác nhau): {ks}; cần đúng một")
    for ds, seed in sorted({(k[0], k[3]) for k in arms}):
        modes = sorted(k[1] for k in arms if k[0] == ds and k[3] == seed)
        for ma, mb in combinations(modes, 2):
            ka = next(k for k in arms if k[0] == ds and k[3] == seed and k[1] == ma)
            kb = next(k for k in arms if k[0] == ds and k[3] == seed and k[1] == mb)
            cols = ["video_id", "label", "score", "cluster"]
            sa, sb, nd = paired_frames(tables[ka][cols], tables[kb][cols])
            if not (sa.label.to_numpy() == sb.label.to_numpy()).all():
                raise ValueError(f"{ds} seed {seed}: {ma} và {mb} gán nhãn khác nhau cho cùng video")
            d, z, p = delong_paired_clustered(sa.label, sa.score, sb.score, sa.cluster)
            dl.append(dict(dataset=ds, seed=int(seed), arm_a=ma, arm_b=mb,
                           auc_a=auc(sa.label, sa.score), auc_b=auc(sb.label, sb.score),
                           diff=d, z=z, p=p, n_videos=len(sa), n_clusters=int(sa.cluster.nunique()),
                           n_dropped=int(nd)))
    delong = pd.DataFrame(dl, columns=["dataset", "seed", "arm_a", "arm_b", "auc_a", "auc_b", "diff", "z", "p",
                                       "n_videos", "n_clusters", "n_dropped"])

    # ---- main.md
    summ["arm"] = [_arm_label(m, s) for m, s in zip(summ["mode"], summ.split)]
    dss = sorted(summ.dataset.unique())

    arm_of = {(m, sp): _arm_label(m, sp) for m, sp in zip(summ["mode"], summ.split)}
    present = {(arm_of[(k[1], k[2])], k[0], int(k[3])) for k in tables}
    all_ds = sorted({k[0] for k in tables})
    absent_arms = [a for a in EXPECTED_ARMS if a not in {x[0] for x in present}]
    missing = [(a, s_, d) for a in sorted({x[0] for x in present} | set(EXPECTED_ARMS)) for d in all_ds
               for s_ in REQUIRED_SEEDS if (a, d, s_) not in present]

    def cell(g, arm=None, d=None):
        miss = sorted(s_ for a, s_, dd in missing if a == arm and dd == d)
        warn = f" ⚠ thiếu seed {miss}" if miss else ""
        if g is None or g.empty:
            return "– ⚠ thiếu" if arm else "–"
        g = g.sort_values("seed")
        sd = g.auc.std(ddof=1) if len(g) > 1 else float("nan")
        f0 = g[g.seed == 0]
        ci = f"[{_fmt(f0.iloc[0].ci_lo)}, {_fmt(f0.iloc[0].ci_hi)}]" if len(f0) else "CI: thiếu seed 0"
        return f"{_fmt(g.auc.mean())} ± {_fmt(sd)} {ci} (n_seed={len(g)}){warn}"

    def fcell(g):
        if g is None or g.empty:
            return "–"
        sd = g.auc_frame.std(ddof=1) if len(g) > 1 else float("nan")
        return f"{_fmt(g.auc_frame.mean())} ± {_fmt(sd)}"

    def table(arm_names):
        head = " | ".join(f"{d} (video) | {d} (frame)" for d in dss)
        lines = ["| arm | " + head + " |", "|---|" + "---|---|" * len(dss)]
        for a in arm_names:
            parts = []
            for d in dss:
                g = summ[(summ.arm == a) & (summ.dataset == d)]
                parts += [cell(g, a, d), fcell(g)]
            lines.append(f"| {a} | " + " | ".join(parts) + " |")
        return lines

    arm_names = sorted(set(summ.arm.unique()) | set(EXPECTED_ARMS))
    md = ["# AUC video theo arm x dataset", "",
          f"Cột (video) = AUC video trung bình các seed ± sd [CI bootstrap {int((1 - ALPHA) * 100)}%, seed 0, n={N_BOOT}]; "
          "cột (frame) = AUC mức frame trung bình các seed ± sd.", ""]
    noprereg = _runs_without_prereg(runs_root)
    cfg_diffs = _arm_config_diffs(runs_root)
    if missing or noprereg or cfg_diffs or stale:
        md += ["## CẢNH BÁO: thiếu kết quả (báo cáo chưa đầy đủ)", ""]
    if absent_arms:
        md += [f"Arm kỳ vọng VẮNG HOÀN TOÀN (không có file scores nào): {', '.join(absent_arms)}", ""]
    if missing:
        md += [f"Cần seed {list(REQUIRED_SEEDS)} x mọi dataset cho mỗi arm. Thiếu {len(missing)} tổ hợp (arm, seed, dataset):", ""]
        md += [f"- {a}, seed {s_}, {d}" for a, s_, d in missing] + [""]
    if noprereg:
        md += [f"Run KHÔNG có prereg_hash trong run_info.json ({len(noprereg)}; không thuộc thiết kế đã đăng ký trước):", ""]
        md += [f"- {r}" for r in noprereg] + [""]
    if stale:
        md += [f"File scores KHÔNG khớp best.pt hiện tại ({len(stale)}; cần eval lại):", ""]
        md += [f"- {f}: best_step trong scores = {s}, best.pt = {'thiếu' if n is None else n}" for f, s, n in stale] + [""]
    if cfg_diffs:
        md += [f"Config các run arm khác nhau ngoài {list(ARM_FREE_KEYS)} ({len(cfg_diffs)} khóa; so sánh arm KHÔNG có kiểm soát):", ""]
        md += [f"- {k}: " + "; ".join(f"{r}={v}" for r, v in sorted(vals.items())) for k, vals in cfg_diffs] + [""]
    md += table(arm_names)
    md += ["", "## Chú thích", "",
           "- AUC video: điểm video = trung bình logit các frame.",
           f"- CI: bootstrap percentile seed 0, n={N_BOOT}, resample theo CỤM video nguồn tid cho "
           f"{', '.join(TID_CLUSTERED)} (real và mọi fake/ô của cùng tid cùng vào/ra); dataset khác: cụm = video.",
           "- DeLong (delong.csv): ghép cặp trên video chung, GOM CỤM theo cùng cụm với CI (tid cho "
           f"{', '.join(TID_CLUSTERED)}; dataset khác: cụm = video), phương sai theo Obuchowski (1997).",
           "- AUC frame: mọi frame thật, bỏ dòng cờ (frame_idx=-1)."]
    for d in dss:
        sd_ = summ[summ.dataset == d]
        nv = sd_.groupby("arm").n_videos.agg(["min", "max"])
        nuf = int(sd_.groupby("arm").n_no_usable_frame.max().max())
        dec = int(sd_.groupby("arm").n_decode_error.max().max())
        nd = int(delong[delong.dataset == d].n_dropped.sum()) if len(delong) else 0
        md.append(f"- {d}: n_videos {int(nv['min'].min())}-{int(nv['max'].max())}; "
                  f"video no_usable_frame = {nuf}, decode_error = {dec} (tính trong AUC với score 0.0); "
                  f"video bị loại khi ghép cặp (n_dropped, cộng mọi cặp) = {nd}")
    md += ["", f"Tổng n_dropped khi ghép cặp: {int(delong.n_dropped.sum()) if len(delong) else 0} (chi tiết: delong.csv)."]
    main_md = "\n".join(md) + "\n"

    # ---- ghi
    atomic_write_bytes(out / "main.md", main_md.encode("utf-8"))
    atomic_write_bytes(out / "delong.csv", delong.to_csv(index=False).encode("utf-8"))
    write_parquet_atomic(summ.drop(columns="arm"), out / "summary.parquet")
    return out
