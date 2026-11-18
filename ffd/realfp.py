"""So sánh tỉ lệ real bị đoán fake của hai run ở ngưỡng chọn trên FF++ val (spec v5, arm raw_rep_filt).

Ngưỡng t của mỗi arm: TPR >= 0.95 trên fake val của chính arm đó; đoán fake <=> logit >= t. Ngưỡng cố định, không
tính lại trong bootstrap; không dùng FF++ test / Celeb-DF để chọn gì.
Chỉ số chính: tỉ lệ frame real Celeb-DF bị đoán fake, diff = b - a, CI95 paired bootstrap trên video real
(tổng frame bị gắn fake / tổng frame), B = 4000, seed 0. AUC video: bootstrap phân tầng theo lớp.
Đọc `<ckpt_root>/<arm>/scores_{val,ffpp_test,celebdf}.parquet`; ghi `realfp_<a>__<b>.json` và `.md` vào ckpt_root.
Dùng: python -m ffd.realfp --config config.yaml --a raw_aug_filt --b raw_rep_filt
Trạng thái: chưa chạy thử trên dữ liệu thật của raw_rep_filt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .io_utils import write_json_atomic

DATASETS = ("val", "ffpp_test", "celebdf")
TPR = 0.95
SEED = 0
DEFAULT_PAIR = ("raw_aug_filt", "raw_rep_filt")  # a có seam, b bỏ seam: cặp mà verdict được định nghĩa
NOTE = ("Exploratory: 1 seed/arm, không prereg (ngoại lệ v2); tiêu chí đặt trước trong spec v5. CI bootstrap theo "
        "video chỉ phản ánh lấy mẫu video test, không phản ánh dao động giữa seed.")
VERDICT_TEXT = {
    "seam_hurts_real": "seam góp phần làm real bị đoán fake (exploratory); bắt buộc thêm seed 1, 2 cho cả hai arm",
    "inconclusive": "không có bằng chứng seam là nguyên nhân; đóng hướng bỏ seam",
    "seam_helps_real": "seam giúp lớp real; giữ seam",
}


def video_table(df: pd.DataFrame) -> pd.DataFrame:
    """Index video_id (sắp xếp); cột label, s (logit trung bình), n (số frame)."""
    return df.groupby("video_id").agg(label=("label", "first"), s=("logit", "mean"), n=("logit", "size"))


def val_threshold(df: pd.DataFrame, tpr: float = TPR, level: str = "frame") -> float:
    """t = f[floor((1 - tpr) * len(f))], f = điểm fake tăng dần (frame hoặc trung bình video)."""
    if level == "frame":
        f = df.loc[df.label == 1, "logit"].to_numpy()
    elif level == "video":
        v = video_table(df)
        f = v.loc[v.label == 1, "s"].to_numpy()
    else:
        raise ValueError(f"level phải là 'frame' hoặc 'video', nhận {level!r}")
    if not len(f):
        raise ValueError("không có mẫu fake để chọn ngưỡng")
    f = np.sort(f)
    return float(f[int(np.floor((1 - tpr) * len(f)))])


def real_fp_rate(df: pd.DataFrame, t: float) -> float:
    """Tỉ lệ frame real có logit >= t."""
    return float((df.loc[df.label == 0, "logit"] >= t).mean())


def _same_videos(va: pd.DataFrame, vb: pd.DataFrame) -> None:
    if not va.index.equals(vb.index):
        raise ValueError(f"tập video khác nhau giữa hai run ({len(va)} vs {len(vb)} video)")


def _real_video_counts(df: pd.DataFrame, t: float, level: str) -> tuple[np.ndarray, np.ndarray]:
    """Theo video real (sắp theo video_id): (mẫu số, tử số). frame: (n frame, n frame >= t); video: (1, s >= t)."""
    r = df[df.label == 0]
    g = r.assign(hit=(r.logit >= t).astype(int)).groupby("video_id").agg(n=("hit", "size"), k=("hit", "sum"),
                                                                       s=("logit", "mean"))
    if level == "frame":
        return g.n.to_numpy(float), g.k.to_numpy(float)
    if level == "video":
        return np.ones(len(g)), (g.s >= t).to_numpy(float)
    raise ValueError(f"level phải là 'frame' hoặc 'video', nhận {level!r}")


def boot_real_fp_diff(df_a, df_b, t_a: float, t_b: float, B: int = 4000, seed: int = SEED,
                      level: str = "frame") -> dict:
    """Paired bootstrap trên video real: rate = tổng tử / tổng mẫu của các video được bốc. diff = b - a."""
    _same_videos(video_table(df_a), video_table(df_b))
    n_a, k_a = _real_video_counts(df_a, t_a, level)
    n_b, k_b = _real_video_counts(df_b, t_b, level)
    rate_a, rate_b = k_a.sum() / n_a.sum(), k_b.sum() / n_b.sum()
    idx = np.random.default_rng(seed).integers(0, len(n_a), size=(B, len(n_a)))
    d = k_b[idx].sum(1) / n_b[idx].sum(1) - k_a[idx].sum(1) / n_a[idx].sum(1)
    return {"rate_a": float(rate_a), "rate_b": float(rate_b), "diff": float(rate_b - rate_a),
            "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}


def boot_auc_video_diff(df_a, df_b, B: int = 4000, seed: int = SEED) -> dict:
    """AUC video (logit trung bình mỗi video), paired bootstrap phân tầng theo lớp. diff = b - a."""
    va, vb = video_table(df_a), video_table(df_b)
    _same_videos(va, vb)
    y, sa, sb = va.label.to_numpy(), va.s.to_numpy(), vb.s.to_numpy()
    r, k = np.where(y == 0)[0], np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for i in range(B):
        ii = np.r_[rng.choice(r, len(r)), rng.choice(k, len(k))]
        d[i] = roc_auc_score(y[ii], sb[ii]) - roc_auc_score(y[ii], sa[ii])
    auc_a, auc_b = float(roc_auc_score(y, sa)), float(roc_auc_score(y, sb))
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": auc_b - auc_a,
            "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}


def verdict(ci95) -> str:
    lo, hi = ci95
    if hi < 0:
        return "seam_hurts_real"
    if lo > 0:
        return "seam_helps_real"
    return "inconclusive"


def _load(ckpt_root: Path, arms) -> dict:
    paths = {(a, ds): ckpt_root / a / f"scores_{ds}.parquet" for a in arms for ds in DATASETS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("thiếu file điểm: " + ", ".join(missing))
    return {k: pd.read_parquet(p) for k, p in paths.items()}


def _sources(df: pd.DataFrame) -> pd.Series | None:
    """Nguồn real Celeb-DF từ tiền tố video_id '<nguồn>__'; None nếu có video real không có tiền tố."""
    ids = df.loc[df.label == 0, "video_id"].astype(str)
    if not ids.str.contains("__", regex=False).all():
        return None
    return df.video_id.astype(str).str.split("__", n=1).str[0]


def compare_real_fp(ckpt_root, a: str, b: str, B: int = 4000) -> dict:
    ckpt_root = Path(ckpt_root)
    S = _load(ckpt_root, (a, b))
    th = {arm: {lvl: val_threshold(S[arm, "val"], level=lvl) for lvl in ("frame", "video")} for arm in (a, b)}
    ca, cb = S[a, "celebdf"], S[b, "celebdf"]
    primary = boot_real_fp_diff(ca, cb, th[a]["frame"], th[b]["frame"], B=B)
    primary["verdict"] = verdict(primary["ci95"])
    ffpp_auc = boot_auc_video_diff(S[a, "ffpp_test"], S[b, "ffpp_test"], B=B)
    sec = {
        "celebdf_video_fp": boot_real_fp_diff(ca, cb, th[a]["video"], th[b]["video"], B=B, level="video"),
        "ffpp_test_real_fp": boot_real_fp_diff(S[a, "ffpp_test"], S[b, "ffpp_test"], th[a]["frame"],
                                               th[b]["frame"], B=B),
        "celebdf_auc_video": boot_auc_video_diff(ca, cb, B=B),
        "ffpp_test_auc_video": ffpp_auc,
        "celebdf_real_fp_logit0": {a: real_fp_rate(ca, 0.0), b: real_fp_rate(cb, 0.0)},
        "inhomain_drop_gt_0_01": bool(ffpp_auc["auc_a"] - ffpp_auc["auc_b"] > 0.01),
    }
    src = _sources(ca)
    if src is not None:
        sb = _sources(cb)
        real_src = sorted(src[ca.label == 0].unique())
        sec["by_source"] = {s: {a: real_fp_rate(ca[src == s], th[a]["frame"]),
                                b: real_fp_rate(cb[sb == s], th[b]["frame"])} for s in real_src}
    out = {"a": a, "b": b, "tpr": TPR, "thresholds": th, "primary": primary, "secondary": sec,
           "B": int(B), "seed": SEED, "note": NOTE}
    write_json_atomic(out, ckpt_root / f"realfp_{a}__{b}.json")
    (ckpt_root / f"realfp_{a}__{b}.md").write_text(_markdown(out), encoding="utf-8")
    return out


def _ci(x: dict) -> str:
    return f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def _markdown(o: dict) -> str:
    a, b, p, s, th = o["a"], o["b"], o["primary"], o["secondary"], o["thresholds"]
    L = [f"# Real bị đoán fake ở ngưỡng FF++ val: {b} so với {a}", "", o["note"], "",
         f"Ngưỡng (TPR = {o['tpr']} trên fake val; đoán fake khi logit >= t). B = {o['B']}, seed {o['seed']}.", "",
         "| arm | t frame | t video |", "|---|---|---|"]
    L += [f"| {arm} | {th[arm]['frame']:.4f} | {th[arm]['video']:.4f} |" for arm in (a, b)]
    L += ["", "## Chỉ số chính: tỉ lệ frame real Celeb-DF bị đoán fake", "",
          f"| {a} | {b} | diff ({b} − {a}) | CI95 bootstrap theo video real |", "|---|---|---|---|",
          f"| {p['rate_a']:.4f} | {p['rate_b']:.4f} | {p['diff']:+.4f} | {_ci(p)} |", "",
          f"**Kết luận theo tiêu chí đặt trước:** `{p['verdict']}`: {VERDICT_TEXT[p['verdict']]}.", ""]
    if (a, b) != DEFAULT_PAIR:
        L += [f"_Cảnh báo: nhãn kết luận chỉ có nghĩa cho cặp {DEFAULT_PAIR[0]} → {DEFAULT_PAIR[1]} (a có seam, "
              f"b bỏ seam); với cặp {a} → {b} chỉ đọc dấu và CI của diff._", ""]
    L += [
          "## Chỉ số phụ (mô tả)", "", f"| chỉ số | {a} | {b} | diff | CI95 |", "|---|---|---|---|---|"]
    for name, key in (("real Celeb-DF bị đoán fake, theo video", "celebdf_video_fp"),
                      ("real FF++ test bị đoán fake, theo frame", "ffpp_test_real_fp")):
        x = s[key]
        L.append(f"| {name} | {x['rate_a']:.4f} | {x['rate_b']:.4f} | {x['diff']:+.4f} | {_ci(x)} |")
    for name, key in (("AUC video Celeb-DF", "celebdf_auc_video"), ("AUC video FF++ test", "ffpp_test_auc_video")):
        x = s[key]
        L.append(f"| {name} | {x['auc_a']:.4f} | {x['auc_b']:.4f} | {x['diff']:+.4f} | {_ci(x)} |")
    z = s["celebdf_real_fp_logit0"]
    L.append(f"| real Celeb-DF bị đoán fake, logit >= 0 | {z[a]:.4f} | {z[b]:.4f} | {z[b] - z[a]:+.4f} | |")
    L += ["", "AUC video FF++ test giảm > 0.01 (hại in-domain): " + ("**có**" if s["inhomain_drop_gt_0_01"] else "không"),
          "", "## Theo nguồn real Celeb-DF (ngưỡng frame của từng arm)", ""]
    if "by_source" in s:
        L += [f"| nguồn | {a} | {b} |", "|---|---|---|"]
        L += [f"| {src} | {v[a]:.4f} | {v[b]:.4f} |" for src, v in s["by_source"].items()]
    else:
        L.append("_video_id không có tiền tố `<nguồn>__`: không tách được theo nguồn._")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ffd.realfp", description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="config.yaml của train; lấy ckpt_root từ đây")
    g.add_argument("--ckpt-root", help="thư mục chứa các run (thay cho --config)")
    ap.add_argument("--a", default=DEFAULT_PAIR[0])
    ap.add_argument("--b", default=DEFAULT_PAIR[1])
    ap.add_argument("--B", type=int, default=4000)
    args = ap.parse_args(argv)
    if args.config:
        from .config import load_train_config
        root = load_train_config(args.config).ckpt_root
    else:
        root = Path(args.ckpt_root)
    out = compare_real_fp(root, args.a, args.b, B=args.B)
    print((Path(root) / f"realfp_{args.a}__{args.b}.md").read_text(encoding="utf-8"))
    print(f"verdict: {out['primary']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
