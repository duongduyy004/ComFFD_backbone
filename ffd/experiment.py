"""Chạy chính (`CompFFD/ke_hoach_run_v2_2026-11-26.md`): train + eval R0 `raw`, R1 `raw_aug`, R2 `raw_rep`, rồi so sánh
mức video và áp tiêu chí ghi trước.

- Dữ liệu: manifest FF++ raw và Celeb-DF phải có cột `w`, `h`, `jpeg_q` (preprocessing hiện tại: JPEG kích thước gốc);
  thiếu Celeb-DF -> FileNotFoundError.
- Run đã có trên đĩa mà `data_hashes` (manifest raw, params.json) khác dữ liệu hiện tại -> ValueError trước khi train
  run nào (`config_hash` không chứa data_root nên không tự phát hiện ckpt_root cũ).
- Không còn pilot: `lr`, `p_jpeg` và mọi trường khác lấy từ config. Trước khi train in khối config hiệu lực
  (`EFFECTIVE_KEYS`).
- Quy tắc chạy tiếp một run (`ensure_run`): có result.json đủ mọi dataset yêu cầu -> bỏ qua; result.json thiếu dataset
  yêu cầu (hoặc celebdf skipped mà nay đã có dữ liệu) -> chỉ eval lại; có done.flag -> chỉ eval; có last.pt ->
  train(resume=True); không có gì -> train mới (còn file dở từ lần chết trước epoch 1 thì force). Không bao giờ truyền
  resume và force cùng lúc. run_info.json có config_hash khác config hiện tại -> ValueError.
- So sánh: `videoboot` gốc `raw` (ΔAUC video, CI95), `realfp` raw→raw_aug (mô tả) và raw_aug→raw_rep (tiêu chí R2),
  Spearman(face_px, logit) trên frame real Celeb-DF (kiểm cơ chế, mô tả).
- Tiêu chí: R1 vs R0 theo CI95 ΔAUC video Celeb-DF (+ ΔAUC video FF++ test >= -0.005); R2 vs R1 theo verdict `realfp`.
Ghi `<ckpt_root>/summary.{json,md}`. Exploratory, 1 seed.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from . import realfp, videoboot
from .config import ARMS, TrainConfig, config_hash
from .data import ffpp as D
from .data.ffpp import celebdf_dir
from .evaluate import DATASETS, evaluate_run
from .io_utils import write_json_atomic
from .rundir import run_dir
from .tables import write_text_atomic
from .train import train

MAIN_DATASETS = ("val", "ffpp_test", "celebdf")
# Khóa in ra đầu `main` (config hiệu lực sau --set).
EFFECTIVE_KEYS = ("data_root", "ckpt_root", "lr", "p_jpeg", "batch_size", "max_epochs", "patience", "seed",
                  "grad_checkpointing", "model_name", "model_revision", "num_workers")


def effective_settings(cfg: TrainConfig) -> str:
    """Khối chữ gọn: một dòng `khóa: giá trị` cho mỗi khóa trong EFFECTIVE_KEYS."""
    w = max(map(len, EFFECTIVE_KEYS))
    return "\n".join(f"  {k.ljust(w)} : {getattr(cfg, k)}" for k in EFFECTIVE_KEYS)


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
        rec = json.loads(info.read_text(encoding="utf-8"))
        old = rec.get("config_hash")
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


DATA_COLS = ("w", "h", "jpeg_q")
FFPP_TOL = -0.005  # ΔAUC video FF++ test (R1 − R0) tối thiểu, điểm ước lượng
NOTE = ("Exploratory: 1 seed/arm, không prereg; tiêu chí ghi trước trong ke_hoach_run_v2_2026-11-26.md. "
        "CI bootstrap theo video chỉ phản ánh lấy mẫu video test, không phản ánh dao động giữa seed.")


def check_data(data_root) -> None:
    """ValueError nếu data_root không phải dữ liệu của preprocessing hiện tại; FileNotFoundError nếu thiếu manifest (kể cả Celeb-DF)."""
    for p in (D.ffpp_dir(data_root) / "manifest_raw.parquet", D.celebdf_dir(data_root) / "manifest.parquet"):
        if not p.exists():
            raise FileNotFoundError(f"Thiếu {p}: main cần FF++ và Celeb-DF đã tiền xử lý")
        cols = set(pd.read_parquet(p, columns=None).columns)
        miss = [c for c in DATA_COLS if c not in cols]
        if miss:
            raise ValueError(f"{p} thiếu cột {miss}: không phải dữ liệu của preprocessing hiện tại (JPEG kích thước gốc)")


def check_run_data(cfg: TrainConfig) -> None:
    """Run đã có run_info.json: data_hashes lúc train phải khớp dữ liệu hiện tại."""
    p = run_dir(cfg) / "run_info.json"
    if not p.exists():
        return
    rec = json.loads(p.read_text(encoding="utf-8")).get("data_hashes") or {}
    now = D.data_hashes(cfg.data_root)
    if (rec.get("manifest_sha256") or {}).get("raw") != now["manifest_sha256"]["raw"] \
            or rec.get("params_json_sha256") != now["params_json_sha256"]:
        raise ValueError(f"{p.parent}: run được train trên dữ liệu khác data_root hiện tại; "
                         "dùng ckpt_root mới")


def ci_verdict(ci, pos: str, neg: str) -> str:
    lo, hi = ci
    return pos if lo > 0 else neg if hi < 0 else "null"


def face_px_spearman(run: Path, data_root) -> dict:
    """Spearman(face_px, logit) trên frame real Celeb-DF; ghép scores với manifest theo (video_id, frame_idx)."""
    s = pd.read_parquet(run / "scores_celebdf.parquet")
    m = pd.read_parquet(D.celebdf_dir(data_root) / "manifest.parquet", columns=["video_id", "frame_idx", "face_px"])
    m["video_id"] = m.video_id.astype(str)
    s = s.assign(video_id=s.video_id.astype(str))
    j = s[s.label == 0].merge(m, on=["video_id", "frame_idx"], how="inner")
    if len(j) != int((s.label == 0).sum()):
        raise ValueError(f"{run}: {int((s.label == 0).sum()) - len(j)} frame real Celeb-DF không ghép được face_px")
    return {"rho": float(j.face_px.corr(j.logit, method="spearman")), "n_frames": int(len(j))}


def decide(vb: dict, fp_rep: dict) -> dict:
    """Áp tiêu chí ghi trước. vb = kết quả videoboot gốc raw; fp_rep = realfp raw_aug→raw_rep."""
    c = vb["celebdf"]["pairs"]["raw__raw_aug"]
    f = vb["ffpp_test"]["pairs"]["raw__raw_aug"]
    r1 = ci_verdict(c["auc_ci95"], "aug_helps", "aug_hurts")
    return {
        "R1_vs_R0": {"verdict": r1, "celebdf_auc_diff": c["auc_diff"], "celebdf_auc_ci95": c["auc_ci95"],
                     "ffpp_test_auc_diff": f["auc_diff"], "ffpp_test_ok": bool(f["auc_diff"] >= FFPP_TOL)},
        "R2_vs_R1": {"verdict": fp_rep["primary"]["verdict"], "real_fp_diff": fp_rep["primary"]["diff"],
                     "real_fp_ci95": fp_rep["primary"]["ci95"]},
        "run_R3": bool(r1 == "aug_helps" and f["auc_diff"] >= FFPP_TOL),
    }


def _f(x, n=4):
    return f"{x:+.{n}f}"


def summary_markdown(out: dict) -> str:
    d, vb = out["decision"], out["videoboot"]
    L = ["# Tổng hợp main (R0 raw, R1 raw_aug, R2 raw_rep)", "", out["note"], "",
         "## AUC video", "", "| arm | FF++ test | Celeb-DF |", "|---|---|---|"]
    L += [f"| {a} | {vb['ffpp_test']['auc'][a]:.4f} | {vb['celebdf']['auc'][a]:.4f} |" for a in ARMS]
    r1, r2 = d["R1_vs_R0"], d["R2_vs_R1"]
    lo, hi = r1["celebdf_auc_ci95"]
    L += ["", "## Tiêu chí", "",
          "| so sánh | chỉ số | giá trị [CI95] | kết luận |", "|---|---|---|---|",
          f"| R1 − R0 | ΔAUC video Celeb-DF | {_f(r1['celebdf_auc_diff'])} [{_f(lo)}, {_f(hi)}] | `{r1['verdict']}` |",
          f"| R1 − R0 | ΔAUC video FF++ test (≥ {FFPP_TOL}) | {_f(r1['ffpp_test_auc_diff'])} | "
          f"{'đạt' if r1['ffpp_test_ok'] else '**không đạt**'} |",
          f"| R2 − R1 | ΔFP real Celeb-DF (frame, ngưỡng TPR 0.95 val) | {_f(r2['real_fp_diff'])} "
          f"[{_f(r2['real_fp_ci95'][0])}, {_f(r2['real_fp_ci95'][1])}] | `{r2['verdict']}` |",
          "", f"Chạy R3 (arm loc, không lọc): **{'có' if d['run_R3'] else 'không'}** (chỉ khi R1 `aug_helps` và FF++ đạt).",
          "", "## Kiểm cơ chế (mô tả): Spearman(face_px, logit), frame real Celeb-DF", "",
          "| arm | rho | n frame |", "|---|---|---|"]
    L += [f"| {a} | {v['rho']:+.3f} | {v['n_frames']} |" for a, v in out["mechanism"].items()]
    L += ["", "## Số ảnh train", "", "| arm | aug_stats |", "|---|---|"]
    L += [f"| {a} | {json.dumps(out['aug_stats'].get(a, {}), ensure_ascii=False)} |" for a in ARMS]
    L += ["", "Chi tiết: `videoboot.md`, `realfp_raw__raw_aug.md`, `realfp_raw_aug__raw_rep.md`."]
    return "\n".join(L) + "\n"


def run_main(base_cfg: TrainConfig, *, B: int = 4000, vision_config=None, allow_cpu: bool = False,
                device=None) -> Path:
    """Train + eval 3 arm, so sánh, áp tiêu chí. Trả đường dẫn summary.json."""
    root = Path(base_cfg.ckpt_root)
    check_data(base_cfg.data_root)
    cfgs = {a: dataclasses.replace(base_cfg, arm=a, run_name=a) for a in ARMS}
    for c in cfgs.values():  # kiểm hết trước khi train run nào
        check_run_data(c)
    print(f"Config hiệu lực của main ({', '.join(ARMS)}):\n{effective_settings(base_cfg)}", flush=True)
    for a, c in cfgs.items():
        ensure_run(c, MAIN_DATASETS, vision_config=vision_config, allow_cpu=allow_cpu, device=device)
    vb = videoboot.compare_video(root, "raw", ["raw_aug", "raw_rep"], B=B, name="videoboot")
    realfp.compare_real_fp(root, "raw", "raw_aug", B=B)
    fp_rep = realfp.compare_real_fp(root, "raw_aug", "raw_rep", B=B)
    aug_stats = {}
    for a, c in cfgs.items():
        info = json.loads((run_dir(c) / "run_info.json").read_text(encoding="utf-8"))
        aug_stats[a] = info.get("aug_stats", {})
    out = {"arms": list(ARMS), "B": int(B), "note": NOTE,
           "videoboot": {ds: {k: vb[ds][k] for k in ("auc", "fpr_at_tpr95", "pairs", "n_real", "n_fake")}
                         for ds in videoboot.DATASETS},
           "mechanism": {a: face_px_spearman(run_dir(c), base_cfg.data_root) for a, c in cfgs.items()},
           "aug_stats": aug_stats, "decision": decide(vb, fp_rep)}
    p = root / "summary.json"
    write_json_atomic(out, p)
    md = summary_markdown(out)
    write_text_atomic(md, root / "summary.md")
    print(md, flush=True)
    return p
