"""Test báo cáo (torch chỉ dùng để ghi best.pt giả). Trạng thái: chưa chạy thử.
Toàn bộ scores.parquet ở đây là TỔNG HỢP (MINH HỌA), không phải kết quả thật."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ffd.report import build_report

torch = pytest.importorskip("torch")  # chỉ để ghi best.pt giả (MINH HỌA) cho kiểm tra best_step

COLS = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag",
        "best_step"]
BEST_STEP = 10
ARMS = ("raw", "raw+comp", "raw+sb")
STRENGTH = {"raw": 0.8, "raw+comp": 1.6, "raw+sb": 1.2}


def fake_scores(mode, seed, dataset="ffpp_test", nvid=20, split="", drop=(), flagged=(), cells=False):
    """MINH HỌA: logit = nhãn * độ mạnh + nhiễu; mỗi video 3 frame."""
    rng = np.random.default_rng(1000 * seed + len(mode))
    rows = []
    for k in range(2 * nvid):
        lab = k % 2
        vid = f"v{k:03d}"
        if vid in drop:
            continue
        cell = ("eyes+mouth|DF" if k % 4 == 1 else "nose|F2F") if (cells and lab) else "real"
        for f in range(3):
            lg = rng.normal(lab * STRENGTH[mode], 1.0)
            flag = ""
            if vid in flagged:
                lg, f, flag = 0.0, -1, "no_usable_frame"
            rows.append((f"{mode}/seed{seed}", mode, seed, split, dataset, vid, f, lab, lg, cell, flag, BEST_STEP))
            if flag:
                break
    return pd.DataFrame(rows, columns=COLS)


def put(root, df, mode, seed, split=""):
    d = root / "aug" / mode / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "best.pt").exists():  # report đối chiếu best_step trong scores với best.pt (I6)
        torch.save({"step": BEST_STEP, "trainable": {}}, d / "best.pt")
    for ds, g in df.groupby("dataset"):  # một file mỗi dataset, như evaluate_run
        g.to_parquet(d / f"scores_{ds}.parquet", index=False)


@pytest.fixture
def tree(tmp_path):  # 3 arm x 3 seed x 1 dataset (MINH HỌA)
    for m in ARMS:
        for s in range(3):
            put(tmp_path, fake_scores(m, s, flagged=("v004",) if m == "raw" and s == 0 else ()), m, s)
    return tmp_path


def test_report_tables(tree):
    out = build_report(tree)
    assert {p.name for p in out.iterdir()} == {"main.md", "per_cell.md", "delong.csv", "summary.parquet"}
    d = pd.read_csv(out / "delong.csv")
    assert set(d.columns) >= {"dataset", "seed", "arm_a", "arm_b", "diff", "z", "p", "n_videos", "n_dropped"}
    assert len(d) == 3 * 3 * 1  # 3 cặp arm x 3 seed x 1 dataset
    assert (d.n_dropped == 0).all() and (d.n_videos == 40).all()
    assert {tuple(x) for x in d[["arm_a", "arm_b"]].itertuples(index=False)} == set(combinations(sorted(ARMS), 2))
    s = pd.read_parquet(out / "summary.parquet")
    assert len(s) == 9 and {"auc", "ci_lo", "ci_hi", "n_videos", "n_no_usable_frame"} <= set(s.columns)
    assert (s.ci_lo <= s.auc).all() and (s.auc <= s.ci_hi).all()
    main = (out / "main.md").read_text(encoding="utf-8")
    assert all(a in main for a in ARMS) and "ffpp_test" in main and "±" in main
    assert "no_usable_frame" in main and "n_dropped" in main


def test_flagged_video_counted_not_dropped(tree):
    s = pd.read_parquet(build_report(tree) / "summary.parquet")
    r = s[(s["mode"] == "raw") & (s.seed == 0)].iloc[0]
    assert r.n_no_usable_frame == 1 and r.n_videos == 40  # video lỗi vẫn nằm trong AUC


def test_pairing_reports_dropped_videos(tree):
    put(tree, fake_scores("raw+sb", 1, drop=("v010", "v011")), "raw+sb", 1)
    d = pd.read_csv(build_report(tree) / "delong.csv")
    hit = d[(d.seed == 1) & ((d.arm_a == "raw+sb") | (d.arm_b == "raw+sb"))]
    assert len(hit) == 2 and (hit.n_dropped == 2).all() and (hit.n_videos == 38).all()
    assert (d[~d.index.isin(hit.index)].n_dropped == 0).all()


def test_duplicate_run_rejected(tree):
    p = tree / "aug" / "raw" / "seed0" / "scores_ffpp_test.parquet"
    d = tree / "aug" / "dup" / "seed0"
    d.mkdir(parents=True)
    pd.read_parquet(p).to_parquet(d / "scores_ffpp_test.parquet", index=False)  # cùng mode/seed/dataset ở nơi khác
    with pytest.raises(ValueError, match="trùng"):
        build_report(tree)


def test_inconsistent_label_rejected(tmp_path):
    df = fake_scores("raw", 0)
    df.loc[df.video_id == "v000", "label"] = [0, 1, 0]
    put(tmp_path, df, "raw", 0)
    with pytest.raises(ValueError, match="nhãn"):
        build_report(tmp_path)


def test_no_scores_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path)


def test_per_cell_and_mixed_datasets(tmp_path):
    for m in ARMS:
        for s in range(2):
            df = pd.concat([fake_scores(m, s), fake_scores(m, s, dataset="compsynth_test", cells=True)])
            put(tmp_path, df, m, s)
    out = build_report(tmp_path)
    pc = (out / "per_cell.md").read_text(encoding="utf-8")
    assert "eyes+mouth|DF" in pc and "nose|F2F" in pc
    assert "## AUC theo method" in pc and "## AUC theo số vùng k'" in pc
    by_m = pc[pc.index("## AUC theo method"):pc.index("## AUC theo số vùng k'")]
    assert "| DF |" in by_m and "| F2F |" in by_m
    by_k = pc[pc.index("## AUC theo số vùng k'"):]
    assert "| 1 |" in by_k and "| 2 |" in by_k
    s = pd.read_parquet(out / "summary.parquet")
    assert set(s.dataset) == {"ffpp_test", "compsynth_test"}
    d = pd.read_csv(out / "delong.csv")
    assert len(d) == 2 * 2 * 3  # 2 seed x 2 dataset x 3 cặp


def test_complete_tree_has_no_warning(tree):
    main = (build_report(tree) / "main.md").read_text(encoding="utf-8")
    assert "CẢNH BÁO" not in main and "⚠" not in main


def test_missing_combinations_listed_not_raised(tmp_path):
    # raw đủ 3 seed x 2 dataset; raw+comp thiếu seed 2 ở ffpp_test và thiếu hẳn compsynth_test (MINH HỌA)
    for s in range(3):
        put(tmp_path, pd.concat([fake_scores("raw", s), fake_scores("raw", s, dataset="compsynth_test", cells=True)]), "raw", s)
    for s in range(2):
        put(tmp_path, fake_scores("raw+comp", s), "raw+comp", s)
    main = (build_report(tmp_path) / "main.md").read_text(encoding="utf-8")
    head = main.split("## CẢNH BÁO", 1)[1].split("| arm", 1)[0]
    for line in ("- raw+comp, seed 2, ffpp_test", "- raw+comp, seed 0, compsynth_test",
                 "- raw+comp, seed 1, compsynth_test", "- raw+comp, seed 2, compsynth_test"):
        assert line in head
    # raw+sb vắng hoàn toàn (I9): 3 seed x 2 dataset = 6 dòng thêm, cộng dòng nêu tên arm vắng
    for s_ in range(3):
        for d in ("ffpp_test", "compsynth_test"):
            assert f"- raw+sb, seed {s_}, {d}" in head
    assert "VẮNG HOÀN TOÀN" in head and "raw+sb" in head.split("VẮNG HOÀN TOÀN", 1)[1].splitlines()[0]
    assert "- raw," not in head and sum(ln.startswith("- ") for ln in head.splitlines()) == 4 + 6
    assert "⚠ thiếu seed [2]" in main and "– ⚠ thiếu" in main


def test_ci_from_seed0_or_flagged_missing(tmp_path):
    for s in (1, 2):  # thiếu seed 0
        put(tmp_path, fake_scores("raw", s), "raw", s)
    main = (build_report(tmp_path) / "main.md").read_text(encoding="utf-8")
    assert "CI: thiếu seed 0" in main
    for s in range(3):
        put(tmp_path, fake_scores("raw", s), "raw", s)
    out = build_report(tmp_path)
    sm = pd.read_parquet(out / "summary.parquet")
    r0 = sm[sm.seed == 0].iloc[0]
    assert f"[{r0.ci_lo:.4f}, {r0.ci_hi:.4f}]" in (out / "main.md").read_text(encoding="utf-8")


def test_two_runs_same_arm_seed_rejected(tree):
    d = tree / "aug" / "raw_other" / "seed0"  # arm run thứ hai cùng (dataset, seed, mode), split khác
    d.mkdir(parents=True)
    fake_scores("raw", 0, split="X").to_parquet(d / "scores_ffpp_test.parquet", index=False)
    with pytest.raises(ValueError, match="cần đúng một"):
        build_report(tree)


def test_report_warns_run_without_prereg_hash(tmp_path):
    import json

    for m in ARMS:
        put(tmp_path, fake_scores(m, 0), m, 0)
    for m, ph in (("raw", None), ("raw+comp", "abc123")):
        (tmp_path / "aug" / m / "seed0" / "run_info.json").write_text(json.dumps({"prereg_hash": ph}))
    main = (build_report(tmp_path) / "main.md").read_text(encoding="utf-8")
    head = main.partition("## Chú thích")[0]
    assert "## CẢNH BÁO" in head and "- aug/raw/seed0" in head
    assert "aug/raw+comp/seed0" not in head


def test_absent_arm_listed(tmp_path):
    """I9: arm kỳ vọng không có file scores nào vẫn được liệt kê dưới CẢNH BÁO và có hàng '– ⚠ thiếu'."""
    for m in ("raw", "raw+comp"):
        for s in range(3):
            put(tmp_path, fake_scores(m, s), m, s)
    main = (build_report(tmp_path) / "main.md").read_text(encoding="utf-8")
    head = main.split("## CẢNH BÁO", 1)[1].split("| arm", 1)[0]
    assert "VẮNG HOÀN TOÀN" in head and all(f"- raw+sb, seed {s}, ffpp_test" in head for s in range(3))
    assert "- raw, " not in head and "- raw+comp, " not in head
    row = [ln for ln in main.splitlines() if ln.startswith("| raw+sb |")]
    assert row and "– ⚠ thiếu" in row[0]


def tid_scores(mode, seed, dataset, ntid=12):
    """MINH HỌA: mỗi tid một video real + hai fake; video_id theo quy ước evaluate (ffpp: "{tid}_{src}",
    compsynth: "{tid}|real" / "{tid}|{region}|{method}")."""
    rng = np.random.default_rng(seed * 7 + len(mode))
    rows = []
    for t in range(ntid):
        tid = f"{t:03d}"
        if dataset == "ffpp_test":
            vids = [(f"{tid}_real", 0, ""), (f"{tid}_DF", 1, ""), (f"{tid}_F2F", 1, "")]
        else:
            vids = [(f"{tid}|real", 0, "real"), (f"{tid}|hair|DF", 1, "hair|DF"), (f"{tid}|skin|F2F", 1, "skin|F2F")]
        for vid, lab, cell in vids:
            for f in range(2):
                rows.append((f"{mode}/seed{seed}", mode, seed, "", dataset, vid, f, lab,
                             rng.normal(lab * STRENGTH[mode], 1.0), cell, "", BEST_STEP))
    return pd.DataFrame(rows, columns=COLS)


def test_frame_auc_column_and_tid_clustered_ci(tmp_path):
    """I8: main.md có cột AUC frame cạnh AUC video; CI ffpp_test/compsynth_test resample theo cụm tid."""
    from ffd.report import N_BOOT, _cluster
    from ffd.stats import auc, bootstrap_ci, video_scores

    assert _cluster("ffpp_test", "012_F2F") == "012" and _cluster("compsynth_test", "012|hair|DF") == "012"
    assert _cluster("compsynth_test", "012|a0ff|hair|DF") == "012" and _cluster("cdf2", "id0_0001") == "id0_0001"
    for m in ARMS:
        for s in range(3):
            put(tmp_path, pd.concat([tid_scores(m, s, "ffpp_test"), tid_scores(m, s, "compsynth_test")]), m, s)
    out = build_report(tmp_path)
    sm = pd.read_parquet(out / "summary.parquet")
    assert {"auc_frame", "n_clusters"} <= set(sm.columns) and (sm.n_clusters == 12).all() and (sm.n_videos == 36).all()
    for ds in ("ffpp_test", "compsynth_test"):
        df = tid_scores("raw", 0, ds)
        v = video_scores(df)
        groups = [_cluster(ds, x) for x in v.video_id]
        exp = bootstrap_ci(v.label, v.score, groups, n=N_BOOT, seed=0)
        r = sm[(sm.dataset == ds) & (sm["mode"] == "raw") & (sm.seed == 0)].iloc[0]
        assert (r.auc, r.ci_lo, r.ci_hi) == pytest.approx(exp)
        assert r.auc_frame == pytest.approx(auc(df.label, df.logit))
        unclustered = bootstrap_ci(v.label, v.score, v.video_id, n=N_BOOT, seed=0)
        assert (r.ci_lo, r.ci_hi) != pytest.approx(unclustered[1:])
    main = (out / "main.md").read_text(encoding="utf-8")
    assert "ffpp_test (video) | ffpp_test (frame)" in main and "cụm" in main.lower()
    raw = [ln for ln in main.splitlines() if ln.startswith("| raw |")][0]
    g = sm[(sm["mode"] == "raw") & (sm.dataset == "compsynth_test")]
    assert f"{g.auc_frame.mean():.4f} ± {g.auc_frame.std(ddof=1):.4f}" in raw


def test_arm_config_diff_warned(tree):
    """I4: run_info["config"] của các arm chỉ được khác ở mode/seed/runs_root/num_workers; khác lr -> CẢNH BÁO."""
    import json

    base = {"mode": "raw", "seed": 0, "runs_root": "a", "num_workers": 4, "lr": 2e-4, "rho": 0.5}
    for m in ARMS:
        for s in range(3):
            c = dict(base, mode=m, seed=s, runs_root=f"r{s}", num_workers=s)
            (tree / "aug" / m / f"seed{s}" / "run_info.json").write_text(
                json.dumps({"prereg_hash": "x", "config": c}), encoding="utf-8")
    assert "CẢNH BÁO" not in (build_report(tree) / "main.md").read_text(encoding="utf-8")
    p = tree / "aug" / "raw+sb" / "seed1" / "run_info.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    j["config"]["lr"] = 1e-3
    p.write_text(json.dumps(j), encoding="utf-8")
    main = (build_report(tree) / "main.md").read_text(encoding="utf-8")
    head = main.split("## CẢNH BÁO", 1)[1].split("| arm", 1)[0]
    line = [ln for ln in head.splitlines() if ln.startswith("- lr:")]
    assert line and "aug/raw+sb/seed1=0.001" in line[0] and "- rho:" not in head and "- seed:" not in head


def test_scores_best_step_mismatch_warned(tree):
    """I6: best.pt đổi (vd train lại) mà scores cũ còn -> CẢNH BÁO; thiếu best.pt cũng cảnh báo."""
    torch.save({"step": 99, "trainable": {}}, tree / "aug" / "raw" / "seed1" / "best.pt")
    (tree / "aug" / "raw+sb" / "seed2" / "best.pt").unlink()
    main = (build_report(tree) / "main.md").read_text(encoding="utf-8")
    head = main.split("## CẢNH BÁO", 1)[1].split("| arm", 1)[0]
    assert "- aug/raw/seed1/scores_ffpp_test.parquet: best_step trong scores = [10], best.pt = 99" in head
    assert "- aug/raw+sb/seed2/scores_ffpp_test.parquet: best_step trong scores = [10], best.pt = thiếu" in head
    assert "aug/raw/seed0" not in head


def test_delong_csv_is_clustered_by_tid(tmp_path):
    """R2: delong.csv dùng DeLong gom cụm theo tid (ffpp_test "{tid}_{src}"); khớp stats.delong_paired_clustered."""
    from ffd.stats import delong_paired_clustered, video_scores

    srcs = ("real", "DF", "F2F", "FS")
    for m in ARMS:
        for s in range(3):
            df = fake_scores(m, s, nvid=24)
            df["video_id"] = [f"t{int(v[1:]) // 4:02d}_{srcs[int(v[1:]) % 4]}" for v in df.video_id]
            df["label"] = [0 if v.endswith("_real") else 1 for v in df.video_id]
            put(tmp_path, df, m, s)
    d = pd.read_csv(build_report(tmp_path) / "delong.csv")
    assert (d.n_clusters == 12).all() and (d.n_videos == 48).all()
    r = d[(d.seed == 0) & (d.arm_a == "raw") & (d.arm_b == "raw+comp")].iloc[0]
    va = video_scores(pd.read_parquet(tmp_path / "aug/raw/seed0/scores_ffpp_test.parquet"))
    vb = video_scores(pd.read_parquet(tmp_path / "aug/raw+comp/seed0/scores_ffpp_test.parquet"))
    dd, z, p = delong_paired_clustered(va.label, va.score, vb.score, va.video_id.str.rsplit("_", n=1).str[0])
    assert r["diff"] == pytest.approx(dd) and r.z == pytest.approx(z) and r.p == pytest.approx(p)
