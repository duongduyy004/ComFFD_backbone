"""Test báo cáo (không cần torch). Trạng thái: chưa chạy thử.
Toàn bộ scores.parquet ở đây là TỔNG HỢP (MINH HỌA), không phải kết quả thật."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ffd.report import build_report

COLS = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag"]
ARMS = ("raw", "raw+comp", "raw+sb")
STRENGTH = {"raw": 0.8, "raw+comp": 1.6, "raw+sb": 1.2, "compffd": 2.0}


def fake_scores(mode, seed, dataset="ffpp_test", nvid=20, split="", drop=(), flagged=(), cells=False):
    """MINH HỌA: logit = nhãn * độ mạnh + nhiễu; mỗi video 3 frame."""
    rng = np.random.default_rng(1000 * seed + len(mode))
    rows = []
    for k in range(2 * nvid):
        lab = k % 2
        vid = f"v{k:03d}"
        if vid in drop:
            continue
        cell = ("hair|DF" if k % 4 == 1 else "skin|F2F") if (cells and lab) else "real"
        for f in range(3):
            lg = rng.normal(lab * STRENGTH[mode], 1.0)
            flag = ""
            if vid in flagged:
                lg, f, flag = 0.0, -1, "no_usable_frame"
            rows.append((f"{mode}/seed{seed}", mode, seed, split, dataset, vid, f, lab, lg, cell, flag))
            if flag:
                break
    return pd.DataFrame(rows, columns=COLS)


def put(root, df, mode, seed, split=""):
    d = root / ("compffd" if mode == "compffd" else "aug") / (split if mode == "compffd" else mode) / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
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


def test_compffd_never_paired(tree):
    for s in range(3):
        put(tree, fake_scores("compffd", s, split="FF-03"), "compffd", s, "FF-03")
    out = build_report(tree)
    d = pd.read_csv(out / "delong.csv")
    assert len(d) == 9 and "compffd" not in set(d.arm_a) | set(d.arm_b)
    assert "compffd" in (out / "main.md").read_text(encoding="utf-8")


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
    assert "hair|DF" in pc and "skin|F2F" in pc
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
    assert "- raw," not in head and sum(ln.startswith("- ") for ln in head.splitlines()) == 4
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
