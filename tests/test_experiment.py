"""Test chạy chính (ffd.experiment): quy tắc ensure_run với train/eval giả; run_main 3 arm trên data_root tổng hợp
(định dạng preprocessing hiện tại). Mọi logit/AUC trong test là MINH HỌA. Trạng thái: chưa chạy thử trên GPU."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from ffd import experiment as X  # noqa: E402
from ffd.config import ARMS, TrainConfig  # noqa: E402
from ffd.data.ffpp import frames  # noqa: E402
from ffd.rundir import write_run_info  # noqa: E402


def _cfg(tmp_path, **kw):
    return TrainConfig(data_root=tmp_path / "data", ckpt_root=tmp_path / "ck", **kw)


def _block(a):
    """Khối chỉ số dataset MINH HỌA."""
    return {"auc": a, "acc": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75, "n_frames": 10, "auc_video": a}


def fake_runs(monkeypatch, aucs):
    """train giả ghi done.flag; eval giả ghi result.json với val.auc lấy theo tên run (`aucs`). Trả nhật ký gọi."""
    calls = []

    def fake_train(cfg, *, resume=False, force=False, **kw):
        assert not (resume and force)
        d = cfg.ckpt_root / (cfg.run_name or cfg.arm)
        d.mkdir(parents=True, exist_ok=True)
        (d / "done.flag").write_text("{}")
        calls.append(("train", cfg.run_name or cfg.arm, resume, force, cfg.lr, cfg.p_jpeg, cfg.arm))
        return d

    def fake_eval(run, *, datasets=("val", "ffpp_test", "celebdf"), **kw):
        calls.append(("eval", run.name, tuple(datasets)))
        p = run / "result.json"
        p.write_text(json.dumps({ds: _block(aucs.get(run.name, 0.5)) for ds in datasets}))
        return p

    monkeypatch.setattr(X, "train", fake_train)
    monkeypatch.setattr(X, "evaluate_run", fake_eval)
    return calls


def _celeb(tmp_path):
    m = tmp_path / "data" / "celebdf" / "manifest.parquet"
    m.parent.mkdir(parents=True)
    m.write_bytes(b"x")


def test_ensure_run_resume_rules(monkeypatch, tmp_path):
    """result.json -> bỏ qua; done.flag -> chỉ eval; last.pt -> train(resume=True); không gì -> train mới."""
    calls = fake_runs(monkeypatch, {})
    cfg = _cfg(tmp_path)
    d = tmp_path / "ck" / "raw"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"val": {"auc": 0.7}}))
    assert X.ensure_run(cfg, ("val",))["val"]["auc"] == 0.7 and calls == []
    (d / "result.json").unlink()
    (d / "done.flag").write_text("{}")
    X.ensure_run(cfg, ("val",))
    assert [c[0] for c in calls] == ["eval"]
    calls.clear()
    (d / "result.json").unlink()
    (d / "done.flag").unlink()
    (d / "last.pt").write_text("x")
    X.ensure_run(cfg, ("val",))
    assert calls[0][:4] == ("train", "raw", True, False)
    calls.clear()
    for p in d.iterdir():
        p.unlink()
    X.ensure_run(cfg, ("val",))
    assert calls[0][:4] == ("train", "raw", False, False)


def test_ensure_run_rejects_other_config(monkeypatch, tmp_path):
    fake_runs(monkeypatch, {})
    d = tmp_path / "ck" / "raw"
    d.mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps({"config_hash": "khac"}))
    (d / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="config_hash"):
        X.ensure_run(_cfg(tmp_path), ("val",))


def test_no_pilot_api():
    for name in ("run_pilot", "fmt_lr", "base_config", "pilot_name"):
        assert not hasattr(X, name)


def test_ensure_run_reevaluates_skipped_celebdf(monkeypatch, tmp_path):
    """result.json có celebdf skipped: chưa có dữ liệu -> bỏ qua; có dữ liệu -> chỉ eval lại (không train)."""
    calls = fake_runs(monkeypatch, {})
    cfg = _cfg(tmp_path)
    d = tmp_path / "ck" / "raw"
    d.mkdir(parents=True)
    old = {"val": {"auc": 0.7}, "ffpp_test": {"auc": 0.6}, "celebdf": {"skipped": "không có"}}
    (d / "result.json").write_text(json.dumps(old))
    assert X.ensure_run(cfg, X.MAIN_DATASETS) == old and calls == []
    _celeb(tmp_path)
    X.ensure_run(cfg, X.MAIN_DATASETS)
    assert calls == [("eval", "raw", ("val", "ffpp_test", "celebdf"))]


def test_ensure_run_reevaluates_missing_dataset(monkeypatch, tmp_path):
    calls = fake_runs(monkeypatch, {})
    d = tmp_path / "ck" / "raw"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"val": {"auc": 0.7}}))
    X.ensure_run(_cfg(tmp_path), ("val", "ffpp_test"))
    assert calls == [("eval", "raw", ("val", "ffpp_test"))]


SEP = {"raw": 1.0, "raw_aug": 2.0, "raw_rep": 1.5}  # độ tách lớp giả theo arm (MINH HỌA)


def fake_full(monkeypatch, data_root):
    calls = []

    def fake_train(cfg, *, resume=False, force=False, **kw):
        d = cfg.ckpt_root / cfg.run_name
        d.mkdir(parents=True, exist_ok=True)
        stats = {"n_seam_dropped": 3} if cfg.arm == "raw_rep" else {}
        write_run_info(d, cfg, {"model_commit": "x", "aug_stats": stats})
        (d / "done.flag").write_text("{}")
        calls.append(("train", cfg.arm))
        return d

    def fake_eval(run, *, datasets=("val", "ffpp_test", "celebdf"), **kw):
        rng = np.random.default_rng(len(run.name))
        for ds in datasets:
            f = frames(data_root, "celebdf", None) if ds == "celebdf" else \
                frames(data_root, "ffpp", "val" if ds == "val" else "test")
            logit = f.label.to_numpy(float) * SEP[run.name] + rng.normal(0, 1, len(f))
            pd.DataFrame({"video_id": f.video_id.astype(str), "frame_idx": f.frame_idx, "label": f.label,
                          "method": f.method, "logit": logit}).to_parquet(run / f"scores_{ds}.parquet", index=False)
        p = run / "result.json"
        p.write_text(json.dumps({ds: {"auc": 0.5} for ds in datasets}))
        calls.append(("eval", run.name))
        return p

    monkeypatch.setattr(X, "train", fake_train)
    monkeypatch.setattr(X, "evaluate_run", fake_eval)
    return calls


def _cfg_data(data_root, tmp_path, **kw):
    return TrainConfig(data_root=data_root, ckpt_root=tmp_path / "ck", **kw)


def test_main_runs_three_arms_and_decides(data_root, tmp_path, monkeypatch):
    calls = fake_full(monkeypatch, data_root)
    p = X.run_main(_cfg_data(data_root, tmp_path), B=50)
    assert [c for c in calls if c[0] == "train"] == [("train", a) for a in X.ARMS]
    out = json.loads(p.read_text(encoding="utf-8"))
    ck = tmp_path / "ck"
    for f in ("summary.md", "videoboot.json", "realfp_raw__raw_aug.json", "realfp_raw_aug__raw_rep.json"):
        assert (ck / f).is_file()
    d = out["decision"]
    assert d["R1_vs_R0"]["verdict"] in ("aug_helps", "aug_hurts", "null")
    assert d["R2_vs_R1"]["verdict"] in ("seam_helps_real", "seam_hurts_real", "inconclusive")
    assert isinstance(d["run_R3"], bool)
    assert set(out["mechanism"]) == set(X.ARMS)
    assert all(np.isfinite(v["rho"]) and v["n_frames"] > 0 for v in out["mechanism"].values())
    assert out["aug_stats"]["raw_rep"] == {"n_seam_dropped": 3}
    assert "R1 − R0" in (ck / "summary.md").read_text(encoding="utf-8")
    calls.clear()
    X.run_main(_cfg_data(data_root, tmp_path), B=50)  # chạy lại: mọi run đã có result.json -> không train/eval
    assert calls == []


def test_main_rejects_old_data_format(data_root, tmp_path):
    root = shutil.copytree(data_root, tmp_path / "r")
    p = root / "ffpp" / "manifest_raw.parquet"
    pd.read_parquet(p).drop(columns=["w", "h", "jpeg_q"]).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="không phải dữ liệu của preprocessing hiện tại"):
        X.run_main(_cfg_data(root, tmp_path), B=10)


def test_main_requires_celebdf(data_root, tmp_path):
    root = shutil.copytree(data_root, tmp_path / "r")
    (root / "celebdf" / "manifest.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Celeb-DF"):
        X.run_main(_cfg_data(root, tmp_path), B=10)


def test_main_refuses_run_from_other_data(data_root, tmp_path, monkeypatch):
    calls = fake_full(monkeypatch, data_root)
    cfg = _cfg_data(data_root, tmp_path)
    d = tmp_path / "ck" / "raw_rep"
    d.mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps({"data_hashes": {"manifest_sha256": {"raw": "v1"},
                                                                 "params_json_sha256": "v1"}}))
    with pytest.raises(ValueError, match="dữ liệu khác"):
        X.run_main(cfg, B=10)
    assert calls == []  # kiểm trước khi train run nào


def test_ci_verdict_and_decide():
    assert X.ci_verdict([0.01, 0.05], "p", "n") == "p"
    assert X.ci_verdict([-0.05, -0.01], "p", "n") == "n"
    assert X.ci_verdict([-0.01, 0.01], "p", "n") == "null"

    def vb(c_ci, f_diff):
        return {"celebdf": {"pairs": {"raw__raw_aug": {"auc_diff": 0.03, "auc_ci95": c_ci}}},
                "ffpp_test": {"pairs": {"raw__raw_aug": {"auc_diff": f_diff}}}}

    fp = {"primary": {"verdict": "inconclusive", "diff": 0.0, "ci95": [-0.01, 0.01]}}
    assert X.decide(vb([0.01, 0.05], 0.0), fp)["run_R3"] is True
    d = X.decide(vb([0.01, 0.05], -0.006), fp)  # FF++ tụt quá ngưỡng -> không chạy R3
    assert d["R1_vs_R0"]["ffpp_test_ok"] is False and d["run_R3"] is False
    assert X.decide(vb([-0.01, 0.05], 0.0), fp)["R1_vs_R0"]["verdict"] == "null"


def test_cli_main_parses():
    from ffd.cli import build_parser

    a = build_parser().parse_args(["main", "--config", "c.yaml", "--B", "100", "--set", "num_workers=2"])
    assert a.cmd == "main" and a.B == 100 and a.set == ["num_workers=2"]


def test_main_uses_config_and_prints_settings_first(data_root, tmp_path, monkeypatch, capsys):
    """main train 3 arm với đúng lr/p_jpeg của config; khối config hiệu lực in trước lần train đầu."""
    calls = fake_full(monkeypatch, data_root)
    seen = []
    orig = X.train

    def spy(cfg, **kw):
        seen.append((capsys.readouterr().out, cfg.lr, cfg.p_jpeg, cfg.arm, cfg.run_name))
        return orig(cfg, **kw)

    monkeypatch.setattr(X, "train", spy)
    X.run_main(_cfg_data(data_root, tmp_path, lr=3e-4, p_jpeg=0.2, arm="raw_aug"), B=20)
    assert [s[1:] for s in seen] == [(3e-4, 0.2, a, a) for a in ARMS]
    assert "Config hiệu lực" in seen[0][0] and "0.0003" in seen[0][0]
    for k in X.EFFECTIVE_KEYS:
        assert k in seen[0][0], k
    assert all(c[1] in ARMS for c in calls if c[0] == "eval")
