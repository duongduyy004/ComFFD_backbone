"""Test chạy chính (ffd.experiment) với train/eval giả, không cần model. Trạng thái: chưa chạy thử trên GPU.
Số AUC/chỉ số trong test là MINH HỌA."""
import json

import pytest

pytest.importorskip("torch")

from ffd import experiment as X  # noqa: E402
from ffd.config import TrainConfig  # noqa: E402
from ffd.experiment import run_main  # noqa: E402


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


def _fake_compare(root):
    """compare.json MINH HỌA (ffpp_test có số, celebdf skipped)."""
    p = root / "compare.json"
    p.write_text(json.dumps({"ffpp_test": {"auc_raw": 0.8, "auc_raw_aug": 0.85, "diff": 0.05,
                                           "diff_ci95": [0.01, 0.09], "delong_p": 0.012},
                             "celebdf": {"skipped": "lý do MINH HỌA"}, "note": "Exploratory"}))
    return p


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


def test_main_uses_config_lr_pjpeg(monkeypatch, tmp_path, capsys):
    """main train raw, raw_aug với đúng lr/p_jpeg của config (không pilot), gọi compare, ghi summary.md."""
    calls = fake_runs(monkeypatch, {"raw": 0.8, "raw_aug": 0.85})
    monkeypatch.setattr(X, "compare_runs", _fake_compare)
    _celeb(tmp_path)
    assert run_main(_cfg(tmp_path, lr=3e-4, p_jpeg=0.2, arm="raw_aug")) == tmp_path / "ck" / "compare.json"
    trains = [c for c in calls if c[0] == "train"]
    assert [(c[1], c[4], c[5], c[6]) for c in trains] == [("raw", 3e-4, 0.2, "raw"), ("raw_aug", 3e-4, 0.2, "raw_aug")]
    assert all(c[2] == ("val", "ffpp_test", "celebdf") for c in calls if c[0] == "eval")
    assert not (tmp_path / "ck" / "pilot").exists()
    md = (tmp_path / "ck" / "summary.md").read_text(encoding="utf-8")
    assert "| raw_aug | celebdf | 0.9000 | 0.8500 | 0.8000 | 0.7000 | 0.7500 |" in md
    assert "| ffpp_test | 0.8000 | 0.8500 | +0.0500 | [+0.0100, +0.0900] | 0.012 |" in md
    assert "| celebdf | skipped: lý do MINH HỌA |" in md
    out = capsys.readouterr().out
    assert "AUC raw_aug" in out and "p DeLong" in out and "RECALL" in out


def test_main_requires_celebdf(monkeypatch, tmp_path):
    calls = fake_runs(monkeypatch, {})
    with pytest.raises(FileNotFoundError, match="celebdf"):
        run_main(_cfg(tmp_path))
    assert calls == []


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
