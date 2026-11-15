"""Test pilot/chạy chính (ffd.experiment) với train/eval giả, không cần model. Trạng thái: chưa chạy thử trên GPU.
Số AUC trong test là MINH HỌA."""
import json

import pytest

pytest.importorskip("torch")

from ffd import experiment as X  # noqa: E402
from ffd.config import TrainConfig  # noqa: E402
from ffd.experiment import fmt_lr, run_main, run_pilot  # noqa: E402


def test_fmt_lr():
    assert [fmt_lr(x) for x in (1e-4, 2e-4, 5e-4)] == ["1e-4", "2e-4", "5e-4"]
    assert fmt_lr(1.5e-4) == "1.5e-4"


def _cfg(tmp_path, **kw):
    return TrainConfig(data_root=tmp_path / "data", ckpt_root=tmp_path / "ck", **kw)


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
        name = run.relative_to(run.parents[1]).as_posix() if run.parent.name == "pilot" else run.name
        calls.append(("eval", name, tuple(datasets)))
        p = run / "result.json"
        p.write_text(json.dumps({"val": {"auc": aucs.get(name, 0.5)}}))
        return p

    monkeypatch.setattr(X, "train", fake_train)
    monkeypatch.setattr(X, "evaluate_run", fake_eval)
    return calls


def _aucs(p0, p1, lr=(0.80, 0.85, 0.83)):
    a = {f"pilot/lr{fmt_lr(x)}_p0.5": v for x, v in zip((1e-4, 2e-4, 5e-4), lr)}
    a["pilot/lr2e-4_p0"], a["pilot/lr2e-4_p1"] = p0, p1
    return a


def test_pilot_selection_rule(monkeypatch, tmp_path):
    calls = fake_runs(monkeypatch, _aucs(0.847, 0.86))
    sel = json.loads(run_pilot(_cfg(tmp_path, arm="raw_aug")).read_text())
    assert (sel["lr"], sel["p_jpeg"]) == (2e-4, 1.0)
    assert set(sel["val_auc"]) == {"pilot/lr1e-4_p0.5", "pilot/lr2e-4_p0.5", "pilot/lr5e-4_p0.5",
                                   "pilot/lr2e-4_p0", "pilot/lr2e-4_p1"}
    trains = [c for c in calls if c[0] == "train"]
    assert len(trains) == 5 and all(c[6] == "raw" for c in trains)  # pilot chỉ arm raw
    assert all(c[2] == ("val",) for c in calls if c[0] == "eval")  # pilot chỉ chấm val


def test_pilot_tie_prefers_smaller_p(monkeypatch, tmp_path):
    fake_runs(monkeypatch, _aucs(0.848, 0.851))
    sel = json.loads(run_pilot(_cfg(tmp_path)).read_text())
    assert (sel["lr"], sel["p_jpeg"]) == (2e-4, 0.0)


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


def test_main_requires_selected(tmp_path):
    with pytest.raises(FileNotFoundError, match="pilot"):
        run_main(_cfg(tmp_path))


def test_main_uses_selected(monkeypatch, tmp_path):
    calls = fake_runs(monkeypatch, {})
    monkeypatch.setattr(X, "compare_runs", lambda root: root / "compare.json")
    sel = tmp_path / "ck" / "pilot" / "selected.json"
    sel.parent.mkdir(parents=True)
    sel.write_text(json.dumps({"lr": 5e-4, "p_jpeg": 1.0, "val_auc": {}}))
    assert run_main(_cfg(tmp_path)) == tmp_path / "ck" / "compare.json"
    trains = [c for c in calls if c[0] == "train"]
    assert [(c[1], c[4], c[5], c[6]) for c in trains] == [("raw", 5e-4, 1.0, "raw"), ("raw_aug", 5e-4, 1.0, "raw_aug")]
    assert all(c[2] == ("val", "ffpp_test", "celebdf") for c in calls if c[0] == "eval")
