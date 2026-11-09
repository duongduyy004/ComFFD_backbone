"""Test vòng train. Trạng thái: chưa chạy thử. Dữ liệu: FF++ tổng hợp (MINH HỌA), CLIP tí hon ngẫu nhiên."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from compsynth_ff.dataset import TrainCore  # noqa: E402
from ffd import train as T  # noqa: E402
from ffd.config import TrainConfig  # noqa: E402
from ffd.guards import LeakageError, allow_read, val_set_for  # noqa: E402
from ffd.train import is_better, train, val_scores  # noqa: E402


def mkcfg(**kw):
    d = dict(ffpp_config=Path("unused.yaml"), runs_root=Path("runs"), mode="raw", seed=0, total_steps=12,
             eval_every=4, ckpt_every=4, batch_size=4, num_workers=0, grad_checkpointing=False,
             lora_r=4, lora_alpha=8)
    d.update(kw)
    return TrainConfig(**d)


def run(cfg, ffpp, tiny, **kw):
    return train(cfg, vision_config=tiny["vision_config"], ff_cfg=ffpp, allow_cpu=True, log_every=1, **kw)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def losses(d):
    return [r["loss"] for r in read_jsonl(Path(d) / "train_log.jsonl")]


def test_tie_break_by_bce():
    assert is_better((0.99995, 0.20), (0.99990, 0.30)) and not is_better((0.99995, 0.40), (0.99990, 0.30))
    assert is_better((0.98, 0.9), None)
    assert is_better((0.99, 0.5), (0.98, 0.1)) and not is_better((0.97, 0.1), (0.98, 0.5))


def test_train_cannot_read_test():
    for w in ("ffpp_test", "compsynth_test", "cross"):
        with pytest.raises(LeakageError):
            allow_read("train", w)
    allow_read("train", "ffpp_val_raw")
    allow_read("eval", "ffpp_test")


def test_val_set_for():
    assert val_set_for(mkcfg(mode="raw+comp")) == "ffpp_val_raw"
    assert val_set_for(mkcfg(mode="compffd", split="s.json")) == "compsynth_val_seen"


def test_missing_cuda_raises(ffpp_synth, tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train(mkcfg(runs_root=tmp_path), vision_config=tiny["vision_config"], ff_cfg=ffpp_synth)


def test_train_writes_artifacts(ffpp_synth, tiny, tmp_path):
    d = run(mkcfg(mode="raw+comp", runs_root=tmp_path), ffpp_synth, tiny)
    assert {"best.pt", "last.pt", "done.flag", "run_info.json", "val_curve.jsonl"} <= set(os.listdir(d))
    vc = read_jsonl(d / "val_curve.jsonl")
    assert [r["step"] for r in vc] == [4, 8, 12]
    assert {"auc_video", "auc_frame", "bce"} <= set(vc[0])
    assert [r["step"] for r in read_jsonl(d / "train_log.jsonl")] == list(range(1, 13))
    assert not [p for p in os.listdir(d) if ".tmp" in p]
    last = torch.load(d / "last.pt", weights_only=False)
    assert last["step"] == 12 and {"trainable", "optimizer", "scheduler", "rng", "best"} <= set(last)
    assert {"torch", "numpy", "python"} <= set(last["rng"])


def test_val_scores_raw(ffpp_synth, tiny):
    from ffd.model import Detector

    m = Detector(mkcfg(), vision_config=tiny["vision_config"])
    df = val_scores(m, mkcfg(), ffpp_synth, batch_size=4, device="cpu")
    assert list(df.columns) == ["video_id", "frame_idx", "label", "logit"]
    assert set(df.label) == {0, 1} and len(df) > 0 and df.logit.notna().all()


def test_val_scores_compffd_filters_to_seen(ffpp_synth, tiny, tmp_path):
    from compsynth_ff.dataset import EvalCore
    from ffd.model import Detector

    core = TrainCore(ffpp_synth)
    cells = sorted(core.allowed)
    seen, unseen = cells[:1], cells[1:]
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps({"seen": [list(c) for c in seen], "unseen": [list(c) for c in unseen]}))
    cfg = mkcfg(mode="compffd", split=str(sp))
    m = Detector(cfg, vision_config=tiny["vision_config"])
    df = val_scores(m, cfg, ffpp_synth, batch_size=4, device="cpu")
    ec = EvalCore(ffpp_synth, "val")
    metas = [ec.meta(i) for i in range(len(ec))]
    exp = [x for x in metas if x["kind"] == "real" or (x["region"], x["method"]) in {tuple(c) for c in seen}]
    assert len(df) == len(exp) and len(df) < len(metas) or not unseen
    assert all(v.endswith("|real") or v.count("|") == 2 for v in df.video_id)
    assert (df.label == 0).sum() == sum(x["kind"] == "real" for x in exp)


def test_resume_matches_continuous(ffpp_synth, tiny, tmp_path, monkeypatch):
    full = losses(run(mkcfg(runs_root=tmp_path / "a"), ffpp_synth, tiny))
    orig, state = T.save_last, {}

    def crashing(path, payload):
        orig(path, payload)
        if payload["step"] == 8 and not state.get("done"):
            state["done"] = True
            raise KeyboardInterrupt("crash giả lập")

    monkeypatch.setattr(T, "save_last", crashing)
    cfgb = mkcfg(runs_root=tmp_path / "b")
    with pytest.raises(KeyboardInterrupt):
        run(cfgb, ffpp_synth, tiny)
    d = Path(T.run_dir(cfgb))
    assert not (d / "done.flag").exists()
    with open(d / "val_curve.jsonl", "a", encoding="utf-8") as f:  # dòng mồ côi sau last.step
        f.write(json.dumps({"step": 12, "auc_video": 0.0, "auc_frame": 0.0, "bce": 9.0}) + "\n")
    with open(d / "train_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 9, "loss": 123.0}) + "\n")
    monkeypatch.setattr(T, "save_last", orig)
    run(cfgb, ffpp_synth, tiny, resume=True)
    resumed = losses(d)
    assert len(resumed) == 12 and resumed[8:] == pytest.approx(full[8:], rel=1e-5)
    assert resumed[:8] == pytest.approx(full[:8], rel=1e-5)
    assert [r["step"] for r in read_jsonl(d / "val_curve.jsonl")] == [4, 8, 12]
    assert (d / "done.flag").exists()


def test_nan_stops(ffpp_synth, tiny, tmp_path, monkeypatch):
    from ffd.heads.binary import BinaryHead

    orig, n = BinaryHead.loss, [0]

    def bad(self, pred, targets):
        n[0] += 1
        l, d = orig(self, pred, targets)
        return (l * float("nan"), d) if n[0] == 3 else (l, d)

    monkeypatch.setattr(BinaryHead, "loss", bad)
    cfg = mkcfg(runs_root=tmp_path)
    with pytest.raises(FloatingPointError, match="step 3"):
        run(cfg, ffpp_synth, tiny)
    e = json.loads((T.run_dir(cfg) / "error.json").read_text(encoding="utf-8"))
    assert e["step"] == 3 and e["video_ids"] and all(isinstance(v, str) for v in e["video_ids"])
    assert not (T.run_dir(cfg) / "done.flag").exists()


def test_force_rerun_clears_stale(ffpp_synth, tiny, tmp_path):
    cfg = mkcfg(runs_root=tmp_path, total_steps=4)
    d = run(cfg, ffpp_synth, tiny)
    assert (d / "done.flag").exists()
    with pytest.raises(FileExistsError):
        run(cfg, ffpp_synth, tiny)
    d = run(cfg, ffpp_synth, tiny, force=True)
    assert [r["step"] for r in read_jsonl(d / "val_curve.jsonl")] == [4]
