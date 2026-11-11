"""Test vòng train. Trạng thái: chưa chạy thử. Dữ liệu: FF++ tổng hợp (MINH HỌA), CLIP tí hon ngẫu nhiên."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

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
    for w in ("ffpp_test", "compsynth_test", "cross", "compsynth_train", "compsynth_val"):
        with pytest.raises(LeakageError):
            allow_read("train", w)
    for w in ("ffpp_test_raw", "ffpp_val_raw", "whatever"):  # chỉ các nguồn train/val đã biết
        with pytest.raises(LeakageError):
            allow_read("train", w)
    for w in ("ffpp_train", "ffpp_val"):
        allow_read("train", w)
    allow_read("eval", "ffpp_test")


def test_val_items_reject_test_subset(ffpp_synth, tmp_path):
    with pytest.raises(LeakageError):
        T._val_items(mkcfg(), ffpp_synth, "test")


def test_val_metrics_single_class_or_nan_raises():
    import pandas as pd

    one = pd.DataFrame({"video_id": ["a", "b"], "frame_idx": [0, 0], "label": [0, 0], "logit": [0.1, 0.2]})
    with pytest.raises(ValueError, match="một lớp"):
        T._val_metrics(one)
    bad = pd.DataFrame({"video_id": ["a", "b"], "frame_idx": [0, 0], "label": [0, 1], "logit": [0.1, float("nan")]})
    with pytest.raises(ValueError, match="hữu hạn"):
        T._val_metrics(bad)


def test_val_set_for():
    assert val_set_for(mkcfg(mode="raw+rep")) == "ffpp_val_raw"


def test_missing_cuda_raises(ffpp_synth, tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train(mkcfg(runs_root=tmp_path), vision_config=tiny["vision_config"], ff_cfg=ffpp_synth)


def test_train_writes_artifacts(ffpp_synth, tiny, tmp_path):
    d = run(mkcfg(mode="raw+rep", runs_root=tmp_path), ffpp_synth, tiny)
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


def test_restart_without_force_or_resume_refused(ffpp_synth, tiny, tmp_path, monkeypatch):
    cfg = mkcfg(runs_root=tmp_path, total_steps=4)
    orig = T.save_last

    def boom(path, payload):
        orig(path, payload)
        raise KeyboardInterrupt("crash")

    monkeypatch.setattr(T, "save_last", boom)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, ffpp_synth, tiny)
    monkeypatch.setattr(T, "save_last", orig)
    with pytest.raises(FileExistsError):  # last.pt, val_curve.jsonl còn lại, không resume/force
        run(cfg, ffpp_synth, tiny)


def test_orphan_best_after_resume(ffpp_synth, tiny, tmp_path, monkeypatch):
    """eval_every=2, ckpt_every=4: best ghi ở step 6 > last.step 4 khi crash; sau resume best.pt phải về step 4."""
    cfg = mkcfg(runs_root=tmp_path, total_steps=8, eval_every=2, ckpt_every=4)
    calls = [0]

    def fake_metrics(df):
        calls[0] += 1
        if calls[0] == 4:  # val ở step 8, sau khi step 6 đã ghi best.pt
            raise KeyboardInterrupt("crash")
        return {"auc_video": 0.5 + 0.01 * calls[0] if calls[0] < 4 else 0.5, "auc_frame": 0.5, "bce": 0.7}

    monkeypatch.setattr(T, "_val_metrics", fake_metrics)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, ffpp_synth, tiny)
    d = T.run_dir(cfg)
    assert torch.load(d / "best.pt", weights_only=False)["step"] == 6  # mồ côi
    run(cfg, ffpp_synth, tiny, resume=True)
    best = torch.load(d / "best.pt", weights_only=False)
    rows = {r["step"]: r for r in read_jsonl(d / "val_curve.jsonl")}
    assert best["step"] == 4 and best["step"] in rows
    assert best["auc_video"] == pytest.approx(rows[4]["auc_video"])
    assert (d / "done.flag").exists()


def test_grad_accum_across_epoch_resume(ffpp_synth, tiny, tmp_path, monkeypatch):
    """bs=6 -> 21 batch/epoch (lẻ), accum=2: step 11 vắt qua biên epoch; resume sau biên vẫn khớp run liên tục."""
    kw = dict(batch_size=6, grad_accum=2, total_steps=14, eval_every=7, ckpt_every=6)
    full = losses(run(mkcfg(runs_root=tmp_path / "a", **kw), ffpp_synth, tiny))
    orig, st = T.save_last, {}

    def crashing(path, payload):
        orig(path, payload)
        if payload["step"] == 12 and not st:
            st["x"] = 1
            raise KeyboardInterrupt("crash")

    monkeypatch.setattr(T, "save_last", crashing)
    cb = mkcfg(runs_root=tmp_path / "b", **kw)
    with pytest.raises(KeyboardInterrupt):
        run(cb, ffpp_synth, tiny)
    monkeypatch.setattr(T, "save_last", orig)
    run(cb, ffpp_synth, tiny, resume=True)
    res = losses(T.run_dir(cb))
    assert len(res) == 14 and res == pytest.approx(full, rel=1e-5)


def test_dataloader_spawn_worker_iterates(ffpp_synth):
    """C1: worker_init_fn pickle được với spawn (mặc định Windows); DataLoader num_workers=1 lấy được một batch."""
    import functools
    import multiprocessing as mp

    from ffd.data.arms import ArmTrainDataset

    ds = ArmTrainDataset(ffpp_synth, "raw+rep", 0.5, 0)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=T._epoch_batches(len(ds), 4, 0, 0)[:1], num_workers=1, persistent_workers=False,
        worker_init_fn=functools.partial(T._seed_worker, seed=0), multiprocessing_context=mp.get_context("spawn"))
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 4


def test_train_with_worker_cpu(ffpp_synth, tiny, tmp_path):
    """C1: train() với num_workers=1 chạy được 2 step trên CPU (spawn trên Windows)."""
    d = run(mkcfg(runs_root=tmp_path, total_steps=2, eval_every=2, ckpt_every=2, num_workers=1), ffpp_synth, tiny)
    assert (d / "done.flag").exists() and len(losses(d)) == 2


def test_epoch_batches_balanced(ffpp_synth):
    """I5: mỗi batch đúng một nửa L0==0 (real, chỉ số chẵn) và một nửa fake; batch_size lẻ -> ValueError."""
    from ffd.data.arms import ArmTrainDataset

    for n, bs in ((126, 6), (40, 4), (41, 8), (2, 2)):
        bb = T._epoch_batches(n, bs, 3, 1)
        assert bb == T._epoch_batches(n, bs, 3, 1) and bb != T._epoch_batches(n, bs, 3, 2) or n == 2
        assert len(bb) == min((n + 1) // 2, n // 2) // (bs // 2)
        for b in bb:
            assert len(b) == bs and sum(i % 2 == 0 for i in b) == bs // 2 and len(set(b)) == bs
        assert len({i for b in bb for i in b}) == len(bb) * bs  # không lặp trong epoch
    with pytest.raises(ValueError, match="chẵn"):
        T._epoch_batches(40, 5, 0, 0)
    ds = ArmTrainDataset(ffpp_synth, "raw+rep", 0.5, 0)
    for b in T._epoch_batches(len(ds), 4, 0, 0)[:3]:
        assert sum(ds[i]["targets"]["L0"] == 0 for i in b) == 2


def test_odd_batch_size_rejected_by_config(tmp_path):
    from ffd.config import load_train_config

    y = tmp_path / "c.yaml"
    y.write_text("ffpp_config: x.yaml\nruns_root: r\nmode: raw\nseed: 0\ntotal_steps: 1\neval_every: 1\n"
                 "ckpt_every: 1\nbatch_size: 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chẵn"):
        load_train_config(y)


def test_train_progress_bar_does_not_change_results(ffpp_synth, tiny, tmp_path, capsys, monkeypatch):
    """Thanh tiến trình ghi stderr (step/total, loss); tắt hay bật thì loss, val và best.pt y hệt nhau."""
    from ffd import progress

    a = run(mkcfg(runs_root=tmp_path / "on"), ffpp_synth, tiny)
    err = capsys.readouterr().err
    assert "train raw seed0" in err and "12/12" in err and "loss=" in err
    monkeypatch.setattr(progress, "ENABLED", False)
    b = run(mkcfg(runs_root=tmp_path / "off"), ffpp_synth, tiny)
    assert capsys.readouterr().err == ""
    assert read_jsonl(a / "train_log.jsonl") == read_jsonl(b / "train_log.jsonl")
    assert read_jsonl(a / "val_curve.jsonl") == read_jsonl(b / "val_curve.jsonl")
    ba, bb = torch.load(a / "best.pt", weights_only=False), torch.load(b / "best.pt", weights_only=False)
    assert ba["step"] == bb["step"] and all(torch.equal(ba["trainable"][k], bb["trainable"][k]) for k in ba["trainable"])
