"""Test vòng train theo epoch. Trạng thái: chưa chạy thử trên GPU.
Dữ liệu: data_root tổng hợp (MINH HỌA), CLIP tí hon ngẫu nhiên, CPU."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ffd import train as T  # noqa: E402
from ffd.config import TrainConfig  # noqa: E402
from ffd.train import epoch_batches, is_better, train  # noqa: E402

COLS = ["epoch", "step", "train_loss", "lr", "val_auc_frame", "val_bce",
        "frac_real", "frac_rep", "frac_seam", "frac_jpeg", "seconds"]


@pytest.fixture
def cfg_tiny(tiny, data_root, tmp_path):
    kw = {k: v for k, v in tiny.items() if k != "vision_config"}
    return TrainConfig(data_root=data_root, ckpt_root=tmp_path / "ck", arm="raw", seed=0, batch_size=32,
                       max_epochs=3, patience=5, num_workers=0, **kw)


def run(cfg, tiny, **kw):
    return train(cfg, vision_config=tiny["vision_config"], allow_cpu=True, **kw)


def scripted(monkeypatch, aucs, bce=0.5):
    """Thay val bằng chuỗi AUC cho sẵn (theo thứ tự gọi); trả list đếm số lần gọi."""
    calls = []

    def fake(model, val_df, cfg, device):
        calls.append(1)
        return {"auc_frame": aucs[len(calls) - 1], "bce": bce}

    monkeypatch.setattr(T, "_validate", fake)
    return calls


def test_train_writes_outputs(cfg_tiny, tiny):
    d = run(cfg_tiny, tiny)
    assert d == cfg_tiny.ckpt_root / "raw"
    log = pd.read_csv(d / "train_log.csv")
    assert list(log.columns) == COLS
    done = json.loads((d / "done.flag").read_text())
    assert (d / "best.pt").exists() and done["epochs_run"] == len(log) == 3
    assert list(log.epoch) == [1, 2, 3] and list(log.step) == [3, 6, 9]  # 96 ảnh / 32 = 3 bước/epoch
    assert done["total_steps"] == 9 and done["stopped_early"] is False
    assert np.isfinite(log.train_loss).all() and (log.seconds > 0).all()
    assert ((log.val_auc_frame >= 0) & (log.val_auc_frame <= 1)).all()
    assert (log.frac_real == 16 / 96).all() and (log.frac_rep == 0).all()
    best = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    assert set(best) == {"trainable", "epoch", "step", "auc_frame", "bce"} and best["epoch"] == done["best_epoch"]
    last = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    assert {"trainable", "optimizer", "scheduler", "rng", "epoch", "step", "best", "best_state",
            "bad_epochs"} <= set(last) and last["epoch"] == 3
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    assert info["model_commit"] == "local" and info["config"]["arm"] == "raw"


def test_epoch_visits_each_item_once():
    b = epoch_batches(10, 4, seed=0, epoch=1)
    assert [len(x) for x in b] == [4, 4, 2]  # batch cuối nhỏ vẫn giữ
    flat = [i for x in b for i in x]
    assert sorted(flat) == list(range(10))
    g = torch.Generator().manual_seed(0 * 100_003 + 1)
    assert flat == torch.randperm(10, generator=g).tolist()
    assert flat != [i for x in epoch_batches(10, 4, seed=0, epoch=2) for i in x]


def test_tie_break_by_bce():
    assert is_better((0.99995, 0.20), (0.99990, 0.30)) and not is_better((0.99995, 0.40), (0.99990, 0.30))
    assert is_better((0.98, 0.9), None)
    assert is_better((0.99, 0.5), (0.98, 0.1)) and not is_better((0.97, 0.1), (0.98, 0.5))


def test_early_stopping(cfg_tiny, tiny, monkeypatch):
    calls = scripted(monkeypatch, [0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9])
    d = run(dataclasses.replace(cfg_tiny, max_epochs=10, batch_size=96), tiny)
    done = json.loads((d / "done.flag").read_text())
    assert len(calls) == 7 and done == {"epochs_run": 7, "best_epoch": 2, "stopped_early": True, "total_steps": 7}
    assert torch.load(d / "best.pt", map_location="cpu", weights_only=False)["epoch"] == 2


def test_resume_keeps_patience(cfg_tiny, tiny, monkeypatch, tmp_path):
    aucs = [0.6, 0.7, 0.7, 0.7, 0.7]
    c = dataclasses.replace(cfg_tiny, max_epochs=5, patience=2)
    scripted(monkeypatch, aucs)
    ref = run(dataclasses.replace(c, ckpt_root=tmp_path / "ref"), tiny)
    assert json.loads((ref / "done.flag").read_text())["epochs_run"] == 4

    class Stop(Exception):
        pass

    save = T._atomic_save

    def save_then_stop(obj, path):
        save(obj, path)
        if path.name == "last.pt" and obj["epoch"] == 3:
            raise Stop

    calls = scripted(monkeypatch, aucs)
    monkeypatch.setattr(T, "_atomic_save", save_then_stop)
    with pytest.raises(Stop):
        run(c, tiny)
    d = c.ckpt_root / "raw"
    assert not (d / "done.flag").exists() and len(pd.read_csv(d / "train_log.csv")) == 3
    monkeypatch.setattr(T, "_atomic_save", save)
    with pytest.raises(FileExistsError):
        run(c, tiny)  # còn kết quả dở: phải resume hoặc force
    run(c, tiny, resume=True)  # bad_epochs = 1 đọc từ last.pt -> dừng sau epoch 4
    assert len(calls) == 4
    a, b = pd.read_csv(ref / "train_log.csv"), pd.read_csv(d / "train_log.csv")
    pd.testing.assert_frame_equal(a.drop(columns="seconds"), b.drop(columns="seconds"))
    assert json.loads((ref / "done.flag").read_text()) == json.loads((d / "done.flag").read_text())
    ba = torch.load(ref / "best.pt", map_location="cpu", weights_only=False)
    bb = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    assert ba["epoch"] == bb["epoch"] == 2
    assert all(torch.equal(ba["trainable"][k], bb["trainable"][k]) for k in ba["trainable"])


def test_refuse_finished_run(cfg_tiny, tiny):
    d = cfg_tiny.ckpt_root / "raw"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run(cfg_tiny, tiny)


def test_raw_aug_fracs(cfg_tiny, tiny):
    c = dataclasses.replace(cfg_tiny, max_epochs=1, p_jpeg=1.0)
    raw = pd.read_csv(run(c, tiny) / "train_log.csv")
    assert (raw.frac_rep + raw.frac_seam == 0).all() and (raw.frac_jpeg == 1.0).all()
    aug = pd.read_csv(run(dataclasses.replace(c, arm="raw_aug"), tiny) / "train_log.csv")
    assert (aug.frac_rep + aug.frac_seam > 0).all()
    assert (aug.frac_rep == 16 / 128).all() and (aug.frac_seam == 16 / 128).all()
    assert (aug.frac_real == (16 + 16) / 128).all()


def test_nan_loss_writes_error(cfg_tiny, tiny, monkeypatch):
    from ffd.heads.binary import BinaryHead

    monkeypatch.setattr(BinaryHead, "loss", lambda self, pred, tg: (pred["logit"].sum() * float("nan"), {}))
    with pytest.raises(FloatingPointError):
        run(cfg_tiny, tiny)
    err = json.loads((cfg_tiny.ckpt_root / "raw" / "error.json").read_text(encoding="utf-8"))
    assert err["epoch"] == 1 and err["step"] == 1 and err["video_ids"]


def test_missing_cuda_raises(cfg_tiny, tiny, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train(cfg_tiny, vision_config=tiny["vision_config"])


def test_score_frames_order(cfg_tiny, tiny):
    from ffd.data.ffpp import frames
    from ffd.model import Detector

    m = Detector(cfg_tiny, vision_config=tiny["vision_config"]).eval()
    df = frames(cfg_tiny.data_root, "ffpp", "val").iloc[:5]
    a = T.score_frames(m, df, 2, "cpu", 0, "t")
    b = T.score_frames(m, df.iloc[::-1], 3, "cpu", 0, "t")
    assert a.shape == (5,) and np.allclose(a, b[::-1], atol=1e-5)


def test_stopped_early_false_when_patience_hits_at_max_epochs(cfg_tiny, tiny, monkeypatch):
    """Đủ patience đúng ở epoch cuối (= max_epochs) -> stopped_early False (dừng vì hết epoch)."""
    scripted(monkeypatch, [0.6, 0.6, 0.6])
    d = run(dataclasses.replace(cfg_tiny, max_epochs=3, patience=2, batch_size=96), tiny)
    done = json.loads((d / "done.flag").read_text())
    assert done["epochs_run"] == 3 and done["stopped_early"] is False


def test_train_with_spawn_worker(cfg_tiny, tiny):
    """Hồi quy spawn (Windows): DataLoader num_workers=1 phải pickle được dataset/worker_init_fn. 1 epoch, 1 bước."""
    d = run(dataclasses.replace(cfg_tiny, max_epochs=1, num_workers=1, batch_size=96), tiny)
    assert json.loads((d / "done.flag").read_text())["epochs_run"] == 1
