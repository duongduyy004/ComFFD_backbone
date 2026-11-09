"""Test evaluate_run. Trạng thái: chưa chạy thử. Dữ liệu: FF++/CDF2 tổng hợp (MINH HỌA), CLIP tí hon ngẫu nhiên."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from synth import FakeDetector  # noqa: E402
from synth_cross import make_synth_cdf2  # noqa: E402

from conftest import make_cfg  # noqa: E402
from ffd.config import TrainConfig, file_hash  # noqa: E402
from ffd.data.cross import prep_cross  # noqa: E402
from ffd.evaluate import evaluate_run  # noqa: E402
from ffd.guards import LeakageError, allow_read  # noqa: E402
from ffd.train import train  # noqa: E402

COLS = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag"]


@pytest.fixture(scope="module")
def vc():
    from transformers import CLIPVisionConfig

    return CLIPVisionConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=24,
                            num_attention_heads=4, image_size=336, patch_size=14)


@pytest.fixture(scope="module")
def trained_tiny_run(ffpp_synth, vc, tmp_path_factory):
    cfg = TrainConfig(ffpp_config=Path("unused.yaml"), runs_root=tmp_path_factory.mktemp("runs"), mode="raw",
                      seed=0, total_steps=2, eval_every=2, ckpt_every=2, batch_size=4, num_workers=0,
                      grad_checkpointing=False, lora_r=4, lora_alpha=8)
    return train(cfg, vision_config=vc, ff_cfg=ffpp_synth, allow_cpu=True)


@pytest.fixture(scope="module")
def synth_cross(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cross")
    root = make_synth_cdf2(tmp / "Celeb_DFv2")
    man = prep_cross("cdf2", root, tmp / "out", make_cfg(tmp, tmp / "ffout"), FakeDetector())
    return man.parent  # <out>/cdf2: manifest.parquet + crops/


def ev(run, ds, vc, ffpp, cross=None):
    return pd.read_parquet(evaluate_run(run, ds, cross or {}, ff_cfg=ffpp, vision_config=vc, device="cpu"))


def test_eval_refuses_unfinished_run(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"")
    with pytest.raises(ValueError):
        evaluate_run(tmp_path, ["ffpp_test"], {})


def test_eval_rejects_unknown_dataset(trained_tiny_run):
    for bad in ("dfdc", "ffpp_val", "ffpp_train"):
        with pytest.raises(ValueError):
            evaluate_run(trained_tiny_run, [bad], {})


def test_eval_refuses_prereg_mismatch(trained_tiny_run, tmp_path):
    run = tmp_path / "r"
    shutil.copytree(trained_tiny_run, run)
    pre = tmp_path / "prereg.yaml"
    pre.write_text("rho: 0.5\n", encoding="utf-8")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    info["prereg_hash"] = file_hash(pre)
    info["config"]["prereg"] = str(pre)
    (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    pre.write_text("rho: 0.6\n", encoding="utf-8")  # prereg đổi sau khi train
    with pytest.raises(ValueError, match="prereg"):
        evaluate_run(run, ["ffpp_test"], {})


def test_eval_guard_blocks_train_val_sources():
    for w in ("ffpp_train", "ffpp_val", "compsynth_val", "whatever"):
        with pytest.raises(LeakageError):
            allow_read("eval", w)
    for w in ("ffpp_test", "compsynth_test", "cdf2"):
        allow_read("eval", w)


def test_run_info_saves_config(trained_tiny_run):
    info = json.loads((trained_tiny_run / "run_info.json").read_text(encoding="utf-8"))
    assert info["config"]["mode"] == "raw" and info["config"]["lora_r"] == 4


def test_eval_ffpp_test_all_frames(trained_tiny_run, ffpp_synth, vc):
    from compsynth_ff.dataset import FFppRawCore

    df = ev(trained_tiny_run, ["ffpp_test"], vc, ffpp_synth)
    core = FFppRawCore(ffpp_synth, "test")
    assert list(df.columns) == COLS and len(df) == len(core)  # không bỏ frame nào
    assert set(df.dataset) == {"ffpp_test"} and (df.flag == "").all() and set(df.label) == {0, 1}
    assert df.logit.notna().all() and set(df["mode"]) == {"raw"} and set(df.seed) == {0}
    assert set(df.video_id) == {core[i]["targets"]["video_id"] for i in range(len(core))}
    assert (trained_tiny_run / "scores.parquet").exists() and not list(trained_tiny_run.glob("*.tmp*"))


def test_eval_compsynth_test_cells(trained_tiny_run, ffpp_synth, vc):
    from compsynth_ff.dataset import EvalCore

    df = ev(trained_tiny_run, ["compsynth_test"], vc, ffpp_synth)
    core = EvalCore(ffpp_synth, "test")
    assert len(df) == len(core)
    kinds = core.man.kind.tolist()
    assert (df.cell[[k == "real" for k in kinds]] == "real").all()
    assert (df.cell[[k == "a0ff" for k in kinds]] == "a0ff").all()
    fk = df[[k == "fake" for k in kinds]]
    assert len(fk) and (fk.cell.str.count(r"\|") == 1).all() and (fk.label == 1).all()
    assert (df[[k != "fake" for k in kinds]].label == 0).all()
    assert df[[k == "a0ff" for k in kinds]].video_id.str.contains(r"\|a0ff\|").all()
    assert df[df.cell == "real"].video_id.str.endswith("|real").all()
    df2 = ev(trained_tiny_run, ["compsynth_test"], vc, ffpp_synth)  # cùng ckpt thì cùng điểm
    assert df.logit.to_numpy() == pytest.approx(df2.logit.to_numpy())


def test_no_face_video_kept(trained_tiny_run, synth_cross, ffpp_synth, vc):
    df = ev(trained_tiny_run, ["cdf2"], vc, ffpp_synth, {"cdf2": synth_cross})
    man = pd.read_parquet(synth_cross / "manifest.parquet")
    row = df[df.flag == "no_usable_frame"]
    assert len(row) == 1 and row.logit.iloc[0] == 0.0 and row.frame_idx.iloc[0] == -1
    assert row.video_id.iloc[0] == "YouTube-real__00002"
    err = df[df.flag == "decode_error"]
    assert len(err) == 1 and err.logit.iloc[0] == 0.0 and err.video_id.iloc[0] == "YouTube-real__00003"
    ok = df[df.flag == ""]
    assert len(ok) == int(man.keep.sum()) and (ok.frame_idx >= 0).all()
    assert set(df.video_id) == set(man.video_id)  # không video nào bị bỏ im lặng
    assert set(df.dataset) == {"cdf2"} and set(df.label) == {0, 1}
