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

COLS = ["run", "mode", "seed", "split", "dataset", "video_id", "frame_idx", "label", "logit", "cell", "flag",
        "best_step"]


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
    out = evaluate_run(run, ds, cross or {}, ff_cfg=ffpp, vision_config=vc, device="cpu")
    assert out == run
    return pd.concat([pd.read_parquet(run / f"scores_{d}.parquet") for d in ds], ignore_index=True)


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
    assert (trained_tiny_run / "scores_ffpp_test.parquet").exists() and not list(trained_tiny_run.glob("*.tmp*"))


def test_eval_compsynth_test_cells(trained_tiny_run, ffpp_synth, vc):
    from compsynth_ff.dataset import EvalCore

    df = ev(trained_tiny_run, ["compsynth_test"], vc, ffpp_synth)
    core = EvalCore(ffpp_synth, "test")
    assert len(df) == len(core)
    kinds = core.man.kind.tolist()
    assert (df.cell[[k == "real" for k in kinds]] == "real").all()
    fk = df[[k == "fake" for k in kinds]]
    assert len(fk) and (fk.cell.str.count(r"\|") == 1).all() and (fk.label == 1).all()
    assert (df[[k != "fake" for k in kinds]].label == 0).all()
    assert fk.cell.str.fullmatch(r"(eyes|nose|mouth|skin)(\+(eyes|nose|mouth|skin))*\|(DF|F2F|FS|NT|FSh)").all()
    assert (fk.video_id.str.count(r"\|") == 2).all()
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


def test_ffpp_test_video_without_kept_frames_flagged(trained_tiny_run, ffpp_synth, vc, monkeypatch):
    import compsynth_ff.dataset as D

    full = ev(trained_tiny_run, ["ffpp_test"], vc, ffpp_synth)
    victim = sorted(full.video_id.unique())[0]
    Orig = D.FFppRawCore

    class Dropping(Orig):  # mô phỏng video không còn frame keep
        def __init__(self, cfg, subset):
            super().__init__(cfg, subset)
            self.items = [it for it in self.items if f"{it[0]}_{it[2]}" != victim]

    monkeypatch.setattr(D, "FFppRawCore", Dropping)
    df = ev(trained_tiny_run, ["ffpp_test"], vc, ffpp_synth)
    row = df[df.flag == "no_usable_frame"]
    assert len(row) == 1 and row.video_id.iloc[0] == victim and row.frame_idx.iloc[0] == -1
    assert row.logit.iloc[0] == 0.0 and row.label.iloc[0] == int(not victim.endswith("_real"))
    assert set(df.video_id) == set(full.video_id) and len(df) == len(full) - (full.video_id == victim).sum() + 1


def test_ffpp_test_unexpected_video_rejected(trained_tiny_run, ffpp_synth, vc, monkeypatch):
    import dataclasses

    import compsynth_ff.select_cells as SC

    cfg = dataclasses.replace(ffpp_synth, data_root=Path("khong_ton_tai_MINH_HOA"))
    orig = SC.load_frames
    # tập kỳ vọng (SC.load_frames) thiếu một video mà FFppRawCore (dataset.load_frames) vẫn chấm
    monkeypatch.setattr(SC, "load_frames", lambda c: (lambda fr: fr[fr.tid != fr[fr.subset == "test"].tid.iloc[0]])(orig(c)))
    with pytest.raises(ValueError, match="không thuộc tập test"):
        ev(trained_tiny_run, ["ffpp_test"], vc, cfg)


def test_reevaluating_one_dataset_keeps_other_files(trained_tiny_run, ffpp_synth, vc):
    ev(trained_tiny_run, ["ffpp_test", "compsynth_test"], vc, ffpp_synth)
    a = trained_tiny_run / "scores_ffpp_test.parquet"
    b = trained_tiny_run / "scores_compsynth_test.parquet"
    assert a.exists() and b.exists()
    before = b.stat().st_mtime_ns, pd.read_parquet(b).logit.sum()
    ev(trained_tiny_run, ["ffpp_test"], vc, ffpp_synth)
    assert (b.stat().st_mtime_ns, pd.read_parquet(b).logit.sum()) == before  # file dataset khác không bị đụng
    assert not (trained_tiny_run / "scores.parquet").exists()


def test_eval_refuses_rebuilt_config_hash_mismatch(trained_tiny_run, tmp_path):
    run = tmp_path / "r"
    shutil.copytree(trained_tiny_run, run)
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    info["config"]["lora_r"] = 8
    (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="config_hash"):
        evaluate_run(run, ["ffpp_test"], {})


def test_compsynth_unknown_kind_keyerror(trained_tiny_run, ffpp_synth, vc, monkeypatch):
    from compsynth_ff.dataset import L0_CODE

    monkeypatch.delitem(L0_CODE, "fake")
    with pytest.raises(KeyError):
        ev(trained_tiny_run, ["compsynth_test"], vc, ffpp_synth)


def test_ffpp_expected_set_without_data_root(trained_tiny_run, ffpp_synth, vc, monkeypatch):
    import dataclasses
    import json as _j

    cfg = dataclasses.replace(ffpp_synth, data_root=Path("khong_ton_tai_MINH_HOA"))
    df = ev(trained_tiny_run, ["ffpp_test"], vc, cfg)
    assert (df.flag == "").all() and len(df) > 0
    info = _j.loads((trained_tiny_run / "eval_info.json").read_text(encoding="utf-8"))
    assert info["ffpp_test"]["expected_source"] == "artifacts_only"
    ev(trained_tiny_run, ["ffpp_test"], vc, ffpp_synth)
    info = _j.loads((trained_tiny_run / "eval_info.json").read_text(encoding="utf-8"))
    assert info["ffpp_test"]["expected_source"] == "artifacts+data_root"


def test_ffpp_video_all_keep_false_in_frames_flagged(trained_tiny_run, ffpp_synth, vc, monkeypatch):
    import dataclasses

    import compsynth_ff.select_cells as SC

    cfg = dataclasses.replace(ffpp_synth, data_root=Path("khong_ton_tai_MINH_HOA"))  # chỉ dùng artifact
    full = ev(trained_tiny_run, ["ffpp_test"], vc, cfg)
    victim = sorted(full.video_id.unique())[0]
    tid, src = victim.rsplit("_", 1)
    orig = SC.load_frames

    def patched(c):
        fr = orig(c).copy()
        fr.loc[(fr.tid == tid) & (fr.src == src), "keep"] = False  # còn trong frames nhưng không frame nào keep
        return fr

    monkeypatch.setattr(SC, "load_frames", patched)
    import compsynth_ff.dataset as D
    monkeypatch.setattr(D, "load_frames", patched)
    df = ev(trained_tiny_run, ["ffpp_test"], vc, cfg)
    row = df[df.flag == "no_usable_frame"]
    assert len(row) == 1 and row.video_id.iloc[0] == victim and row.logit.iloc[0] == 0.0
    assert set(df.video_id) == set(full.video_id)


def test_run_info_records_model_and_ffpp_provenance(trained_tiny_run, ffpp_synth):
    """I2/I3: run_info ghi model_revision (None khi chưa ghim), model_commit ("local" với vision_config),
    ffpp_config_hash = ffpp_config_hash(ff_cfg) (R1: Config đã parse, bỏ đường dẫn) và
    ffpp_stage_hash = stage_hash(ff_cfg, 'render')."""
    from compsynth_ff.config import stage_hash

    from ffd.rundir import ffpp_config_hash

    info = json.loads((trained_tiny_run / "run_info.json").read_text(encoding="utf-8"))
    assert info["model_revision"] is None and info["model_commit"] == "local"
    assert info["ffpp_config_hash"] == ffpp_config_hash(ffpp_synth)
    assert info["ffpp_stage_hash"] == stage_hash(ffpp_synth, "render")


def test_eval_refuses_changed_ffpp_artifacts(trained_tiny_run, ffpp_synth, vc, tmp_path):
    """I3: ff_cfg lúc eval khác stage render, hoặc hash file config FF++ khác -> ValueError."""
    import dataclasses

    run = tmp_path / "r"
    shutil.copytree(trained_tiny_run, run)
    other = dataclasses.replace(ffpp_synth, jpeg_q=(ffpp_synth.jpeg_q[0], ffpp_synth.jpeg_q[1] - 1))
    with pytest.raises(ValueError, match="ffpp_config_hash|ffpp_stage_hash"):  # R1: config hash phủ cả stage
        evaluate_run(run, ["ffpp_test"], {}, ff_cfg=other, vision_config=vc, device="cpu")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    info["ffpp_config_hash"] = "deadbeef0000"
    (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="ffpp_config_hash"):
        evaluate_run(run, ["ffpp_test"], {}, ff_cfg=ffpp_synth, vision_config=vc, device="cpu")


def test_eval_refuses_model_commit_mismatch(trained_tiny_run, ffpp_synth, vc, tmp_path):
    run = tmp_path / "r"
    shutil.copytree(trained_tiny_run, run)
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    info["model_commit"] = "abc123"
    (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="commit"):
        evaluate_run(run, ["ffpp_test"], {}, ff_cfg=ffpp_synth, vision_config=vc, device="cpu")


def test_scores_carry_best_step(trained_tiny_run, ffpp_synth, vc):
    """I6: mỗi dòng scores mang step của best.pt đã chấm."""
    df = ev(trained_tiny_run, ["ffpp_test"], vc, ffpp_synth)
    step = int(torch.load(trained_tiny_run / "best.pt", weights_only=False)["step"])
    assert set(df.best_step) == {step} and df.best_step.dtype == "int64"


def test_eval_accepts_moved_paths(trained_tiny_run, ffpp_synth, vc, tmp_path):
    """R1: eval trên "máy khác": data_root/splits_dir khác lúc train, file config FF++ và prereg ở chỗ mới
    -> vẫn chấm được (hash chỉ phụ thuộc tham số và nội dung prereg)."""
    import dataclasses

    run = tmp_path / "r"
    shutil.copytree(trained_tiny_run, run)
    pre_old, pre_new = tmp_path / "old" / "prereg.yaml", tmp_path / "new" / "prereg.yaml"
    pre_new.parent.mkdir()
    pre_new.write_text("rho: 0.5\n", encoding="utf-8")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    info["prereg_hash"] = file_hash(pre_new)
    info["config"]["prereg"] = str(pre_old)  # đường dẫn của máy train, không tồn tại ở đây
    (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    moved = dataclasses.replace(ffpp_synth, data_root=tmp_path / "khong_co", splits_dir=tmp_path / "sp_khac")
    with pytest.raises(ValueError, match="--prereg"):
        evaluate_run(run, ["ffpp_test"], {}, ff_cfg=moved, vision_config=vc, device="cpu")
    evaluate_run(run, ["ffpp_test"], {}, ff_cfg=moved, prereg=pre_new, vision_config=vc, device="cpu")
    assert (run / "scores_ffpp_test.parquet").exists()
    pre_new.write_text("rho: 0.6\n", encoding="utf-8")  # nội dung đổi -> vẫn từ chối
    with pytest.raises(ValueError, match="prereg"):
        evaluate_run(run, ["ffpp_test"], {}, ff_cfg=moved, prereg=pre_new, vision_config=vc, device="cpu")
