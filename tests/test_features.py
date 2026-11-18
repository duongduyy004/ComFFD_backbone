"""Test cache đặc trưng CLS (ffd.features) trên CLIP tí hon + data_root tổng hợp (MINH HỌA). CPU."""
import json

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("transformers")

from ffd import features as F  # noqa: E402
from ffd.data.ffpp import frames  # noqa: E402
from ffd.data.transforms import eval_transform  # noqa: E402
from ffd.io_utils import read_image  # noqa: E402

_SPLITS = {"train": ("ffpp", "train"), "val": ("ffpp", "val"), "ffpp_test": ("ffpp", "test"),
           "celebdf": ("celebdf", None)}


def _vc():
    from transformers import CLIPVisionConfig

    return CLIPVisionConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=24,
                            num_attention_heads=4, image_size=224, patch_size=14)


def _extract(data_root, root, **kw):
    kw.setdefault("datasets", ("val",))
    return F.extract_features(data_root, root, model_name="local-tiny", model_revision=None,
                              vision_config=_vc(), batch_size=4, num_workers=0, chunk=5, device="cpu", **kw)


@pytest.fixture(scope="module")
def feat_dir(data_root, tmp_path_factory):
    return _extract(data_root, tmp_path_factory.mktemp("feat"), datasets=F.FEAT_DATASETS)


def test_shapes_and_index_order(feat_dir, data_root):
    for ds, (d, s) in _SPLITS.items():
        cls, idx, meta = F.load_features(feat_dir, ds)
        fr = frames(data_root, d, s)
        assert cls.shape == (len(fr), 25, 32) and cls.dtype == np.float16
        assert list(idx.video_id) == list(fr.video_id) and list(idx.frame_idx) == list(fr.frame_idx)
        assert list(idx.label) == list(fr.label)
        assert meta["commit"] == "local" and meta["num_layers"] == 24 and meta["N"] == len(fr)
        assert "n_videos_dropped" in meta and meta["verify"]["max_abs_pooler"] < 1e-2


def test_last_slot_is_pooler_and_slots_are_hidden_cls(feat_dir, data_root):
    cls, idx, _ = F.load_features(feat_dir, "val")
    vit, _ = F._build_model("local-tiny", None, _vc())
    x = torch.stack([eval_transform(read_image(p)) for p in idx.path[:3]])
    with torch.no_grad():
        o = vit(pixel_values=x, output_hidden_states=True)
    c = np.asarray(cls[:3], dtype=np.float32)
    assert np.allclose(c[:, 24], o.pooler_output.numpy(), atol=1e-2)
    for l in (1, 12, 24):
        assert np.allclose(c[:, l - 1], o.hidden_states[l][:, 0].numpy(), rtol=1e-2, atol=1e-2)


def test_resume_after_interrupt_identical(tmp_path, data_root):
    a = _extract(data_root, tmp_path / "a")
    _extract(data_root, tmp_path / "b", max_chunks=1)
    assert not (tmp_path / "b" / "local" / "val" / "meta.json").exists()
    st = json.loads((tmp_path / "b" / "local" / "val" / "state.json").read_text(encoding="utf-8"))
    assert st["n_done"] == 8 and st["batch_size"] == 4  # khối ghi ở ranh giới batch: >= chunk (5), bội của batch (4)
    b = _extract(data_root, tmp_path / "b")
    assert np.array_equal(np.load(a / "val" / "cls.npy"), np.load(b / "val" / "cls.npy"))


def test_meta_mismatch_refuses(tmp_path, data_root):
    d = _extract(data_root, tmp_path)
    p = d / "val" / "state.json"
    st = json.loads(p.read_text(encoding="utf-8"))
    st["key"]["transform_id"] = "x"
    p.write_text(json.dumps(st), encoding="utf-8")
    mt = (d / "val" / "cls.npy").stat().st_mtime_ns
    with pytest.raises(ValueError, match="khác"):
        _extract(data_root, tmp_path)
    assert (d / "val" / "cls.npy").stat().st_mtime_ns == mt


def test_load_incomplete_raises(tmp_path, data_root):
    _extract(data_root, tmp_path, max_chunks=1)
    with pytest.raises(FileNotFoundError, match="chưa hoàn tất"):
        F.load_features(tmp_path / "local", "val")


def test_check_order_mismatch_raises(feat_dir, tmp_path):
    _, idx, _ = F.load_features(feat_dir, "val")
    good = idx[["video_id", "frame_idx", "label"]].assign(method="x", logit=0.0)
    good.to_parquet(tmp_path / "ok.parquet")
    F.check_order(idx, tmp_path / "ok.parquet")
    good.iloc[::-1].reset_index(drop=True).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="thứ tự"):
        F.check_order(idx, tmp_path / "bad.parquet")


def test_feat_dir_of_uses_full_commit(tmp_path, data_root):
    from ffd.config import TrainConfig

    rev = "32bd64288804d66eefd0ccbe215aa642df71cc41"
    assert F.feat_dir_of(TrainConfig(data_root=data_root, model_revision=rev), tmp_path) == tmp_path / rev


def test_main_cli(tmp_path, data_root, monkeypatch):
    real = F._build_model
    monkeypatch.setattr(F, "_build_model", lambda name, rev, vc: real("local-tiny", None, _vc()))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"data_root": str(data_root), "num_workers": 0}), encoding="utf-8")
    assert F.main(["--config", str(cfg), "--feat-root", str(tmp_path / "f"), "--datasets", "val,celebdf",
                   "--batch-size", "4", "--device", "cpu"]) == 0
    for ds in ("val", "celebdf"):
        assert (tmp_path / "f" / "local" / ds / "meta.json").is_file()


def test_verify_close_relative_tolerance():  # fp16 làm tròn ~0.03 ở |x| ~ 40: không được báo lỗi giả
    ref = np.full((4, 25, 32), 40.015, np.float32)
    got = ref.astype(np.float16).astype(np.float32)
    out = F._verify_close(got, ref)
    assert out["max_abs_pooler"] > 1e-2 and out["max_abs_ref"] == pytest.approx(40.015, rel=1e-4)
    with pytest.raises(ValueError, match="lệch"):
        F._verify_close(got + 1.0, ref)


def test_nonfinite_features_refused(tmp_path, data_root, monkeypatch):
    real = F.cls_slots
    monkeypatch.setattr(F, "cls_slots", lambda vit, x: real(vit, x) * float("inf"))
    with pytest.raises(ValueError, match="không hữu hạn"):
        _extract(data_root, tmp_path)
    assert not (tmp_path / "local" / "val" / "meta.json").exists()


def _bad_run(tmp_path, data_root, commit="local"):
    """Run giả có scores_val.parquet sai thứ tự và run_info với model_commit cho trước."""
    run = tmp_path / "raw"
    run.mkdir()
    fr = frames(data_root, "ffpp", "val")
    fr[["video_id", "frame_idx", "label"]].iloc[::-1].reset_index(drop=True).assign(method="x", logit=0.0)         .to_parquet(run / "scores_val.parquet")
    (run / "run_info.json").write_text(json.dumps({"model_commit": commit}), encoding="utf-8")
    return run


def test_check_against_runs_before_forward(tmp_path, data_root, monkeypatch):
    run = _bad_run(tmp_path, data_root)
    monkeypatch.setattr(F, "cls_slots", lambda vit, x: (_ for _ in ()).throw(RuntimeError("forward đã chạy")))
    with pytest.raises(ValueError, match="thứ tự"):
        _extract(data_root, tmp_path / "f", check_against=run)
    assert not (tmp_path / "f" / "local" / "val" / "cls.npy").exists()


def test_check_against_commit_mismatch(tmp_path, data_root, monkeypatch):
    run = _bad_run(tmp_path, data_root, commit="abc")
    with pytest.raises(ValueError, match="commit"):
        _extract(data_root, tmp_path / "f", check_against=run)


def test_one_loader_per_dataset(tmp_path, data_root, monkeypatch):
    n, real = [], F._loader
    monkeypatch.setattr(F, "_loader", lambda *a, **k: n.append(1) or real(*a, **k))
    _extract(data_root, tmp_path)  # val 48 frame, chunk 5 -> trước đây ~10 loader
    assert len(n) == 2  # 1 loader trích + 1 loader kiểm 16 frame


def test_precision_recorded(feat_dir):
    _, _, meta = F.load_features(feat_dir, "val")
    assert meta["precision"] == "fp32"  # CPU; CUDA mặc định "tf32"


def test_progress_prints_rate(tmp_path, data_root, capsys):
    _extract(data_root, tmp_path)
    assert "frame/s" in capsys.readouterr().out
