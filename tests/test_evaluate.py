"""Test evaluate_run -> result.json. Dữ liệu tổng hợp (MINH HỌA), CLIP tí hon ngẫu nhiên, CPU.
Trạng thái: chưa chạy thử trên GPU."""
import json
import shutil

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ffd.config import TrainConfig  # noqa: E402
from ffd.evaluate import evaluate_run  # noqa: E402
from ffd.guards import LeakageError, allow_read  # noqa: E402
from ffd.train import train  # noqa: E402


@pytest.fixture(scope="module")
def tiny_mod():
    pytest.importorskip("transformers")
    from transformers import CLIPVisionConfig

    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=24,
                          num_attention_heads=4, image_size=224, patch_size=14)
    return dict(model_name="local-tiny", lora_r=4, lora_alpha=8, lora_dropout=0.0,
                lora_targets=("q_proj", "v_proj"), grad_checkpointing=False, vision_config=vc)


@pytest.fixture(scope="module")
def trained(tiny_mod, data_root, tmp_path_factory):
    """Một run tí hon train trên CPU (2 epoch); trả (thư mục run, vision_config)."""
    kw = {k: v for k, v in tiny_mod.items() if k != "vision_config"}
    cfg = TrainConfig(data_root=data_root, ckpt_root=tmp_path_factory.mktemp("ck"), arm="raw", seed=0,
                      batch_size=32, max_epochs=2, patience=5, num_workers=0, **kw)
    return train(cfg, vision_config=tiny_mod["vision_config"], allow_cpu=True), tiny_mod["vision_config"]


def _ev(trained, **kw):
    run, vc = trained
    kw.setdefault("device", "cpu")
    return evaluate_run(run, vision_config=vc, **kw)


def test_result_json_schema(trained, data_root):
    p = _ev(trained, data_root=data_root)
    r = json.loads(p.read_text(encoding="utf-8"))
    assert set(r) >= {"arm", "config", "best_epoch", "epochs_run", "stopped_early", "total_steps",
                      "val", "ffpp_test", "celebdf"}
    assert set(r["ffpp_test"]["auc_per_method"]) == {"DF", "F2F", "FS", "NT", "FSh"}
    assert r["ffpp_test"]["n_videos_dropped"] == 1
    assert r["arm"] == "raw" and r["epochs_run"] == 2 and r["config"]["lora"]["r"] == 4
    assert r["config"]["manifest_sha256"]["raw"] and r["config"]["params_json_sha256"]
    from ffd.data.ffpp import data_hashes
    assert r["config"]["manifest_sha256"]["celebdf"] == data_hashes(data_root)["manifest_sha256"]["celebdf"]
    md = (trained[0] / "result.md").read_text(encoding="utf-8")
    assert "| dataset | n_frames | ACC | AUC | PREC | RECALL | F1 | AUC_video |" in md
    assert "best_epoch" in md and "AUC theo method" in md
    for ds in ("val", "ffpp_test", "celebdf"):
        v = r[ds]
        row = next(x for x in md.splitlines() if x.startswith(f"| {ds} |"))
        assert row == (f"| {ds} | {v['n_frames']} | {v['acc']:.4f} | {v['auc']:.4f} | {v['precision']:.4f} | "
                       f"{v['recall']:.4f} | {v['f1']:.4f} | {v['auc_video']:.4f} |")
        s = pd.read_parquet(trained[0] / f"scores_{ds}.parquet")
        assert list(s.columns) == ["video_id", "frame_idx", "label", "method", "logit"]
        assert len(s) == r[ds]["n_frames"]


def test_eval_without_celebdf(trained, data_root, tmp_path, capsys):
    root = tmp_path / "dr"
    shutil.copytree(data_root, root, ignore=shutil.ignore_patterns("celebdf"))
    r = json.loads(_ev(trained, data_root=root).read_text(encoding="utf-8"))
    out = capsys.readouterr().out
    head = next(x for x in out.splitlines() if x.startswith("dataset"))
    assert head.split() == ["dataset", "n_frames", "ACC", "AUC", "PREC", "RECALL", "F1", "AUC_video"]
    assert any(x.startswith("celebdf") and "skipped: " in x for x in out.splitlines())
    assert any(x.startswith("ffpp_test") and len(x.split()) == 8 for x in out.splitlines())
    assert list(r["celebdf"]) == ["skipped"] and "celebdf/manifest.parquet" in r["celebdf"]["skipped"]
    assert r["config"]["manifest_sha256"].get("celebdf") is None  # không chấm -> không ghi hash
    assert "auc" in r["ffpp_test"]


def test_eval_val_only(trained, data_root):
    r = json.loads(_ev(trained, data_root=data_root, datasets=("val",)).read_text(encoding="utf-8"))
    assert "val" in r and "ffpp_test" not in r and "celebdf" not in r


def test_eval_rejects_changed_data(trained, data_root, tmp_path):
    root = tmp_path / "dr"
    shutil.copytree(data_root, root)
    m = root / "ffpp" / "manifest_raw.parquet"
    pd.read_parquet(m).iloc[:-1].to_parquet(m)  # sửa manifest sau khi train
    with pytest.raises(ValueError, match="manifest_raw"):
        _ev(trained, data_root=root)


def test_eval_guard():
    for w in ("ffpp_val", "ffpp_test", "celebdf"):
        allow_read("eval", w)
    for w in ("ffpp_train", "ffpp_aug", "whatever"):
        with pytest.raises(LeakageError):
            allow_read("eval", w)
    with pytest.raises(LeakageError):
        allow_read("train", "ffpp_test")


def _copy_run(trained, tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(trained[0], dst)
    return dst, trained[1]


@pytest.mark.parametrize("missing", ["done.flag", "best.pt"])
def test_eval_refuses_unfinished_run(trained, data_root, tmp_path, missing):
    run, vc = _copy_run(trained, tmp_path)
    (run / missing).unlink()
    with pytest.raises(ValueError, match=missing):
        evaluate_run(run, data_root=data_root, vision_config=vc, device="cpu")


def test_eval_rejects_model_commit_mismatch(trained, data_root, tmp_path):
    run, vc = _copy_run(trained, tmp_path)
    p = run / "run_info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    info["model_commit"] = "khac"
    p.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="commit"):
        evaluate_run(run, data_root=data_root, vision_config=vc, device="cpu")


def test_eval_requires_data_hashes(trained, data_root, tmp_path):
    run, vc = _copy_run(trained, tmp_path)
    p = run / "run_info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    del info["data_hashes"]
    p.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu data_hashes"):
        evaluate_run(run, data_root=data_root, vision_config=vc, device="cpu")


def test_eval_rejects_best_epoch_none(trained, data_root, tmp_path):
    run, vc = _copy_run(trained, tmp_path)
    p = run / "done.flag"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["best_epoch"] = None
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="best_epoch"):
        evaluate_run(run, data_root=data_root, vision_config=vc, device="cpu")


def test_eval_rejects_changed_params_json(trained, data_root, tmp_path):
    root = tmp_path / "dr"
    shutil.copytree(data_root, root)
    p = root / "ffpp" / "params.json"
    p.write_bytes(p.read_bytes() + b"\n ")
    run, vc = _copy_run(trained, tmp_path)
    for p in [*run.glob("scores_*.parquet"), run / "result.json"]:
        p.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="params.json"):
        evaluate_run(run, data_root=root, vision_config=vc, device="cpu")
    assert not list(run.glob("scores_*.parquet")) and not (run / "result.json").exists()  # không chấm gì


def test_eval_rejects_changed_celebdf_manifest(trained, data_root, tmp_path):
    root = tmp_path / "dr"
    shutil.copytree(data_root, root)
    m = root / "celebdf" / "manifest.parquet"
    pd.read_parquet(m).iloc[:-1].to_parquet(m)
    run, vc = _copy_run(trained, tmp_path)
    for p in [*run.glob("scores_*.parquet"), run / "result.json"]:
        p.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="celebdf"):
        evaluate_run(run, data_root=root, vision_config=vc, device="cpu")
    assert not list(run.glob("scores_*.parquet"))  # kiểm trước vòng chấm: val cũng không bị ghi
    # chỉ chấm val thì không đụng celebdf
    assert _ev(trained, data_root=root, datasets=("val",)).exists()
