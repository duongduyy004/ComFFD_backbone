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
    for ds in ("val", "ffpp_test", "celebdf"):
        s = pd.read_parquet(trained[0] / f"scores_{ds}.parquet")
        assert list(s.columns) == ["video_id", "frame_idx", "label", "method", "logit"]
        assert len(s) == r[ds]["n_frames"]


def test_eval_without_celebdf(trained, data_root, tmp_path):
    root = tmp_path / "dr"
    shutil.copytree(data_root, root, ignore=shutil.ignore_patterns("celebdf"))
    r = json.loads(_ev(trained, data_root=root).read_text(encoding="utf-8"))
    assert list(r["celebdf"]) == ["skipped"] and "celebdf/manifest.parquet" in r["celebdf"]["skipped"]
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
