"""Fixture dùng chung cho test ffd. Trạng thái: chưa chạy thử.

`ffpp_synth` dựng dataset đã tiền xử lý từ FF++ TỔNG HỢP (MINH HỌA, không phải dữ liệu thật),
theo cách của `ready_cfg` trong preprocessing/tests/test_dataset.py. Scope session, chạy một lần (chậm).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_PRE_TESTS = Path(__file__).resolve().parents[2] / "preprocessing" / "tests"

# conftest.py của preprocessing trùng tên module với file này, nên nạp bằng đường dẫn dưới tên riêng.
if str(_PRE_TESTS) not in sys.path:
    sys.path.insert(0, str(_PRE_TESTS))
_spec = importlib.util.spec_from_file_location("pre_conftest", _PRE_TESTS / "conftest.py")
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)

fake_reenc = _pre.fake_reenc
make_cfg = _pre.make_cfg
write_mp4 = _pre.write_mp4
synth_root = _pre.synth_root  # fixture (session) của preprocessing


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False, help="chạy test cần model thật/GPU")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip = pytest.mark.skip(reason="cần --run-gpu")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def ffpp_synth(synth_root, tmp_path_factory):
    """Config đã chạy extract -> select -> render val/test trên FF++ tổng hợp (MINH HỌA)."""
    from synth import FakeDetector, FakeParser
    from compsynth_ff.extract import run_extract
    from compsynth_ff.pairs import discover_targets
    from compsynth_ff.render_eval import render_subset
    from compsynth_ff.select_cells import run_select

    cfg = make_cfg(synth_root, tmp_path_factory.mktemp("ds"))
    run_extract(cfg, discover_targets(cfg), FakeDetector, FakeParser, reencoder=fake_reenc)
    run_select(cfg)
    render_subset(cfg, "val")
    render_subset(cfg, "test")
    return cfg


@pytest.fixture
def tiny():
    """Kwargs dựng ClipLoraTrunk nhỏ (random weights, không cần mạng). MINH HỌA, không phải CLIP thật."""
    pytest.importorskip("transformers")
    from transformers import CLIPVisionConfig

    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=24,
                          num_attention_heads=4, image_size=336, patch_size=14)
    return dict(model_name="local-tiny", lora_r=4, lora_alpha=8, lora_dropout=0.0,
                lora_targets=("q_proj", "v_proj"), grad_checkpointing=False, vision_config=vc)
