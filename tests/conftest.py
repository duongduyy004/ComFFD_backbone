"""Fixture dùng chung cho test ffd. Trạng thái: chưa chạy thử.

`data_root` = thư mục data_root TỔNG HỢP (MINH HỌA, không phải FF++/Celeb-DF thật) do `synth_ffpp.make_data_root`
sinh đúng hợp đồng dữ liệu (ffpp/, celebdf/). Không import preprocessing. Scope session.
"""
import pytest


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
def data_root(tmp_path_factory):
    """Path tới data_root tổng hợp (MINH HỌA): 4/2/2 video nguồn train/val/test, real + 5 method, 4 frame/video."""
    from synth_ffpp import make_data_root

    return make_data_root(tmp_path_factory.mktemp("data_root"))


@pytest.fixture
def tiny():
    """Kwargs dựng ClipLoraTrunk nhỏ (random weights, không cần mạng). MINH HỌA, không phải CLIP thật."""
    pytest.importorskip("transformers")
    from transformers import CLIPVisionConfig

    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=24,
                          num_attention_heads=4, image_size=224, patch_size=14)
    return dict(model_name="local-tiny", lora_r=4, lora_alpha=8, lora_dropout=0.0,
                lora_targets=("q_proj", "v_proj"), grad_checkpointing=False, vision_config=vc)
