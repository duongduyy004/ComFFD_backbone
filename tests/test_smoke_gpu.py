"""Smoke backbone thật trên GPU (chạy với --run-gpu). Trạng thái: chưa chạy thử trên GPU."""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ffd.config import load_train_config  # noqa: E402
from ffd.smoke import smoke  # noqa: E402


@pytest.mark.gpu
def test_smoke_real_model(tmp_path):
    cfg_path = os.environ.get("FFD_REAL_CONFIG", "backbones/clip_vit_b16/config.yaml")  # config có data_root trỏ tới dữ liệu thật đã tiền xử lý
    cfg = load_train_config(cfg_path, {"ckpt_root": str(tmp_path)})
    p = smoke(cfg, steps=50)
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["peak_vram_gb"] < 16 and s["sec_per_step"] > 0 and s["model_commit"]
    assert len(list((Path(p).parent / "samples").glob("rep_*"))) == 16
    assert len(list((Path(p).parent / "samples").glob("seam_*"))) == 16
