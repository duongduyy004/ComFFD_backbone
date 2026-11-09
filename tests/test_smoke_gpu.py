"""Smoke ViT-L thật trên GPU (chạy với --run-gpu). Trạng thái: chưa chạy thử."""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ffd.config import load_train_config  # noqa: E402
from ffd.smoke import smoke  # noqa: E402


@pytest.mark.gpu
def test_smoke_real_model(tmp_path):
    cfg_path = os.environ.get("FFD_REAL_CONFIG", "config.example.yaml")  # trỏ tới ffpp_config thật đã tiền xử lý
    cfg = load_train_config(cfg_path, {"runs_root": str(tmp_path)})
    p = smoke(cfg, steps=50)
    s = json.loads(p.read_text())
    assert s["peak_vram_gb"] < 16 and s["sec_per_step"] > 0
    assert len(list((Path(p).parent / "smoke_samples").glob("comp_*.png"))) == 16
    assert len(list((Path(p).parent / "smoke_samples").glob("sb_*.png"))) == 16
