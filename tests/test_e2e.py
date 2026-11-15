"""E2E qua CLI trên CPU: pilot (5 run) -> main (raw, raw_aug) -> compare. Trạng thái: chưa chạy thử trên GPU.
Dữ liệu tổng hợp, CLIP tí hon ngẫu nhiên (MINH HỌA); số AUC không có ý nghĩa."""
import functools
import json

import pytest

pytest.importorskip("torch")

from ffd import cli, experiment, progress  # noqa: E402
from ffd.compare import NOTE  # noqa: E402


def test_e2e_tiny(data_root, tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "ENABLED", progress.ENABLED)  # --no-progress đổi cờ toàn cục; trả lại sau test
    kw = dict(vision_config=tiny["vision_config"], allow_cpu=True, device="cpu")
    for name in ("run_pilot", "run_main"):  # CLI không truyền allow_cpu; test tiêm qua monkeypatch
        monkeypatch.setattr(experiment, name, functools.partial(getattr(experiment, name), **kw))
    ck = tmp_path / "ck"
    c = tmp_path / "c.yaml"
    c.write_text(f"data_root: {data_root.as_posix()}\nckpt_root: {ck.as_posix()}\nmodel_name: local-tiny\n"
                 "lora_r: 4\nlora_alpha: 8\ngrad_checkpointing: false\nnum_workers: 0\n", encoding="utf-8")
    common = ["--config", str(c), "--set", "max_epochs=2", "--no-progress"]
    assert cli.main(["pilot", *common]) == 0
    sel = json.loads((ck / "pilot" / "selected.json").read_text(encoding="utf-8"))
    assert sel["lr"] in (1e-4, 2e-4, 5e-4) and sel["p_jpeg"] in (0.0, 0.5, 1.0) and len(sel["val_auc"]) == 5
    r = json.loads((ck / "pilot" / "lr1e-4_p0.5" / "result.json").read_text(encoding="utf-8"))
    assert "val" in r and "ffpp_test" not in r and "celebdf" not in r  # pilot chỉ chấm val

    assert cli.main(["main", *common]) == 0
    for arm in ("raw", "raw_aug"):
        r = json.loads((ck / arm / "result.json").read_text(encoding="utf-8"))
        assert r["arm"] == arm and r["config"]["lr"] == sel["lr"] and r["config"]["p_jpeg"] == sel["p_jpeg"]
        assert {"val", "ffpp_test", "celebdf"} <= set(r)
    comp = json.loads((ck / "compare.json").read_text(encoding="utf-8"))
    assert comp["note"] == NOTE and "diff" in comp["ffpp_test"]
    # chạy lại: mọi run đã có result.json -> bỏ qua, không lỗi
    assert cli.main(["main", *common]) == 0
