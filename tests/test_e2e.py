"""E2E qua CLI trên CPU: main 3 arm -> videoboot, realfp, summary. Trạng thái: chưa chạy thử trên GPU.
Dữ liệu tổng hợp, CLIP tí hon ngẫu nhiên (MINH HỌA); số AUC không có ý nghĩa."""
import functools
import json

import pytest

pytest.importorskip("torch")

from ffd import cli, experiment, progress  # noqa: E402
from ffd.config import ARMS  # noqa: E402


def test_e2e_tiny(data_root, tiny, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(progress, "ENABLED", progress.ENABLED)  # --no-progress đổi cờ toàn cục; trả lại sau test
    kw = dict(vision_config=tiny["vision_config"], allow_cpu=True, device="cpu")
    # CLI không truyền allow_cpu; test tiêm qua monkeypatch
    monkeypatch.setattr(experiment, "run_main", functools.partial(experiment.run_main, **kw))
    ck = tmp_path / "ck"
    c = tmp_path / "c.yaml"
    c.write_text(f"data_root: {data_root.as_posix()}\nckpt_root: {ck.as_posix()}\nmodel_name: local-tiny\n"
                 "lora_r: 4\nlora_alpha: 8\ngrad_checkpointing: false\nnum_workers: 0\n"
                 "lr: 1.0e-4\np_jpeg: 0.5\n", encoding="utf-8")
    common = ["--config", str(c), "--set", "max_epochs=1", "--no-progress"]
    assert cli.main(["main", *common, "--B", "30"]) == 0
    for arm in ARMS:
        r = json.loads((ck / arm / "result.json").read_text(encoding="utf-8"))
        assert r["arm"] == arm and r["config"]["lr"] == 1e-4 and r["config"]["p_jpeg"] == 0.5
        assert {"val", "ffpp_test", "celebdf"} <= set(r)
        assert ("aug_stats" in r) == (arm != "raw")
        md = (ck / arm / "result.md").read_text(encoding="utf-8")
        for ds in ("val", "ffpp_test", "celebdf"):
            assert f"| {ds} | {r[ds]['n_frames']} | {r[ds]['acc']:.4f} | {r[ds]['auc']:.4f} |" in md
    for f in ("videoboot.json", "realfp_raw__raw_aug.json", "realfp_raw_aug__raw_rep.json", "summary.json"):
        assert (ck / f).is_file(), f
    s = json.loads((ck / "summary.json").read_text(encoding="utf-8"))
    assert set(s["decision"]) == {"R1_vs_R0", "R2_vs_R1", "run_R3"}
    assert "Tổng hợp main" in capsys.readouterr().out
    # chạy lại: mọi run đã có result.json -> bỏ qua, không lỗi; summary ghi lại
    assert cli.main(["main", *common, "--B", "30"]) == 0
    assert (ck / "summary.md").exists()
