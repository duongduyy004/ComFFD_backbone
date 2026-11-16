"""E2E qua CLI trên CPU: main (raw, raw_aug) -> compare -> summary.md. Trạng thái: chưa chạy thử trên GPU.
Dữ liệu tổng hợp, CLIP tí hon ngẫu nhiên (MINH HỌA); số AUC không có ý nghĩa."""
import functools
import json

import pytest

pytest.importorskip("torch")

from ffd import cli, experiment, progress  # noqa: E402
from ffd.compare import NOTE  # noqa: E402


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
    common = ["--config", str(c), "--set", "max_epochs=2", "--no-progress"]
    assert cli.main(["main", *common]) == 0
    for arm in ("raw", "raw_aug"):
        r = json.loads((ck / arm / "result.json").read_text(encoding="utf-8"))
        assert r["arm"] == arm and r["config"]["lr"] == 1e-4 and r["config"]["p_jpeg"] == 0.5
        assert {"val", "ffpp_test", "celebdf"} <= set(r)
        md = (ck / arm / "result.md").read_text(encoding="utf-8")
        for ds in ("val", "ffpp_test", "celebdf"):
            assert f"| {ds} | {r[ds]['n_frames']} | {r[ds]['acc']:.4f} | {r[ds]['auc']:.4f} |" in md
    assert not (ck / "pilot").exists()
    comp = json.loads((ck / "compare.json").read_text(encoding="utf-8"))
    assert comp["note"] == NOTE and "diff" in comp["ffpp_test"]
    s = (ck / "summary.md").read_text(encoding="utf-8")
    assert "| arm | dataset | ACC | AUC | PREC | RECALL | F1 |" in s and "p DeLong" in s
    assert sum(x.startswith("| raw") for x in s.splitlines()) == 6  # 2 arm x 3 dataset
    assert "Tổng hợp main" in capsys.readouterr().out
    # chạy lại: mọi run đã có result.json -> bỏ qua, không lỗi; summary.md ghi lại
    assert cli.main(["main", *common]) == 0
    assert (ck / "summary.md").exists()
