"""E2E qua CLI trên CPU: main 4 arm -> compare -> summary.md. Trạng thái: chưa chạy thử trên GPU.
Dữ liệu tổng hợp, CLIP tí hon ngẫu nhiên (MINH HỌA); số AUC không có ý nghĩa."""
import functools
import json

import pytest

pytest.importorskip("torch")

from ffd import cli, experiment, progress  # noqa: E402
from ffd.compare import NOTE, PAIRS  # noqa: E402
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
    assert cli.main(["main", *common]) == 0
    for arm in ARMS:
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
    assert sum(x.startswith("| raw") and x.split("|")[2].strip() in ("val", "ffpp_test", "celebdf")
               for x in s.splitlines()) == 12  # 4 arm x 3 dataset
    for a, b in PAIRS:
        c = json.loads((ck / f"compare_{a}__{b}.json").read_text(encoding="utf-8"))
        assert c["note"] == NOTE and "diff" in c["ffpp_test"] and f"auc_{b}" in c["ffpp_test"]
    assert len(list(ck.glob("compare_*__*.json"))) == 4
    assert "Tổng hợp main" in capsys.readouterr().out
    # chạy lại: mọi run đã có result.json -> bỏ qua, không lỗi; summary.md ghi lại
    assert cli.main(["main", *common]) == 0
    assert (ck / "summary.md").exists()
