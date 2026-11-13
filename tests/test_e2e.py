"""E2E 3 arm trên CPU: train -> evaluate -> report. Trạng thái: chưa chạy thử.
Dữ liệu FF++ TỔNG HỢP, CLIP tí hon ngẫu nhiên (MINH HỌA); số AUC không có ý nghĩa."""
import pytest

pytest.importorskip("torch")

from ffd.evaluate import evaluate_run  # noqa: E402
from ffd.report import build_report  # noqa: E402
from ffd.train import train  # noqa: E402
from test_train import mkcfg  # noqa: E402


def test_e2e_three_arms(ffpp_synth, tiny, tmp_path):
    for arm in ("raw", "raw+rep"):
        cfg = mkcfg(mode=arm, runs_root=tmp_path, total_steps=4, eval_every=2, ckpt_every=2, ffpp_root=ffpp_synth)
        d = train(cfg, vision_config=tiny["vision_config"], allow_cpu=True)
        evaluate_run(d, ["ffpp_test"], vision_config=tiny["vision_config"])
        assert (d / "scores_ffpp_test.parquet").exists()
    out = build_report(tmp_path)
    main = (out / "main.md").read_text(encoding="utf-8")
    head, _, _ = main.partition("## Chú thích")
    assert "## CẢNH BÁO: thiếu kết quả" in head
    for arm in ("raw", "raw+rep"):
        for s_ in (1, 2):
            assert f"- {arm}, seed {s_}, ffpp_test" in head
