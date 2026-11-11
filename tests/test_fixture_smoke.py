"""Smoke test cho fixture ffpp_synth (dữ liệu tổng hợp, MINH HỌA). Trạng thái: chưa chạy thử."""


def test_ffpp_synth_builds(ffpp_synth):
    assert (ffpp_synth.out_root / "ffpp_raw" / "manifest.parquet").exists()
    assert (ffpp_synth.out_root / "ffpp_aug" / "manifest.parquet").exists()
