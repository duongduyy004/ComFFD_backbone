"""Smoke test cho fixture data_root (dữ liệu tổng hợp, MINH HỌA). Trạng thái: chưa chạy thử."""


def test_data_root_builds(data_root):
    assert (data_root / "ffpp" / "manifest_raw.parquet").exists()
    assert (data_root / "celebdf" / "manifest.parquet").exists()
