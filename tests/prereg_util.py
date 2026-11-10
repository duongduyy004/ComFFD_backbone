"""Helper test: prereg đủ khóa. Trạng thái: chưa chạy thử."""
from pathlib import Path

import yaml

PREREG_EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "prereg_aug.example.yaml"


def full_prereg(**kw) -> dict:
    """Prereg đủ khóa (lấy từ configs/prereg_aug.example.yaml, giá trị MINH HỌA), ghi đè bằng kw."""
    d = yaml.safe_load(PREREG_EXAMPLE.read_text(encoding="utf-8"))
    d.update(kw)
    return d


def write_prereg(path, **kw) -> Path:
    path = Path(path)
    path.write_text(yaml.safe_dump(full_prereg(**kw)), encoding="utf-8")
    return path
