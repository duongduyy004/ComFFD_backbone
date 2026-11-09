"""Registry head. Mỗi head: `__init__(dim)`, `forward(trunk_out) -> dict`, `loss(pred, targets)`.

Trạng thái: chưa chạy thử.
"""
HEADS: dict[str, type] = {}


def register_head(name):
    def deco(cls):
        HEADS[name] = cls
        return cls
    return deco


def build_head(name, dim):
    if name not in HEADS:
        raise KeyError(f"head không có trong registry: {name!r}; có: {sorted(HEADS)}")
    return HEADS[name](dim)


from . import binary  # noqa: E402,F401  (đăng ký "binary")
