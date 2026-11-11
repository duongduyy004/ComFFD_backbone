"""Test bộ bốc nguồn/method (thuần numpy). Trạng thái: chưa chạy thử. Spec can_bang_duong_noi §4."""
import numpy as np
import pytest

from ffd.data.recipe import ARMS, draw_real_source, draw_source, sample_item

MET = ("DF", "F2F", "FS", "NT", "FSh")


def test_arms():
    assert ARMS == ("raw", "raw+rep")
    with pytest.raises(ValueError):
        draw_source(np.random.default_rng(0), "raw+comp", 0.5)
    with pytest.raises(ValueError):
        draw_real_source(np.random.default_rng(0), "raw+sb", 0.5)


def test_rho():
    rng = np.random.default_rng(0)
    assert abs(np.mean([draw_source(rng, "raw+rep", 0.5) == "rep" for _ in range(20000)]) - 0.5) < 0.02
    assert abs(np.mean([draw_real_source(rng, "raw+rep", 0.5) == "seam" for _ in range(20000)]) - 0.5) < 0.02
    assert {draw_source(rng, "raw", 0.9) for _ in range(500)} == {"raw"}
    assert {draw_real_source(rng, "raw", 0.9) for _ in range(500)} == {"raw"}


def _population(n=300, frac_empty=0.3, seed=0):
    """Quần thể frame MINH HỌA: ~frac_empty frame thiếu một số method."""
    g = np.random.default_rng(seed)
    return [MET if g.random() >= frac_empty else tuple(m for m in MET if g.random() >= 0.6) for _ in range(n)]


def test_method_uniform_with_missing_methods():
    pop = _population()
    rng = np.random.default_rng(1)
    got = []
    for i in range(20000):
        k, m = sample_item(rng, len(pop), lambda j: pop[j], i % len(pop))
        assert m in pop[k]
        got.append(m)
    assert max(abs(np.mean([x == m for x in got]) - 0.2) for m in MET) < 0.02


def test_sample_item_raises_after_max_tries():
    with pytest.raises(RuntimeError):
        sample_item(np.random.default_rng(0), 3, lambda j: (), 0, 50)
