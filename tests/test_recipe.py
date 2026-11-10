"""Test bộ bốc recipe (thuần numpy). Trạng thái: chưa chạy thử.

Spec augmentation 11/11 §5: recipe chỉ còn (source, method); vùng và mask do compsynth_ff.TrainCore.sample_mask bốc.
"""
import numpy as np
import pytest

from ffd.data.recipe import Recipe, sample_item, sample_recipe

MET = ("DF", "F2F", "FS", "NT", "FSh")


def test_recipe_is_namedtuple():
    assert Recipe._fields == ("source", "method")


def test_rho_and_method_balance():
    rng = np.random.default_rng(0)
    rs = [sample_recipe(rng, "raw+comp", 0.5, MET) for _ in range(20000)]
    assert abs(np.mean([r.source == "comp" for r in rs]) - 0.5) < 0.02
    assert max(abs(np.mean([r.method == m for r in rs]) - 0.2) for m in MET) < 0.02


def test_raw_arm_never_composites():
    rs = [sample_recipe(np.random.default_rng(i), "raw", 0.5, MET) for i in range(500)]
    assert {r.source for r in rs} == {"raw"}


def test_method_not_on_frame_is_none():
    assert sample_recipe(np.random.default_rng(0), "raw+comp", 1.0, ("DF",), method="NT") is None
    assert sample_recipe(np.random.default_rng(0), "raw+comp", 1.0, ()) is None


def test_comp_and_sb_share_method_sequence():
    a = [sample_recipe(np.random.default_rng(i), "raw+comp", 1.0, MET) for i in range(2000)]
    b = [sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, MET) for i in range(2000)]
    assert [r.method for r in a] == [r.method for r in b]
    assert {r.source for r in a} == {"comp"} and {r.source for r in b} == {"sb"}


def _population(n=300, frac_empty=0.3, seed=0):
    """Quần thể frame MINH HỌA: ~frac_empty frame thiếu một số method (lệch hàng/không có)."""
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ms = MET if g.random() >= frac_empty else tuple(m for m in MET if g.random() >= 0.6)
        out.append(ms)
    return out


def _run(arm, rho, n_items=20000, pop=None):
    pop = pop or _population()
    rng = np.random.default_rng(1)
    out = []
    for i in range(n_items):
        k, rec = sample_item(rng, arm, rho, len(pop), lambda j: pop[j], i % len(pop))
        assert rec.method in pop[k]
        out.append(rec)
    return out


def test_effective_rho_equals_nominal_with_missing_methods():
    rs = _run("raw+comp", 0.5)
    assert abs(np.mean([r.source == "comp" for r in rs]) - 0.5) < 0.02


def test_method_uniform_with_missing_methods_on_some_frames():
    rs = [r for r in _run("raw+comp", 1.0) if r.source == "comp"]
    assert max(abs(np.mean([r.method == m for r in rs]) - 0.2) for m in MET) < 0.02


def test_comp_and_sb_same_methods_through_resampling():
    pop = _population()
    a, b = _run("raw+comp", 1.0, pop=pop), _run("raw+sb", 1.0, pop=pop)
    assert [r.method for r in a] == [r.method for r in b]


def test_sample_item_raises_after_max_tries():
    with pytest.raises(RuntimeError):
        sample_item(np.random.default_rng(0), "raw+comp", 1.0, 3, lambda j: (), 0, 50)
