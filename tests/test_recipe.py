"""Test bộ bốc recipe (thuần numpy). Trạng thái: chưa chạy thử."""
import numpy as np

from ffd.data.recipe import Recipe, sample_item, sample_recipe

MET = ("DF", "F2F", "FS", "NT", "FSh")
REG = ("eyes", "nose", "mouth", "skin", "boundary")


def test_rho_and_method_balance():
    rng = np.random.default_rng(0)
    vr = {m: REG for m in MET}
    rs = [sample_recipe(rng, "raw+comp", 0.5, MET, vr) for _ in range(20000)]
    assert abs(np.mean([r.source == "comp" for r in rs]) - 0.5) < 0.02
    assert max(abs(np.mean([r.method == m for r in rs]) - 0.2) for m in MET) < 0.02
    comp = [r for r in rs if r.source == "comp"]
    assert max(abs(np.mean([r.region == g for r in comp]) - 0.2) for g in REG) < 0.02


def test_raw_arm_never_composites():
    rs = [sample_recipe(np.random.default_rng(i), "raw", 0.5, MET, {m: REG for m in MET}) for i in range(500)]
    assert {r.source for r in rs} == {"raw"}
    assert all(r.region is None and r.borrowed is None for r in rs)


def test_method_without_valid_cells_is_skipped():
    vr = {"DF": ("mouth",), "NT": ()}
    rs = [sample_recipe(np.random.default_rng(i), "raw+comp", 1.0, ("DF", "NT"), vr) for i in range(200)]
    assert {r.method for r in rs if r} == {"DF"}  # method rỗng vùng hợp lệ -> None (caller đổi frame)
    assert any(r is None for r in rs)
    assert sample_recipe(np.random.default_rng(0), "raw+comp", 1.0, ("NT",), {"NT": ()}) is None


def test_sb_borrows_valid_method():
    for i in range(50):
        r = sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, ("DF", "F2F"), {"DF": ("eyes",), "F2F": ("mouth",)})
        assert r.source == "sb" and (r.region, r.borrowed) in {("eyes", "DF"), ("mouth", "F2F")}


def test_sb_borrowed_equals_method():
    vr = {"DF": ("eyes",), "F2F": ("eyes",), "FS": ("eyes", "nose")}
    rs = [sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, tuple(vr), vr) for i in range(500)]
    assert all(r.borrowed == r.method and r.region in vr[r.method] for r in rs)


def test_comp_and_sb_share_method_region_marginals():
    vr = {"DF": ("eyes", "nose"), "F2F": ("mouth",)}
    a = [sample_recipe(np.random.default_rng(i), "raw+comp", 1.0, tuple(vr), vr) for i in range(4000)]
    b = [sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, tuple(vr), vr) for i in range(4000)]
    assert [(r.method, r.region) for r in a] == [(r.method, r.region) for r in b]


def test_recipe_is_namedtuple():
    assert Recipe._fields == ("source", "method", "region", "borrowed")


def _population(n=300, frac_empty=0.3, seed=0):
    """Quần thể frame MINH HỌA: ~frac_empty frame có valid_regions rỗng cho một số method, phần còn lại đủ."""
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vr = {m: REG for m in MET}
        if g.random() < frac_empty:
            for m in MET:
                if g.random() < 0.6:
                    vr[m] = ()
        out.append((MET, vr))
    return out


def _run(arm, rho, n_items=20000, pop=None):
    pop = pop or _population()
    rng = np.random.default_rng(1)
    out = []
    for i in range(n_items):
        k, rec = sample_item(rng, arm, rho, len(pop), lambda j: pop[j], i % len(pop))
        assert rec.method in pop[k][0]
        if rec.source != "raw":
            assert rec.region in pop[k][1][rec.method]
        out.append(rec)
    return out


def test_effective_rho_equals_nominal_with_empty_frames():
    rs = _run("raw+comp", 0.5)
    assert abs(np.mean([r.source == "comp" for r in rs]) - 0.5) < 0.02


def test_method_uniform_with_empty_valid_regions_on_some_frames():
    rs = [r for r in _run("raw+comp", 1.0) if r.source == "comp"]
    assert max(abs(np.mean([r.method == m for r in rs]) - 0.2) for m in MET) < 0.02


def test_comp_and_sb_same_method_region_marginals_through_resampling():
    pop = _population()
    a, b = _run("raw+comp", 1.0, pop=pop), _run("raw+sb", 1.0, pop=pop)
    assert [(r.method, r.region) for r in a] == [(r.method, r.region) for r in b]


def test_sample_item_raises_after_max_tries():
    import pytest

    with pytest.raises(RuntimeError):
        sample_item(np.random.default_rng(0), "raw+comp", 1.0, 3, lambda j: (("DF",), {"DF": ()}), 0, 50)
