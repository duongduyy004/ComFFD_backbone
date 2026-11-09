"""Test bộ bốc recipe (thuần numpy). Trạng thái: chưa chạy thử."""
import numpy as np

from ffd.data.recipe import Recipe, sample_recipe

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
    assert {r.method for r in rs} == {"DF"}
    assert sample_recipe(np.random.default_rng(0), "raw+comp", 1.0, ("NT",), {"NT": ()}) is None


def test_sb_borrows_valid_method():
    for i in range(50):
        r = sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, MET, {"DF": ("eyes",), "F2F": ("mouth",)})
        assert r.source == "sb" and (r.region, r.borrowed) in {("eyes", "DF"), ("mouth", "F2F")}


def test_sb_borrowed_uniform_among_valid():
    vr = {"DF": ("eyes",), "F2F": ("eyes",), "FS": ("eyes", "nose")}
    rs = [sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, tuple(vr), vr) for i in range(6000)]
    eyes = [r for r in rs if r.region == "eyes"]
    assert {r.borrowed for r in eyes} == {"DF", "F2F", "FS"}
    assert all(r.borrowed in {m for m in vr if r.region in vr[m]} for r in rs)
    assert max(abs(np.mean([r.borrowed == m for r in eyes]) - 1 / 3) for m in vr) < 0.03


def test_comp_and_sb_share_method_region_marginals():
    vr = {"DF": ("eyes", "nose"), "F2F": ("mouth",), "NT": ()}
    a = [sample_recipe(np.random.default_rng(i), "raw+comp", 1.0, tuple(vr), vr) for i in range(4000)]
    b = [sample_recipe(np.random.default_rng(i), "raw+sb", 1.0, tuple(vr), vr) for i in range(4000)]
    assert [(r.method, r.region) for r in a] == [(r.method, r.region) for r in b]


def test_recipe_is_namedtuple():
    assert Recipe._fields == ("source", "method", "region", "borrowed")
