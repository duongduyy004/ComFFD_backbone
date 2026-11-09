"""Test ffd.stats (không cần torch). Trạng thái: chưa chạy thử."""
import numpy as np
import pandas as pd
import pytest

from ffd.stats import auc, bootstrap_ci, delong_paired, paired_frames, video_scores


def make_grouped(n_videos, frames, seed):
    rng = np.random.default_rng(seed)
    vy = rng.integers(0, 2, n_videos)
    voff = rng.normal(0, 1.0, n_videos)            # hiệu ứng theo video => frame tương quan
    y = np.repeat(vy, frames)
    g = np.repeat(np.arange(n_videos), frames)
    s = 0.5 * y + np.repeat(voff, frames) + rng.normal(0, 0.3, n_videos * frames)
    return y, s, g


def slow_delong(y, s1, s2):
    """Tham chiếu O(n^2): hạt nhân Mann-Whitney psi(x,y)=1,.5,0 viết tách biệt."""
    y = np.asarray(y)
    out = []
    for s in (s1, s2):
        s = np.asarray(s, float)
        P, N = s[y == 1], s[y == 0]
        psi = (P[:, None] > N[None, :]) + 0.5 * (P[:, None] == N[None, :])
        out.append(psi)
    m, n = out[0].shape
    a = [p.mean() for p in out]
    v10 = [p.mean(1) for p in out]
    v01 = [p.mean(0) for p in out]
    c10 = np.cov(np.stack(v10), ddof=1)
    c01 = np.cov(np.stack(v01), ddof=1)
    S = c10 / m + c01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = a[0] - a[1]
    return d, d / np.sqrt(var)


Y8 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
S1 = np.array([.1, .2, .3, .6, .4, .7, .8, .9])
S2 = np.array([.2, .1, .5, .3, .6, .4, .9, .7])
# S3 là bộ chấm thứ ba, tính tay (xem report): AUC1=.9375, AUC3=.875
S3 = np.array([.3, .1, .2, .5, .4, .35, .9, .6])


def test_delong_known_case():
    d, z, p = delong_paired(Y8, S1, S2)
    assert d == pytest.approx(auc(Y8, S1) - auc(Y8, S2)) and 0 <= p <= 1
    assert delong_paired(Y8, S1, S1)[2] == pytest.approx(1.0)


def test_delong_hand_computed():
    # d=.0625; Var(d)=.015625/4+.015625/4=.0078125; z=.0625/sqrt(.0078125)=1/sqrt(2); p=2*sf(z)
    d, z, p = delong_paired(Y8, S1, S3)
    assert d == pytest.approx(0.0625)
    assert z == pytest.approx(0.70710678, abs=1e-6)
    assert p == pytest.approx(0.4795, abs=1e-4)
    d2, z2, _ = delong_paired(Y8, S3, S1)
    assert d2 == pytest.approx(-0.0625) and z2 == pytest.approx(-0.70710678, abs=1e-6)


def test_delong_matches_slow_reference_random():
    rng = np.random.default_rng(1)
    for k in range(5):
        n = 40 + 10 * k
        y = rng.integers(0, 2, n)
        base = rng.normal(size=n) + y
        s1 = np.round(base + rng.normal(0, 1, n), 1)      # làm tròn => có ties
        s2 = np.round(base + rng.normal(0, 1.5, n), 1)
        d, z, _ = delong_paired(y, s1, s2)
        d_ref, z_ref = slow_delong(y, s1, s2)
        assert d == pytest.approx(d_ref, abs=1e-12)
        assert z == pytest.approx(z_ref, rel=1e-9)


def test_delong_one_class_raises():
    with pytest.raises(ValueError):
        delong_paired(np.zeros(5), np.arange(5.0), np.arange(5.0))


def test_bootstrap_resamples_groups():
    y, s, g = make_grouped(n_videos=40, frames=8, seed=0)
    lo_f, hi_f = bootstrap_ci(y, s, np.arange(len(y)))[1:]
    lo_v, hi_v = bootstrap_ci(y, s, g)[1:]
    assert (hi_v - lo_v) > (hi_f - lo_f)


def test_bootstrap_point_and_determinism_and_one_class_resample():
    y, s, g = make_grouped(n_videos=6, frames=2, seed=3)   # ít video => hay rút một lớp
    a = bootstrap_ci(y, s, g, n=200, seed=5)
    b = bootstrap_ci(y, s, g, n=200, seed=5)
    assert a == b and a[0] == pytest.approx(auc(y, s)) and a[1] <= a[2]


def test_paired_reports_dropped():
    a = pd.DataFrame({"video_id": ["v1", "v2", "v3"]})
    b = pd.DataFrame({"video_id": ["v1", "v3"]})
    _, _, dropped = paired_frames(a, b)
    assert dropped == 1


def test_paired_aligned_and_symmetric_drop():
    a = pd.DataFrame({"video_id": ["v3", "v1", "v2"], "score": [3, 1, 2]})
    b = pd.DataFrame({"video_id": ["v1", "v4", "v3"], "score": [10, 40, 30]})
    sa, sb, dropped = paired_frames(a, b)
    assert dropped == 2                                   # v2 chỉ ở a, v4 chỉ ở b
    assert list(sa.video_id) == list(sb.video_id) == ["v1", "v3"]
    assert list(sa.score) == [1, 3] and list(sb.score) == [10, 30]


def test_video_scores_mean_logit():
    df = pd.DataFrame({"video_id": ["a", "a", "b"], "label": [1, 1, 0], "logit": [1.0, 3.0, -2.0]})
    out = video_scores(df).sort_values("video_id").reset_index(drop=True)
    assert list(out.columns) == ["video_id", "label", "score"]
    assert list(out.score) == [2.0, -2.0] and list(out.label) == [1, 0]


def test_paired_duplicate_keys_raise():
    a = pd.DataFrame({"video_id": ["v1", "v1", "v2"]})
    b = pd.DataFrame({"video_id": ["v1", "v2"]})
    with pytest.raises(ValueError, match="'v1'"):
        paired_frames(a, b)
    with pytest.raises(ValueError, match="bảng b"):
        paired_frames(b, pd.DataFrame({"video_id": ["v2", "v2"]}))


def test_delong_too_few_per_class_raises():
    y = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError):
        delong_paired(y, np.array([.1, .2, .3, .4]), np.array([.2, .1, .3, .5]))
