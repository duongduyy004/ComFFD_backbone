"""Test lọc/cân bằng ảnh aug (hàm thuần). Dữ liệu tổng hợp (MINH HỌA). Trạng thái: chưa chạy thử trên GPU."""
import pandas as pd
import pytest

from ffd.data.augfilter import FILTER_MIN_R, LOC_LAMBDA, arm_constants, balance_aug, filter_aug
from ffd.data.ffpp import aug_frames


def make_aug(rep_r, rep_m, seam_r, seam_m):
    rows = []
    for kind, rs, ms in (("rep", rep_r, rep_m), ("seam", seam_r, seam_m)):
        for i, (r, m) in enumerate(zip(rs, ms)):
            rows.append(dict(kind=kind, label=int(kind == "rep"), tid=f"{i:03d}", frame_idx=i, method="DF",
                             r=r, mask_frac_final=m))
    return pd.DataFrame(rows)


def make_aug_counts(seam, rep):
    rows = [dict(kind="seam", label=0, tid=f"s{i:03d}", frame_idx=i, method="DF", r=1.0, mask_frac_final=0.05)
            for i in range(seam)]
    rows += [dict(kind="rep", label=1, tid=f"p{i:03d}", frame_idx=i, method=("DF", "F2F")[i % 2], r=1.0,
                  mask_frac_final=0.05) for i in range(rep)]
    return pd.DataFrame(rows)


def test_filter_rule():
    aug = make_aug(rep_r=[0.25, 0.75, 1.0, 1.0], rep_m=[0.05, 0.01, 0.03, 0.05],
                   seam_r=[0.5, 1.0], seam_m=[0.05, 0.04])
    kept, st = filter_aug(aug)  # tau_m = median(0.05, 0.01, 0.03, 0.05) = 0.04
    assert FILTER_MIN_R == 0.75
    assert st["tau_m"] == pytest.approx(0.04)
    assert sorted(kept[kept.kind == "rep"].mask_frac_final) == [0.05]  # r>=.75 và m>=.04
    assert list(kept[kept.kind == "seam"].r) == [1.0]
    assert (st["n_rep_before"], st["n_seam_before"], st["n_rep_kept"], st["n_seam_kept"]) == (4, 2, 1, 1)


def test_balance_integer_rule():  # R=96, F=480, S=57, P=195 -> giữ 39 seam, bỏ 18
    kept, st = balance_aug(make_aug_counts(seam=57, rep=195), n_real=96, n_fake=480)
    assert st["n_seam_after"] == 39 and st["n_rep_after"] == 195
    assert (kept.kind == "seam").sum() == 39 and (kept.kind == "rep").sum() == 195
    assert st["p_real"] == pytest.approx(57 / (96 + 57)) and st["p_fake"] == pytest.approx(195 / (480 + 195))
    assert st["n_real"] == 96 and st["n_fake"] == 480
    assert st["p_real_after"] == pytest.approx(39 / 135) and st["p_fake_after"] == pytest.approx(195 / 675)


def test_balance_rep_side():  # p_fake > p_real: R=10, F=10, S=5, P=30 -> giữ rep S*F//R = 5
    kept, st = balance_aug(make_aug_counts(seam=5, rep=30), n_real=10, n_fake=10)
    assert st["n_rep_after"] == 5 and st["n_seam_after"] == 5


def test_balance_deterministic_and_idempotent():
    aug = make_aug_counts(seam=57, rep=195)
    k1, s1 = balance_aug(aug, 96, 480)
    k2, s2 = balance_aug(aug, 96, 480)
    pd.testing.assert_frame_equal(k1, k2)
    assert s1 == s2
    k3, s3 = balance_aug(k1, 96, 480)  # gọi lại trên kết quả -> không bỏ thêm
    pd.testing.assert_frame_equal(k1, k3)
    assert s3["n_seam_after"] == s1["n_seam_after"] and s3["n_rep_after"] == s1["n_rep_after"]


def test_balance_no_change_when_balanced():
    aug = make_aug_counts(seam=10, rep=10)
    kept, st = balance_aug(aug, n_real=10, n_fake=10)
    assert len(kept) == 20 and st["n_seam_after"] == 10 and st["n_rep_after"] == 10


def test_filter_empty_raises():
    aug = make_aug(rep_r=[0.25, 0.25], rep_m=[0.05, 0.05], seam_r=[1.0], seam_m=[0.05])
    with pytest.raises(ValueError):
        filter_aug(aug)
    aug = make_aug(rep_r=[1.0, 1.0], rep_m=[0.05, 0.05], seam_r=[0.25], seam_m=[0.05])
    with pytest.raises(ValueError):  # seam rỗng
        filter_aug(aug)


def test_arm_constants():
    assert arm_constants("raw") == {} and arm_constants("raw_aug") == {}
    assert arm_constants("raw_aug_filt") == {"filter_min_r": 0.75, "filter_mask": "median_rep"}
    assert arm_constants("raw_aug_filt_loc") == {"filter_min_r": 0.75, "filter_mask": "median_rep",
                                                 "loc_lambda": LOC_LAMBDA}
    assert LOC_LAMBDA == 1.0


def test_aug_frames_columns(data_root):
    df = aug_frames(data_root)
    for c in ("method", "r", "mask_frac_final", "mask_path"):
        assert c in df.columns
    assert all(p.exists() for p in df.mask_path) and all(p.is_absolute() for p in df.mask_path)


def test_balance_empty_side_raises():
    with pytest.raises(ValueError):  # R rất nhỏ -> seam giữ 0
        balance_aug(make_aug_counts(seam=5, rep=30), n_real=1, n_fake=100)
    with pytest.raises(ValueError):  # rep giữ 0
        balance_aug(make_aug_counts(seam=1, rep=30), n_real=100, n_fake=1)


def test_arm_constants_unknown_arm_raises():
    with pytest.raises(ValueError):
        arm_constants("raw_aug_filt_typo")


def test_filter_nan_raises():
    aug = make_aug(rep_r=[1.0, 1.0], rep_m=[0.05, float("nan")], seam_r=[1.0], seam_m=[0.05])
    with pytest.raises(ValueError, match="NaN"):
        filter_aug(aug)
    aug = make_aug(rep_r=[1.0, float("nan")], rep_m=[0.05, 0.05], seam_r=[1.0], seam_m=[0.05])
    with pytest.raises(ValueError, match="NaN"):
        filter_aug(aug)
