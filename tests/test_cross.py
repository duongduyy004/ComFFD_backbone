"""Test chuẩn bị cross-dataset (Celeb-DF v2 TỔNG HỢP, MINH HỌA). Trạng thái: chưa chạy thử."""
import pandas as pd
import pytest
from synth import FakeDetector
from synth_cross import make_synth_cdf2

from conftest import make_cfg
from ffd.data.cross import CrossListError, prep_cross, read_cdf2


def ff_cfg(tmp_path):
    return make_cfg(tmp_path, tmp_path / "ffout")


def touch_videos(root, rels):
    for r in rels:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def write_list(root, text):
    (root / "List_of_testing_videos.txt").write_text(text, encoding="utf-8")


def test_prep_cdf2(tmp_path):
    root = make_synth_cdf2(tmp_path / "Celeb_DFv2")
    cfg = ff_cfg(tmp_path)
    out = prep_cross("cdf2", root, tmp_path / "out", cfg, FakeDetector())
    man = pd.read_parquet(out)
    assert list(man.columns) == ["dataset", "video_id", "frame_idx", "label", "keep", "drop_reason"]
    assert set(man.label) == {0, 1} and set(man.dataset) == {"cdf2"}
    assert (man.drop_reason == "no_face").sum() == 1 and (man.drop_reason == "decode_error").sum() == 1
    assert man[man.keep].groupby("video_id").size().max() <= 32
    row = man[man.drop_reason == "no_face"].iloc[0]
    assert row.frame_idx == -1 and not row.keep and row.video_id == "YouTube-real__00002"
    kept = man[man.keep]
    assert kept.video_id.nunique() == 5
    for r in kept.itertuples():
        assert (tmp_path / "out" / "cdf2" / "crops" / r.video_id / f"{r.frame_idx:05d}.png").exists()
    import cv2
    img = cv2.imread(str(tmp_path / "out" / "cdf2" / "crops" / kept.iloc[0].video_id / f"{kept.iloc[0].frame_idx:05d}.png"))
    assert img.shape[:2] == (cfg.crop_size, cfg.crop_size)
    assert man[man.video_id.str.startswith("Celeb-synthesis")].label.eq(1).all()


def test_cdf2_labels_from_folder(tmp_path):
    touch_videos(tmp_path, ["YouTube-real/00003.mp4", "Celeb-real/id0_0000.mp4", "Celeb-synthesis/id0_id1_0000.mp4"])
    write_list(tmp_path, "1 YouTube-real/00003.mp4\n1 Celeb-real/id0_0000.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n")
    items = read_cdf2(tmp_path)
    assert [i.label for i in items] == [0, 0, 1]
    assert items[2].video_id == "Celeb-synthesis__id0_id1_0000"
    assert items[2].path == tmp_path / "Celeb-synthesis/id0_id1_0000.mp4"


def test_cdf2_convention_mismatch_raises(tmp_path):
    touch_videos(tmp_path, ["Celeb-synthesis/id0_id1_0000.mp4"])
    write_list(tmp_path, "1 Celeb-synthesis/id0_id1_0000.mp4\n")
    with pytest.raises(CrossListError, match="id0_id1_0000"):
        read_cdf2(tmp_path)


def test_cdf2_missing_file_raises(tmp_path):
    write_list(tmp_path, "1 YouTube-real/99999.mp4\n")
    with pytest.raises(CrossListError):
        read_cdf2(tmp_path)


def test_cdf2_unknown_folder_raises(tmp_path):
    touch_videos(tmp_path, ["Other/a.mp4"])
    write_list(tmp_path, "1 Other/a.mp4\n")
    with pytest.raises(CrossListError):
        read_cdf2(tmp_path)
