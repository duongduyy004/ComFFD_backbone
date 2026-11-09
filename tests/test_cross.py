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
    assert list(man.columns) == ["dataset", "video_id", "frame_idx", "label", "keep", "drop_reason", "error"]
    assert set(man.label) == {0, 1} and set(man.dataset) == {"cdf2"}
    assert (man.drop_reason == "no_usable_frame").sum() == 1 and (man.drop_reason == "decode_error").sum() == 1
    assert man[man.keep].groupby("video_id").size().max() <= 32
    row = man[man.drop_reason == "no_usable_frame"].iloc[0]
    assert row.frame_idx == -1 and not row.keep and row.video_id == "YouTube-real__00002"
    kept = man[man.keep]
    # video không có frame dùng được: đúng một dòng; video giữ lại vẫn có thể kèm dòng frame >= 0 với lý do theo frame
    assert (man[man.drop_reason == "no_usable_frame"].frame_idx == -1).all()
    assert (man[man.video_id == "YouTube-real__00002"].shape[0]) == 1
    per_frame = man[(man.frame_idx >= 0) & ~man.keep]
    assert set(per_frame.video_id) <= set(kept.video_id) and set(per_frame.drop_reason) <= {"not_selected", "no_face", "second_face", "small_face", "yaw"}
    assert (per_frame.drop_reason == "not_selected").any()  # 40 frame -> 40 ứng viên, giữ 32
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


def test_prep_per_video_exception_does_not_crash(tmp_path, monkeypatch):
    import ffd.data.cross as cross
    root = make_synth_cdf2(tmp_path / "Celeb_DFv2")
    real = cross.read_frames

    def short(path, idx):
        fr = real(path, idx)
        return fr if "YouTube-real" in str(path) else {k: v for k, v in fr.items() if k != idx[-1]}

    monkeypatch.setattr(cross, "read_frames", short)
    man = pd.read_parquet(prep_cross("cdf2", root, tmp_path / "out", ff_cfg(tmp_path), FakeDetector()))
    de = man[man.drop_reason == "decode_error"]
    assert de.video_id.nunique() == 5 and (de.frame_idx == -1).all() and "KeyError" in set(de.error)
    assert man[man.keep].video_id.unique().tolist() == ["YouTube-real__00001"]


def test_prep_all_decode_error_raises(tmp_path, monkeypatch):
    import ffd.data.cross as cross
    root = make_synth_cdf2(tmp_path / "Celeb_DFv2")
    real = cross.read_frames
    monkeypatch.setattr(cross, "read_frames", lambda path, idx: {k: v for k, v in real(path, idx).items() if k != idx[-1]})
    with pytest.raises(RuntimeError, match="KeyError"):
        prep_cross("cdf2", root, tmp_path / "out", ff_cfg(tmp_path), FakeDetector())


def test_prep_exception_after_writes_cleans_crops(tmp_path, monkeypatch):
    import ffd.data.cross as cross
    root = make_synth_cdf2(tmp_path / "Celeb_DFv2")
    real = cross.similarity_to_template
    state = {"n": 0}

    def flaky(kps):
        state["n"] += 1
        if state["n"] == 3:
            raise ValueError("MINH HOA")
        return real(kps)

    monkeypatch.setattr(cross, "similarity_to_template", flaky)
    man = pd.read_parquet(prep_cross("cdf2", root, tmp_path / "out", ff_cfg(tmp_path), FakeDetector()))
    bad = man[(man.drop_reason == "decode_error") & (man.error == "ValueError")]
    assert len(bad) == 1
    assert not (tmp_path / "out" / "cdf2" / "crops" / bad.iloc[0].video_id).exists()
    assert not man[man.video_id == bad.iloc[0].video_id].keep.any()


def test_cdf2_duplicate_video_id_raises(tmp_path):
    touch_videos(tmp_path, ["Celeb-real/a.mp4", "Celeb-real/a.mov"])
    write_list(tmp_path, "1 Celeb-real/a.mp4\n1 Celeb-real/a.mp4\n")
    with pytest.raises(CrossListError, match="dòng 1 và dòng 2"):
        read_cdf2(tmp_path)
    write_list(tmp_path, "1 Celeb-real/a.mp4\n1 Celeb-real/a.mov\n")
    with pytest.raises(CrossListError, match="trùng video_id"):
        read_cdf2(tmp_path)
