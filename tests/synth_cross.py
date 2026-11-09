"""Cây thư mục Celeb-DF v2 TỔNG HỢP cho test — MINH HỌA, không phải dữ liệu thật.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from pathlib import Path

from conftest import write_mp4
from synth import render_face_frame

N = 40


def _face_video(path: Path, tid: str, method=None) -> None:
    write_mp4(path, [render_face_frame(tid, f, method) for f in range(N)])


def _noface_video(path: Path) -> None:
    from synth import _BG

    write_mp4(path, [_BG.copy() for _ in range(N)])


def make_synth_cdf2(root: Path) -> Path:
    """Vài video mỗi thư mục, 1 video không mặt (YouTube-real/00002), 1 file hỏng (YouTube-real/00003).
    List_of_testing_videos.txt theo quy ước `1` = real."""
    root = Path(root)
    rels = []
    for i, rel in enumerate(["Celeb-real/id0_0000.mp4", "Celeb-real/id1_0000.mp4"]):
        _face_video(root / rel, f"{i:03d}"); rels.append((1, rel))
    for i, rel in enumerate(["Celeb-synthesis/id0_id1_0000.mp4", "Celeb-synthesis/id1_id0_0000.mp4"]):
        _face_video(root / rel, f"{i:03d}", "DF"); rels.append((0, rel))
    _face_video(root / "YouTube-real/00001.mp4", "002"); rels.append((1, "YouTube-real/00001.mp4"))
    _noface_video(root / "YouTube-real/00002.mp4"); rels.append((1, "YouTube-real/00002.mp4"))
    (root / "YouTube-real/00003.mp4").write_bytes(b"not a video, MINH HOA")
    rels.append((1, "YouTube-real/00003.mp4"))
    (root / "List_of_testing_videos.txt").write_text("".join(f"{l} {r}\n" for l, r in rels), encoding="utf-8")
    return root
