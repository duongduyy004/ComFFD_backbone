"""Chuẩn bị bộ cross-dataset (hiện chỉ Celeb-DF v2) với đúng cách bốc frame, lọc mặt và căn 336x336 của FF++.

Để AUC cross-dataset không bị nhiễu bởi khác biệt tiền xử lý: dùng chung Config của compsynth_ff
(n_candidates, n_keep, max_yaw_deg, min_face_px, second_face_ratio, crop_size), chung choose_primary / yaw / template.
Nhãn lấy theo THƯ MỤC; số `l` trong List_of_testing_videos.txt chỉ để đối chiếu (quy ước 1 = real chưa kiểm chứng).
Lưu ý trùng lặp: vòng lọc theo frame mô phỏng lại extract.extract_target (không có hàm dùng lại được);
nếu extract.py đổi logic lọc thì phải đổi cả ở đây.
Hợp đồng manifest: mỗi video có >= 1 dòng. Video không có frame nào qua lọc <=> đúng một dòng
frame_idx == -1 và drop_reason == "no_usable_frame" (gồm cả không mặt, yaw, mặt nhỏ, mặt phụ). Lỗi giải mã / ngoại lệ
bất kỳ theo video <=> một dòng frame_idx == -1, drop_reason == "decode_error", cột `error` ghi tên lớp ngoại lệ.
Video có frame qua lọc có thể kèm dòng frame_idx >= 0 keep=False với lý do theo frame của choose_primary
(no_face/second_face/small_face/yaw) hoặc "not_selected". Video có ít hơn cfg.min_keep frame giữ lại được GIỮ CÓ CHỦ Ý
(tập test, không bỏ im lặng), khác với FF++ extract.
Nếu MỌI video đều decode_error thì prep_cross raise RuntimeError (không ghi manifest 'thành công'). Crop dở dang của video lỗi bị xóa.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from compsynth_ff.align import choose_primary, estimate_yaw_deg, similarity_to_template, warp
from compsynth_ff.io_utils import write_parquet_atomic, write_png
from compsynth_ff.video import count_frames, keep_evenly, read_frames, select_candidates

LIST_NAME = "List_of_testing_videos.txt"
_FOLDER_LABEL = {"Celeb-synthesis": 1, "Celeb-real": 0, "YouTube-real": 0}
COLUMNS = ["dataset", "video_id", "frame_idx", "label", "keep", "drop_reason", "error"]


class TestItem(NamedTuple):
    __test__ = False  # không để pytest coi là lớp test
    video_id: str
    path: Path
    label: int  # 1 = fake


class CrossListError(Exception):
    pass


def read_cdf2(root) -> list:
    root = Path(root)
    lst = root / LIST_NAME
    if not lst.is_file():
        raise CrossListError(f"Không thấy {lst}")
    items = []
    for n, raw in enumerate(lst.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or parts[0] not in ("0", "1"):
            raise CrossListError(f"{lst.name} dòng {n} sai định dạng: {raw!r}")
        l, rel = int(parts[0]), parts[1].strip().replace("\\", "/")
        folder = rel.split("/", 1)[0]
        if folder not in _FOLDER_LABEL:
            raise CrossListError(f"{lst.name} dòng {n}: thư mục lạ {folder!r} ({rel})")
        label = _FOLDER_LABEL[folder]
        if l != 1 - label:
            raise CrossListError(
                f"{lst.name} dòng {n} ({rel}): l={l} lệch quy ước l = 1 - label (label theo thư mục = {label}); "
                "quy ước chưa kiểm chứng, không tự đảo")
        path = root / rel
        if not path.is_file():
            raise CrossListError(f"{lst.name} dòng {n}: file không có trên đĩa: {path}")
        items.append(TestItem(video_id=str(Path(rel).with_suffix("")).replace("\\", "/").replace("/", "__"),
                              path=path, label=label))
    return items


READERS = {"cdf2": read_cdf2}


def _video_rows(name, item, cfg, detector, n_frames, crops_dir) -> list:
    base = dict(dataset=name, video_id=item.video_id, label=item.label, error="")
    vdir = crops_dir / item.video_id
    shutil.rmtree(vdir, ignore_errors=True)  # bỏ crop cũ của lần chạy trước
    try:
        return _video_rows_inner(base, item, cfg, detector, n_frames, crops_dir)
    except Exception as e:  # lỗi một video không được làm sập cả lượt chạy
        shutil.rmtree(vdir, ignore_errors=True)  # đĩa phải khớp manifest: không để crop dở dang
        return [dict(base, frame_idx=-1, keep=False, drop_reason="decode_error", error=type(e).__name__)]


def _video_rows_inner(base, item, cfg, detector, n_frames, crops_dir) -> list:
    n = count_frames(item.path)
    cands = select_candidates(n, cfg.n_candidates)
    frames = read_frames(item.path, cands)
    rows, ok = [], {}
    for f in cands:
        dets = detector.detect(frames[f])
        main, why = choose_primary(dets, cfg)
        if main is not None and abs(estimate_yaw_deg(main.kps)) > cfg.max_yaw_deg:
            why = "yaw"
        if why == "ok":
            ok[f] = main
        rows.append((f, why))
    keep = keep_evenly([f for f, w in rows if w == "ok"], n_frames)
    if not keep:
        return [dict(base, frame_idx=-1, keep=False, drop_reason="no_usable_frame")]
    keep_set = set(keep)
    out = []
    for f, why in rows:
        if f in keep_set:
            write_png(crops_dir / item.video_id / f"{f:05d}.png",
                      warp(frames[f], similarity_to_template(ok[f].kps), cfg.crop_size))
            out.append(dict(base, frame_idx=int(f), keep=True, drop_reason=""))
        else:
            out.append(dict(base, frame_idx=int(f), keep=False, drop_reason=why if why != "ok" else "not_selected"))
    return out


def prep_cross(name, root, out, cfg, detector, n_frames: int = 32) -> Path:
    if name not in READERS:
        raise CrossListError(f"Bộ cross không hỗ trợ: {name!r} (có: {sorted(READERS)})")
    items = READERS[name](root)
    d = Path(out) / name
    rows = []
    for it in items:
        rows += _video_rows(name, it, cfg, detector, n_frames, d / "crops")
    errs = [r["error"] for r in rows if r["drop_reason"] == "decode_error"]
    if items and len(errs) == len(items):
        raise RuntimeError(f"Mọi video ({len(items)}) đều decode_error, nghi lỗi hệ thống; "
                           f"ngoại lệ phổ biến nhất: {Counter(errs).most_common(1)[0][0]}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["frame_idx"] = df.frame_idx.astype("int64")
    df["label"] = df.label.astype("int64")
    df["keep"] = df.keep.astype(bool)
    path = d / "manifest.parquet"
    write_parquet_atomic(df, path)
    return path
