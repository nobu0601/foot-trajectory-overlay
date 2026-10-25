from pathlib import Path

import numpy as np

from foot_overlay.batch import collect_items, write_summary, BatchResult
from foot_overlay.session import Session, write_result
from foot_overlay.tracking import TrackingResult, TrackPoint


def test_collect_items_from_folders_and_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for p in ("a/cam.mp4", "b/cam.mp4", "top.MOV", "memo.txt"):
        (tmp_path / p).write_bytes(b"x")
    items = collect_items([tmp_path, tmp_path / "top.MOV"])  # 同じ動画を2回指定しても1回だけ
    assert [i.name for i in items] == ["a_cam", "b_cam", "top"]

    other = tmp_path / "other"
    other.mkdir()
    (other / "top.mp4").write_bytes(b"x")
    names = [i.name for i in collect_items([tmp_path / "top.MOV", other / "top.mp4"])]
    assert names == ["top", "top(2)"]  # 出力名が重ならない


def test_write_summary(tmp_path):
    item = collect_items([_video(tmp_path)])[0]
    write_summary(tmp_path / "s.csv", [BatchResult(item, "完了", people=3, frames=10, seconds=1.2)])
    text = (tmp_path / "s.csv").read_text(encoding="utf-8-sig")
    assert "完了" in text and "v.png" in text


def _video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"x")
    return p


def test_time_filter_changes_display_only(tmp_path):
    frame = np.full((200, 300, 3), 100, dtype=np.uint8)
    tracks = {
        1: [TrackPoint(i, 20 + i * 10, 100) for i in range(26)],  # 0〜2.5秒
        2: [TrackPoint(100 + i, 280 - i * 10, 150) for i in range(26)],  # 10〜12.5秒
    }
    out = tmp_path / "r"
    write_result(out, tmp_path / "v.mp4", TrackingResult(frame, 10.0, 126, tracks), tracks, smooth=5, min_points=10)
    s = Session(out)
    assert round(s.duration, 1) == 12.6 and not s.is_time_filtered()
    full = s.overlay().copy()
    s.time_start, s.time_end = 9.0, 13.0
    assert s.is_time_filtered() and list(s.visible_tracks()) == [2]
    assert not np.array_equal(full, s.overlay())
    # 保存用の画像は常に全員
    assert np.array_equal(full, s.overlay(all_tracks=True))
    # 集計は絞り込みに関係なく全員
    from foot_overlay.counting import CountLine

    s.add_line(CountLine(150, 0, 150, 199))
    assert s.counts[0].total == 2


def test_find_result_dirs_and_all_counts(tmp_path):
    import csv as _csv

    from foot_overlay.batch import find_result_dirs, write_all_counts
    from foot_overlay.counting import CountLine

    frame = np.full((200, 300, 3), 100, dtype=np.uint8)
    tracks = {1: [TrackPoint(i, 20 + i * 10, 100) for i in range(26)]}
    dirs = []
    for name in ("a_overlay", "sub/b_overlay"):
        out = tmp_path / name
        write_result(out, tmp_path / f"{Path(name).name[:-8]}.mp4", TrackingResult(frame, 10.0, 26, tracks), tracks, 5, 10)
        dirs.append(out)
    s = Session(dirs[0])
    s.set_lines([CountLine(150, 0, 150, 199, "入口")])
    s.save()
    found = find_result_dirs(tmp_path)
    assert found == sorted(dirs, key=lambda p: str(p).lower())
    n = write_all_counts(tmp_path / "all_counts.csv", found)
    rows = list(_csv.reader((tmp_path / "all_counts.csv").open(encoding="utf-8-sig")))
    assert n == 2 and rows[1][2] == "入口" and rows[1][5] == "1"
    assert rows[2][2] == "（カウント線なし）"


def test_scale_lines_keeps_name_and_direction():
    from foot_overlay.counting import CountLine, scale_lines

    out = scale_lines([CountLine(10, 20, 10, 80, "入口")], 100, 100, 200, 50)
    assert (out[0].x1, out[0].y1, out[0].y2, out[0].name) == (20, 10, 40, "入口")
