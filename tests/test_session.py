import json

import numpy as np

from foot_overlay.counting import CountLine
from foot_overlay.session import Session, write_result
from foot_overlay.tracking import TrackingResult, TrackPoint


def make_result_dir(tmp_path):
    frame = np.full((200, 300, 3), 100, dtype=np.uint8)
    tracks = {
        1: [TrackPoint(i, 280 - i * 10, 100) for i in range(26)],  # 右→左に歩く
        2: [TrackPoint(i, 20 + i * 10, 150) for i in range(26)],  # 左→右に歩く
    }
    result = TrackingResult(first_frame=frame, fps=10.0, total_frames=26, tracks=tracks)
    out = tmp_path / "動画_overlay"
    write_result(out, tmp_path / "動画.mp4", result, tracks, smooth=5, min_points=10)
    return out


def test_session_counts_and_saves(tmp_path):
    out = make_result_dir(tmp_path)
    s = Session(out)
    assert s.fps == 10.0
    assert s.add_line(CountLine(150, 0, 150, 199))
    assert not s.add_line(CountLine(10, 10, 11, 11))  # 短すぎる線は無視
    assert [(c.a_to_b, c.b_to_a) for c in s.counts] == [(1, 1)]

    s.line_width = 7
    s.save()
    for name in ("overlay.png", "count_result.png", "counts.csv", "crossings.csv", "lines.json"):
        assert (out / name).is_file()
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["view"]["line_width"] == 7

    # 開き直すと線と表示設定が復元される
    s2 = Session(out)
    assert len(s2.lines) == 1 and s2.line_width == 7
    assert s2.counts[0].total == 2


def test_line_width_changes_overlay(tmp_path):
    s = Session(make_result_dir(tmp_path))
    s.line_width = 1
    thin = s.overlay().copy()
    s.line_width = 10
    thick = s.overlay()
    assert not np.array_equal(thin, thick)
    # 太いほうが線の画素が多い（背景から変化した画素数で比較）
    base = s.first_frame.astype(int)
    assert np.count_nonzero(np.abs(thick.astype(int) - base).sum(axis=2) > 60) > np.count_nonzero(
        np.abs(thin.astype(int) - base).sum(axis=2) > 60
    )


def test_undo_remove_clear(tmp_path):
    s = Session(make_result_dir(tmp_path))
    s.set_lines([CountLine(100, 0, 100, 199), CountLine(200, 0, 200, 199), CountLine(250, 0, 250, 199)])
    s.remove_line(0)
    assert [ln.x1 for ln in s.lines] == [200, 250]
    s.undo()
    assert [ln.x1 for ln in s.lines] == [200]
    assert len(s.counts) == 1
    s.clear()
    assert s.lines == [] and s.counts == []


def test_reanalysis_clears_old_count_files(tmp_path):
    out = make_result_dir(tmp_path)
    s = Session(out)
    s.add_line(CountLine(150, 0, 150, 199))
    s.save()
    make_result_dir(tmp_path)  # 同じ動画を解析し直す
    assert not (out / "lines.json").exists()
    assert Session(out).lines == []
