import csv

import numpy as np

from foot_overlay.drawing import color_for_id, draw_overlay, filter_short_tracks, smooth
from foot_overlay.export import save_image, save_tracks_csv
from foot_overlay.tracking import TrackPoint, foot_point


def make_track(points):
    return [TrackPoint(i, x, y) for i, (x, y) in enumerate(points)]


def test_foot_point_is_bottom_center():
    assert foot_point(10, 20, 30, 100) == (20, 100)


def test_filter_short_tracks():
    tracks = {1: make_track([(0, 0)] * 3), 2: make_track([(0, 0)] * 10)}
    assert list(filter_short_tracks(tracks, 5)) == [2]


def test_smooth_keeps_length_and_endpoints_of_straight_line():
    pts = make_track([(i * 10, 50) for i in range(20)])
    xy = smooth(pts, 5)
    assert xy.shape == (20, 2)
    assert np.allclose(xy[:, 1], 50)
    assert np.all(np.diff(xy[:, 0]) >= 0)


def test_smooth_handles_short_tracks():
    assert smooth(make_track([(1, 2), (3, 4)]), 5).shape == (2, 2)


def test_colors_differ_between_neighbor_ids():
    assert color_for_id(1) != color_for_id(2)


def test_draw_overlay_keeps_size_and_draws_something():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    tracks = {
        1: make_track([(20 + i * 10, 200 - i * 5) for i in range(25)]),
        2: make_track([(300, 230)] * 10),  # 動かない人（終点は四角で描かれる）
    }
    out = draw_overlay(frame, tracks, grid=50)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert np.array_equal(frame, np.full((240, 320, 3), 128, dtype=np.uint8))  # 元画像は変更しない


def test_save_outputs_with_japanese_path(tmp_path):
    out_dir = tmp_path / "出力"
    out_dir.mkdir()
    save_image(out_dir / "画像.png", np.zeros((10, 10, 3), dtype=np.uint8))
    assert (out_dir / "画像.png").stat().st_size > 0

    save_tracks_csv(out_dir / "tracks.csv", {3: make_track([(1.25, 2.5)])}, fps=10)
    with (out_dir / "tracks.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows == [["track_id", "frame", "time_sec", "foot_x", "foot_y"], ["3", "0", "0.000", "1.2", "2.5"]]
