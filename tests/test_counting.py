import numpy as np
import pytest

from foot_overlay.counting import (
    CountLine,
    count_lines,
    find_crossings,
    parse_line_arg,
    segments_intersect,
)
from foot_overlay.tracking import TrackPoint


def track(points):
    return [TrackPoint(i, float(x), float(y)) for i, (x, y) in enumerate(points)]


def walk(x0, x1, y, n=30):
    return track([(x0 + (x1 - x0) * i / (n - 1), y) for i in range(n)])


# 縦の線 x=100（上→下に引く）。法線は左向きなので、右側が A、左側が B
VERTICAL = CountLine(100, 0, 100, 200)


def test_segments_intersect():
    a, b = np.array([0, 0]), np.array([10, 10])
    assert segments_intersect(np.array([0, 10]), np.array([10, 0]), a, b)
    assert not segments_intersect(np.array([20, 0]), np.array([30, 10]), a, b)


def test_direction_is_counted():
    counts, crossings = count_lines({1: walk(200, 0, 100), 2: walk(0, 200, 120)}, [VERTICAL])
    assert (counts[0].a_to_b, counts[0].b_to_a, counts[0].total) == (1, 1, 2)
    by_track = {c.track_id: c.direction for c in crossings}
    assert by_track == {1: "A->B", 2: "B->A"}
    assert all(abs(c.x - 100) < 1e-6 for c in crossings)


def test_passing_outside_line_end_is_not_counted():
    counts, _ = count_lines({1: walk(200, 0, 300)}, [VERTICAL])  # 線の下端(y=200)より下を通る
    assert counts[0].total == 0


def test_jitter_on_the_line_is_not_double_counted():
    # 線の上で ±2px 揺れてから渡りきる
    pts = [(150, 100), (120, 100), (101, 100), (99, 100), (101, 100), (99, 100), (101, 100), (80, 100), (50, 100)]
    xy = np.array(pts, dtype=float)
    crossings = find_crossings(VERTICAL, 0, 1, track(pts), xy, margin=3)
    assert len(crossings) == 1


def test_walk_in_and_back_out_counts_both_directions():
    pts = [(150 - 10 * i, 100) for i in range(10)] + [(60 + 10 * i, 100) for i in range(10)]
    counts, _ = count_lines({1: track(pts)}, [VERTICAL], smooth_window=1)
    assert (counts[0].a_to_b, counts[0].b_to_a) == (1, 1)


def test_multiple_lines_are_counted_separately():
    lines = [VERTICAL, CountLine(300, 0, 300, 200)]
    counts, _ = count_lines({1: walk(0, 200, 100)}, lines)
    assert [c.total for c in counts] == [1, 0]


def test_parse_line_arg():
    assert parse_line_arg("1, 2,3,4") == CountLine(1, 2, 3, 4)
    with pytest.raises(ValueError):
        parse_line_arg("1,2,3")
