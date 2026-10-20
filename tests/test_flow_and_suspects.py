import math

import numpy as np

from foot_overlay.areas import Area
from foot_overlay.flowlines import centerline_points, flow_candidates, snapped
from foot_overlay.suspects import find_suspects

W, H = 640, 480


def corridor_xy(n=15):
    """右下がりの斜めの通路を歩く人たち（流れの向きは約 20°）。"""
    out = {}
    for k in range(n):
        t = np.linspace(0, 1, 60)
        off = (k % 5) * 8
        out[k + 1] = np.stack([60 + 500 * t, 150 + 180 * t + off], axis=1)
    return out


AREA = Area(polygon=[[50, 130], [570, 320], [570, 380], [50, 190]])


def test_centerline_runs_along_the_corridor():
    pts = centerline_points(AREA, W, H)
    assert len(pts) > 200  # 約1pxおき
    assert all(AREA.contains(x, y) for x, y in pts[::20])


def test_candidates_are_perpendicular_to_flow_and_span_the_area():
    cands = flow_candidates(AREA, corridor_xy(), W, H, n_max=5000)
    assert 0 < len(cands) <= 5000
    flow = math.degrees(math.atan2(180, 500))
    for x1, y1, x2, y2 in cands[::25]:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        diff = abs((ang - flow) % 180 - 90)
        assert diff <= 21  # 流れに直角 ±20°
        # 縁から縁まで: 両端がエリアの縁の近くにある
        for x, y in ((x1, y1), (x2, y2)):
            d = abs(__import__("cv2").pointPolygonTest(AREA.points().astype(np.float32), (float(x), float(y)), True))
            assert d < 2.0


def test_snap_to_vertical_within_3_degrees():
    rect = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=float)
    line = np.array([200 - 4, 100, 200 + 4, 300], dtype=float)  # 垂直から約2.3°
    s = snapped(line, rect, W, H)
    assert s is not None and abs(s[0] - s[2]) < 1e-6
    assert snapped(np.array([150, 100, 250, 300], dtype=float), rect, W, H) is None  # 26°は対象外


def test_find_suspects_detects_each_kind():
    tracks = corridor_xy(12)
    jump = corridor_xy(1)[1].copy()
    jump[30:] += [0, 150]  # 途中で150px瞬間移動
    tracks[100] = jump
    rng = np.random.default_rng(0)
    zig = np.cumsum(rng.choice([-12, 12], size=(60, 2)), axis=0) + [300, 250]  # 前に進まない
    tracks[101] = zig
    tracks[102] = np.stack([np.linspace(50, 590, 60), np.full(60, 40.0)], axis=1)  # 誰も歩かない上の方
    found = find_suspects(tracks, W, H, fps=10)
    assert "jump" in found.get(100, [])
    assert "zigzag" in found.get(101, [])
    assert "offpath" in found.get(102, [])
    assert not any(t in found for t in range(1, 13))  # 普通に歩いた人は対象外


def test_round_trip_is_not_zigzag():
    there = np.stack([np.linspace(60, 560, 50), np.full(50, 200.0)], axis=1)
    tracks = {1: np.vstack([there, there[::-1]])}
    assert find_suspects(tracks, W, H, fps=10) == {}
