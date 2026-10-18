import numpy as np

from foot_overlay.areas import Area, detect_areas, people_in_area
from foot_overlay.counting import count_lines
from foot_overlay.suggest import clip_line_to_polygon, generate_candidates, suggest_lines
from foot_overlay.tracking import TrackPoint

W, H = 400, 300


def corridor_tracks(n=12):
    """横向きの通路（y=120〜180）を左右に歩く人たち。"""
    tracks = {}
    for k in range(n):
        y = 125 + (k * 5) % 50
        xs = np.linspace(20, 380, 60) if k % 2 == 0 else np.linspace(380, 20, 60)
        tracks[k + 1] = [TrackPoint(i, float(x), float(y + 2 * np.sin(i))) for i, x in enumerate(xs)]
    return tracks


def as_xy(tracks):
    return {tid: np.array([(p.x, p.y) for p in pts]) for tid, pts in tracks.items()}


def test_clip_line_to_polygon_spans_edge_to_edge():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    a, b = clip_line_to_polygon(np.array([50.0, 50.0]), np.array([1.0, 0.0]), square)
    assert np.allclose(sorted([a[0], b[0]]), [0, 100])
    assert np.allclose([a[1], b[1]], [50, 50])


def test_candidates_stay_inside_area_and_are_reproducible():
    area = Area(polygon=[[50, 100], [350, 100], [350, 200], [50, 200]])
    c1 = generate_candidates(area, W, H, 500, seed=1)
    c2 = generate_candidates(area, W, H, 500, seed=1)
    assert len(c1) == 500 and np.array_equal(c1, c2)
    assert c1[:, [0, 2]].min() >= 49.9 and c1[:, [0, 2]].max() <= 350.1
    assert c1[:, [1, 3]].min() >= 99.9 and c1[:, [1, 3]].max() <= 200.1


def test_detect_areas_finds_corridor():
    tracks = corridor_tracks()
    areas = detect_areas(as_xy(tracks), W, H)
    assert len(areas) == 1
    area = areas[0]
    assert area.contains(200, 150)
    assert not area.contains(200, 30)
    assert people_in_area(area, as_xy(tracks), W, H) == 12


def test_suggestion_crosses_the_corridor_and_counts_everyone():
    tracks = corridor_tracks()
    area = Area(polygon=[[10, 110], [390, 110], [390, 190], [10, 190]])
    suggestions = suggest_lines(area, 0, tracks, W, H, area_people=12, n_candidates=2000, margin=3)
    assert len(suggestions) == 3
    best = suggestions[0]
    assert best.people == 12 and best.duplicates == 0 and best.coverage == 1.0
    # 通路を横切る（ほぼ縦向きの）線になっている
    assert abs(best.line.y2 - best.line.y1) > abs(best.line.x2 - best.line.x1)
    # 提案どおりに線を引けば、カウント処理でも同じ人数になる
    counts, _ = count_lines(tracks, [best.line], 5, 3)
    assert counts[0].total == best.passes == 12
    # 3本は互いに離れた位置
    mids = [((s.line.x1 + s.line.x2) / 2, (s.line.y1 + s.line.y2) / 2) for s in suggestions]
    assert all(np.hypot(mids[i][0] - mids[j][0], mids[i][1] - mids[j][1]) >= 10 for i in range(3) for j in range(i))


def test_quick_back_and_forth_is_penalized_but_round_trip_is_not():
    area = Area(polygon=[[0, 0], [400, 0], [400, 300], [0, 300]])
    # 人1: x=200 の付近で行ったり来たり（数え間違いの元）
    wobble = [TrackPoint(i, 200 + (15 if i % 2 else -15), 150) for i in range(10)]
    # 人2: 右へ歩いて、しばらくして戻る（正しい往復）
    trip = [TrackPoint(i, x, 100) for i, x in enumerate(list(range(20, 381, 10)) + list(range(380, 19, -10)))]
    from foot_overlay.suggest import exact_scores

    line = np.array([[200, 0, 200, 299]], dtype=float)
    people, dup, passes = exact_scores(line, {1: wobble, 2: trip}, 1, 3, repeat_frames=5)
    assert people[0] == 2
    assert passes[0] >= 4
    # 人1の素早い往復だけが二重カウント。人2の往復（時間が空いている）は減点しない
    wobble_only = exact_scores(line, {1: wobble}, 1, 3, repeat_frames=5)
    trip_only = exact_scores(line, {2: trip}, 1, 3, repeat_frames=5)
    assert wobble_only[1][0] > 0
    assert trip_only[1][0] == 0 and trip_only[2][0] == 2
