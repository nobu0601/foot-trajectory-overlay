import json

import numpy as np

from foot_overlay.counting import CountLine, count_lines
from foot_overlay.flows import crossing_angle, find_flows, flow_candidates
from foot_overlay.session import Session, write_result
from foot_overlay.stitch import StitchSettings, fill_gaps, stitch_tracks
from foot_overlay.suspects import find_suspects
from foot_overlay.tracking import TrackingResult, TrackPoint

W, H, FPS = 800, 600, 10.0


def walk(f0, f1, p0, p1):
    """f0〜f1 フレームに p0 → p1 をまっすぐ歩く。"""
    n = f1 - f0
    return [TrackPoint(f0 + i, p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n) for i in range(n + 1)]


# ---------- 途切れた追跡線をつなぐ ----------


def test_occluded_person_is_stitched_and_counted_once():
    # 右へ歩く人が x=380〜420 の柱の陰で 1 秒見えなくなり、別のIDで現れる
    full = walk(0, 60, (100, 300), (700, 300))
    tracks = {1: [p for p in full if p.x < 380], 2: [p for p in full if p.x > 420]}
    line = CountLine(400, 200, 400, 400)
    counts, _ = count_lines(tracks, [line])
    assert counts[0].total == 0  # つながないと、線の手前と奥で切れて数え漏れる
    stitched, n = stitch_tracks(tracks, W, H, FPS)
    assert n == 1 and list(stitched) == [1]
    frames = [p.frame for p in stitched[1]]
    assert frames == list(range(61))  # 見えなかった区間は補われている
    counts, _ = count_lines(stitched, [line])
    assert counts[0].total == 1


def test_filled_gap_is_not_flagged_as_jump():
    full = walk(0, 60, (100, 300), (700, 300))
    tracks = {1: [p for p in full if p.x < 380], 2: [p for p in full if p.x > 420]}
    stitched, _ = stitch_tracks(tracks, W, H, FPS)
    xy = {t: np.array([[p.x, p.y] for p in pts]) for t, pts in stitched.items()}
    assert find_suspects(xy, W, H, FPS) == {}


def test_not_stitched_when_far_late_reversed_or_at_edge():
    a = walk(0, 20, (100, 300), (300, 300))  # 右へ歩いて x=300 で消える
    far = walk(25, 45, (600, 100), (700, 100))  # 遠くに現れた別人
    late = walk(60, 80, (330, 300), (500, 300))  # 4 秒後（既定は 2 秒まで）
    back = walk(23, 43, (330, 300), (130, 300))  # すぐ近くに現れたが逆向きに歩く人
    for other in (far, late, back):
        _, n = stitch_tracks({1: a, 2: other}, W, H, FPS)
        assert n == 0
    # 画面の端で消えた人（外へ出た）は、すぐ近くに人が現れてもつながない
    edge = walk(0, 20, (400, 300), (795, 300))
    near = walk(22, 40, (780, 290), (600, 290))
    assert stitch_tracks({1: edge, 2: near}, W, H, FPS)[1] == 0


def test_stitch_can_be_turned_off_and_gap_changed():
    full = walk(0, 60, (100, 300), (700, 300))
    tracks = {1: [p for p in full if p.x < 380], 2: [p for p in full if p.x > 420]}
    assert stitch_tracks(tracks, W, H, FPS, StitchSettings(on=False))[1] == 0
    assert stitch_tracks(tracks, W, H, FPS, StitchSettings(max_gap_sec=0.3))[1] == 0


def test_fill_gaps():
    pts = [TrackPoint(0, 0, 0), TrackPoint(4, 40, 0), TrackPoint(30, 100, 0)]
    out = fill_gaps(pts, max_gap=5)
    assert [p.frame for p in out] == [0, 1, 2, 3, 4, 30]  # 長すぎる切れ目は補わない
    assert out[2].x == 20


# ---------- 出入口と人流 ----------


def crossing_scene():
    """左の出入口 ↔ 右の出入口（横の通路）を 12 人、上 ↔ 下を 8 人が歩く。"""
    rng = np.random.default_rng(0)
    tracks = {}
    tid = 1
    for k in range(12):
        y = 300 + rng.uniform(-30, 30)
        a, b = ((30, y), (770, y + rng.uniform(-10, 10)))
        if k % 3 == 0:
            a, b = b, a  # 帰りの人
        tracks[tid] = walk(k * 15 + 10, k * 15 + 80, a, b)
        tid += 1
    for k in range(8):
        x = 400 + rng.uniform(-25, 25)
        tracks[tid] = walk(k * 20 + 5, k * 20 + 60, (x, 30), (x + rng.uniform(-10, 10), 570))
        tid += 1
    return tracks


def to_xy(tracks):
    return {t: np.array([[p.x, p.y] for p in pts]) for t, pts in tracks.items()}


def test_gateways_and_flows_are_found():
    xy = to_xy(crossing_scene())
    gateways, flows = find_flows(xy, W, H)
    assert len(gateways) == 4
    assert [f.people for f in flows] == [12, 8]
    horiz = flows[0]
    assert (horiz.forward, horiz.backward) == (8, 4) or (horiz.forward, horiz.backward) == (4, 8)
    # 代表経路は横の通路を通る
    path = np.array(horiz.path)
    assert np.all(np.abs(path[:, 1] - 300) < 40)


def test_flow_candidates_cross_the_flow_at_the_angle():
    xy = to_xy(crossing_scene())
    _, flows = find_flows(xy, W, H)
    for min_angle in (60, 85):
        lines = flow_candidates(flows[0], xy, W, H, FPS, min_angle)
        assert len(lines) > 0
        d = lines[:, 2:4] - lines[:, 0:2]
        d = d / np.hypot(*d.T)[:, None]
        # 横向きの流れ（1, 0）に対して min_angle 度以上
        angles = np.array([crossing_angle(u, np.array([1.0, 0.0])) for u in d])
        assert angles.min() >= min_angle - 1


def test_session_flow_suggestions_end_to_end(tmp_path):
    tracks = crossing_scene()
    frame = np.full((H, W, 3), 90, np.uint8)
    result = TrackingResult(first_frame=frame, fps=FPS, total_frames=260, tracks=tracks)
    out = tmp_path / "動画_overlay"
    write_result(out, tmp_path / "動画.mp4", result, tracks, smooth=5, min_points=10)
    s = Session(out)
    assert s.suggest_mode == "flow"
    s.detect_flows()
    assert len(s.flows) == 2
    s.suggestions = s.compute_suggestions()
    by_flow = {}
    for sg in s.suggestions:
        by_flow.setdefault(sg.flow_index, []).append(sg)
    assert set(by_flow) == {0, 1}
    best = by_flow[0][0]
    assert best.label == "F1-1" and best.people == 12 and best.area_people == 12
    s.save()
    assert (out / "flows.json").is_file() and (out / "suggestions.csv").is_file()
    s2 = Session(out)
    assert len(s2.flows) == 2 and s2.suggestions[0].flow_index == 0
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["suggest_mode"] == "flow"
