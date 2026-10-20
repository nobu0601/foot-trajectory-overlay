"""信頼度評価の「故障の注入テスト」: きれいな軌跡に数え間違いの原因を混ぜ、指標と信頼度が反応するか確かめる。"""

import numpy as np

from foot_overlay.counting import CountLine
from foot_overlay.drawing import smooth
from foot_overlay.evaluate import evaluate_line, perturbed_lines
from foot_overlay.tracking import TrackPoint

W, H, FPS = 640, 480, 10.0
MARGIN = 0.01 * max(W, H)
LINE = CountLine(320, 140, 320, 340)  # 通路（y=150〜330）を横切る縦線


def walkers(n=12, frames=80, gap=40):
    """左から右へ、時間をずらして1人ずつ歩く人たち（きれいな軌跡）。"""
    tracks = {}
    for k in range(n):
        y = 160 + (k * 13) % 160
        start = k * gap
        xs = np.linspace(40, 600, frames)
        tracks[k + 1] = [TrackPoint(start + i, float(x), float(y)) for i, x in enumerate(xs)]
    return tracks


def evaluate(tracks, line=LINE, suspects=None):
    xy = {t: smooth(p, 5) for t, p in tracks.items()}
    return evaluate_line(line, tracks, xy, W, H, FPS, MARGIN, suspects=suspects)


def test_clean_tracks_are_reliable_and_range_is_exact():
    ev = evaluate(walkers())
    assert ev.passes == 12 and ev.grade == "A" and ev.reliability >= 95
    assert (ev.lower, ev.upper) == (12, 12)
    assert ev.reasons == []


def test_breaks_near_the_line_raise_fragment_and_widen_range():
    tracks = walkers()
    next_id = 100
    for tid in (1, 2, 3, 4):  # 4人の追跡を線の手前で途切れさせ、続きを別IDにする
        pts = tracks[tid]
        cut = next(i for i, p in enumerate(pts) if p.x > 300)
        tracks[tid], tracks[next_id] = pts[:cut], pts[cut + 6 :]
        next_id += 1
    clean, broken = evaluate(walkers()), evaluate(tracks)
    assert broken.metrics["fragment"] > 0.1
    assert broken.reliability < clean.reliability
    assert broken.upper >= 12  # 推定範囲の上限は、混ぜる前の通過回数を含む
    assert "線の近くで追跡が途切れている" in broken.reasons


def test_jitter_on_the_line_raises_duplicates():
    tracks = walkers()
    for tid in (1, 2, 3):  # 線の上で、0.4秒ずつ前後に±20px ゆっくり揺れてから渡る
        pts = tracks[tid]
        i = next(i for i, p in enumerate(pts) if p.x > 320)
        f0 = pts[i].frame
        wobble = [TrackPoint(f0 + j, 320 + (20 if (j // 4) % 2 else -20), pts[i].y) for j in range(16)]
        tracks[tid] = pts[:i] + wobble + [TrackPoint(p.frame + 16, p.x, p.y) for p in pts[i:]]
    ev = evaluate(tracks)
    assert ev.metrics["duplicate"] > 0.05
    assert ev.lower <= 12 <= ev.upper
    assert ev.reliability < evaluate(walkers()).reliability


def test_short_line_raises_bypass():
    ev = evaluate(walkers(), CountLine(320, 150, 320, 230))  # 通路の半分しか横切らない線
    assert ev.metrics["bypass"] > 0.1
    assert "線の端の外をすり抜ける人がいる" in ev.reasons
    assert ev.upper >= 12


def test_line_along_the_flow_raises_shallow_angle():
    tracks = {}
    for k in range(10):  # 少し斜めに歩く人たちに、ほぼ平行な線を引く
        tracks[k + 1] = [TrackPoint(k * 40 + i, 40 + i * 7, 200 + i * 1.2 + k * 3) for i in range(80)]
    ev = evaluate(tracks, CountLine(100, 230, 560, 270))
    assert ev.metrics["shallow"] > 0.2


def test_standing_on_the_line_raises_dwell():
    tracks = walkers()
    for tid in (1, 2, 3, 4):  # 線のすぐそばで3秒立ち止まる
        pts = tracks[tid]
        i = next(i for i, p in enumerate(pts) if p.x > 318)
        f0 = pts[i].frame
        stay = [TrackPoint(f0 + j, 321, pts[i].y) for j in range(30)]
        tracks[tid] = pts[:i] + stay + [TrackPoint(p.frame + 30, p.x, p.y) for p in pts[i:]]
    ev = evaluate(tracks)
    assert ev.metrics["dwell"] > 0.1


def test_suspect_tracks_crossing_lower_reliability():
    ev = evaluate(walkers(), suspects={1, 2})
    assert ev.metrics["suspect"] > 0.05
    assert "誤判定らしい追跡線が通っている" in ev.reasons


def test_perturbed_lines_are_ten_small_variations():
    lines = perturbed_lines(LINE, W, H)
    assert len(lines) == 10
    mids = [((ln.x1 + ln.x2) / 2, (ln.y1 + ln.y2) / 2) for ln in lines]
    assert all(abs(mx - 320) <= 0.02 * W + 1 for mx, _ in mids)
