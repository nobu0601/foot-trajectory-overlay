"""エリアごとに、最も多く人を数えられるカウント線の候補を提案する。

手順:
  1. 人の流れの中心線に沿って、流れに直角（±20°）でエリアの縁から縁まで引いた線を
     最大 n_candidates 本（既定 20,000 本）作る（flowlines.py）
  2. 誤判定らしい追跡線（suspects.py）が通る線は除く
  3. まず全候補を高速な方法で評価し、上位の候補を実際のカウント処理と同じ方法で数え直す
  4. 上位の候補の信頼度（evaluate.py）を出し、評価値 = 人数 × 信頼度÷100 の高い順に、
     互いに離れた位置にある線を top_k 本（既定3本）選ぶ
同じエリア・同じ軌跡なら毎回同じ結果になる（乱数は使っていない）。
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .areas import Area
from .counting import CountLine, count_lines
from .drawing import smooth
from .tracking import TrackPoint

DEFAULT_CANDIDATES = 20000
RESCORE = 400  # 実際のカウント処理で数え直す候補の数
EVALUATE = 30  # 信頼度を出す候補の数（1本あたり百ミリ秒ほどかかるので絞る）
REPEAT_SEC = 1.0  # この秒数以内に同じ人が線をまたぎ直したら二重カウントとみなす
CHUNK = 256  # 一度にまとめて評価する線の本数（メモリ使用量の上限を決める）


@dataclass
class Suggestion:
    area_index: int
    rank: int
    line: CountLine
    people: int  # 線をまたいだ人数
    duplicates: int  # 二重カウントの回数（同じ人が短時間にまたぎ直した回数）
    area_people: int  # エリアを通った人数
    passes: int = 0  # 通過回数の合計（往復した人は2回）
    reliability: float = 0.0  # 信頼度 0〜100
    grade: str = ""  # A / B / C
    lower: int = 0  # 推定範囲の下限
    upper: int = 0  # 推定範囲の上限
    reasons: list[str] = field(default_factory=list)  # 信頼度が低い理由

    @property
    def final_score(self) -> float:
        """順位を決める評価値 = 人数 × 信頼度÷100"""
        return self.people * self.reliability / 100

    @property
    def score(self) -> int:
        return self.people - self.duplicates

    @property
    def coverage(self) -> float:
        """エリアを通った人のうち、この線で数えられた割合。"""
        return self.people / self.area_people if self.area_people else 0.0


ProgressCallback = Callable[[int, int], None]


def clip_line_to_polygon(point: np.ndarray, direction: np.ndarray, polygon: np.ndarray) -> tuple | None:
    """polygon 内の point を通る direction 向きの直線を、polygon の縁で切った線分。"""
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    e = b - a
    denom = direction[0] * e[:, 1] - direction[1] * e[:, 0]
    ok = np.abs(denom) > 1e-12
    if not ok.any():
        return None
    w = a - point
    t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0])[ok] / denom[ok]
    s = (w[:, 0] * direction[1] - w[:, 1] * direction[0])[ok] / denom[ok]
    t = t[(s >= 0) & (s <= 1)]
    neg, pos = t[t < 0], t[t > 0]
    if not len(neg) or not len(pos):
        return None
    return point + direction * neg.max(), point + direction * pos.min()


def generate_candidates(area: Area, width: int, height: int, n: int, seed: int = 0) -> np.ndarray:
    """エリアを横断する候補線を n 本作る。戻り値は (n, 4) = x1, y1, x2, y2。"""
    rng = np.random.default_rng(seed)
    poly = area.points()
    ys, xs = np.nonzero(area.mask(width, height))
    if len(xs) == 0:
        return np.zeros((0, 4))
    lines = []
    attempts = 0
    while len(lines) < n and attempts < n * 3:
        batch = min(n - len(lines), 4096)
        attempts += batch
        idx = rng.integers(0, len(xs), batch)
        pts = np.stack([xs[idx], ys[idx]], axis=1) + rng.random((batch, 2))
        angles = rng.random(batch) * np.pi
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        for p, d in zip(pts, dirs):
            seg = clip_line_to_polygon(p, d, poly)
            if seg is not None and np.hypot(*(seg[1] - seg[0])) >= 5:
                lines.append(np.concatenate(seg))
    return np.array(lines, dtype=np.float64).reshape(-1, 4)


def decimate(xy: np.ndarray, step: float) -> np.ndarray:
    """足元の点を step px 以上の間隔に間引く（細かな揺れで交差を数えすぎないため）。"""
    if len(xy) < 2:
        return xy
    kept = [xy[0]]
    for p in xy[1:]:
        if np.hypot(*(p - kept[-1])) >= step:
            kept.append(p)
    if not np.array_equal(kept[-1], xy[-1]):
        kept.append(xy[-1])
    return np.array(kept)


def _segments_near(tracks_xy: dict[int, np.ndarray], bbox: tuple, step: float):
    """エリア付近を通る軌跡の線分を、人ごとにまとめて並べる。"""
    x0, y0, x1, y1 = bbox
    starts, P, Q = [], [], []
    for xy in tracks_xy.values():
        pts = decimate(xy, step)
        if len(pts) < 2:
            continue
        p, q = pts[:-1], pts[1:]
        near = (
            (np.maximum(p[:, 0], q[:, 0]) >= x0)
            & (np.minimum(p[:, 0], q[:, 0]) <= x1)
            & (np.maximum(p[:, 1], q[:, 1]) >= y0)
            & (np.minimum(p[:, 1], q[:, 1]) <= y1)
        )
        if not near.any():
            continue
        starts.append(sum(len(s) for s in P))
        P.append(p[near])
        Q.append(q[near])
    if not P:
        return None
    return np.concatenate(P), np.concatenate(Q), np.array(starts)


def score_lines(lines: np.ndarray, P: np.ndarray, Q: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各線について (またいだ人数, またいだ回数の合計) を返す。

    線の束が狭い範囲にまとまっているときは、その範囲の近くを通る線分だけを比べて速くする。
    """
    person = np.zeros(len(P), dtype=np.int64)  # 各線分が何人目の軌跡か
    person[starts[1:]] = 1
    person = np.cumsum(person)
    lo = np.minimum(lines[:, 0:2], lines[:, 2:4]).min(axis=0)
    hi = np.maximum(lines[:, 0:2], lines[:, 2:4]).max(axis=0)
    near = (
        (np.maximum(P[:, 0], Q[:, 0]) >= lo[0])
        & (np.minimum(P[:, 0], Q[:, 0]) <= hi[0])
        & (np.maximum(P[:, 1], Q[:, 1]) >= lo[1])
        & (np.minimum(P[:, 1], Q[:, 1]) <= hi[1])
    )
    if not near.any():
        return np.zeros(len(lines), dtype=np.int64), np.zeros(len(lines), dtype=np.int64)
    P, Q, person = P[near].astype(np.float32), Q[near].astype(np.float32), person[near]
    lines = lines.astype(np.float32)
    A, B = lines[:, None, 0:2], lines[:, None, 2:4]  # (L, 1, 2)
    ab = B - A

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    d1 = cross(ab, P[None] - A)
    d2 = cross(ab, Q[None] - A)
    pq = (Q - P)[None]
    d3 = cross(pq, A - P[None])
    d4 = cross(pq, B - P[None])
    hit = (((d1 * d2) < 0) & ((d3 * d4) < 0)).astype(np.float32)  # (L, M)
    # 線分ごとの交差を人ごとに合計する（近くを通った人だけの対応表との行列の掛け算）
    ids, local = np.unique(person, return_inverse=True)
    onehot = np.zeros((len(person), len(ids)), dtype=np.float32)
    onehot[np.arange(len(person)), local] = 1
    per_person = hit @ onehot  # (L, 人数)
    return (per_person > 0).sum(axis=1).astype(np.int64), per_person.sum(axis=1).round().astype(np.int64)


def exact_scores(
    lines: np.ndarray,
    tracks: dict[int, list[TrackPoint]],
    smooth_window: int,
    margin: float,
    repeat_frames: int,
):
    """実際のカウント処理（count_lines）と同じ方法で、(またいだ人数, 二重カウント数, 通過回数) を数える。

    二重カウント = 同じ人が repeat_frames フレーム以内に同じ線をまたぎ直した回数。
    線の上でうろうろしたり足元の位置が揺れたりして、1回の通過が複数回に数えられたもの。
    行って戻ってきた人のように、時間を空けて往復した場合は正しい2回の通過なので減点しない。
    """
    line_objs = [CountLine(*map(float, ln)) for ln in lines]
    _, crossings = count_lines(tracks, line_objs, smooth_window, margin)
    people = np.zeros(len(lines), dtype=np.int64)
    repeats = np.zeros(len(lines), dtype=np.int64)
    passes = np.zeros(len(lines), dtype=np.int64)
    last_frame: dict[tuple[int, int], int] = {}
    for c in sorted(crossings, key=lambda c: (c.line_index, c.track_id, c.frame)):
        key = (c.line_index, c.track_id)
        passes[c.line_index] += 1
        if key not in last_frame:
            people[c.line_index] += 1
        elif c.frame - last_frame[key] <= repeat_frames:
            repeats[c.line_index] += 1
        last_frame[key] = c.frame
    return people, repeats, passes


def _spread(order: np.ndarray, mids: np.ndarray, min_dist: float, limit: int) -> list[int]:
    """order の順に、すでに選んだ線と中点が min_dist 以上離れた線だけを limit 本まで選ぶ。

    足りなければ、離れていない線も順に足して limit 本にする（候補が少ないエリアのため）。
    """
    chosen: list[int] = []
    kept = np.zeros((0, 2))
    for k in order:
        if len(kept) == 0 or np.min(np.hypot(*(kept - mids[k]).T)) >= min_dist:
            chosen.append(int(k))
            kept = np.vstack([kept, mids[k]])
            if len(chosen) >= limit:
                return chosen
    picked = set(chosen)
    for k in order:
        if len(chosen) >= limit:
            break
        if int(k) not in picked:
            chosen.append(int(k))
    return chosen


def suggest_lines(
    area: Area,
    area_index: int,
    tracks: dict[int, list[TrackPoint]],
    width: int,
    height: int,
    area_people: int,
    smooth_window: int = 5,
    margin: float = 3.0,
    fps: float = 30.0,
    n_candidates: int = DEFAULT_CANDIDATES,
    top_k: int = 3,
    seed: int = 0,
    progress: ProgressCallback | None = None,
    suspects: set[int] | None = None,
    repeat_sec: float = REPEAT_SEC,
) -> list[Suggestion]:
    """1つのエリアについて、カウント線の候補を最大 top_k 本返す。

    1. 人の流れに直角（±20°）で、エリアの縁から縁まで引いた線を最大 n_candidates 本作る
    2. 誤判定らしい追跡線（suspects）が通る線は除く
    3. 全候補を高速な方法でふるいにかけ、上位 RESCORE 本を実際のカウント処理で数え直す
    4. 上位 EVALUATE 本の信頼度を出し、「人数 × 信頼度÷100」の高い順に位置が離れた3本を選ぶ
    """
    from .evaluate import evaluate_line
    from .flowlines import flow_candidates, snapped

    suspects = suspects or set()
    tracks_xy = {tid: smooth(pts, smooth_window) for tid, pts in tracks.items()}
    normal_xy = {tid: xy for tid, xy in tracks_xy.items() if tid not in suspects}
    candidates = flow_candidates(area, normal_xy, width, height, n_candidates)
    if len(candidates) == 0:
        return []
    poly = area.points()
    pad = 10
    bbox = (poly[:, 0].min() - pad, poly[:, 1].min() - pad, poly[:, 0].max() + pad, poly[:, 1].max() + pad)
    step = max(2.0, 0.004 * max(width, height))
    segs = _segments_near(normal_xy, bbox, step)
    if segs is None:
        return []
    P, Q, starts = segs
    suspect_segs = _segments_near({t: tracks_xy[t] for t in suspects if t in tracks_xy}, bbox, step)

    people = np.zeros(len(candidates), dtype=np.int64)
    total = np.zeros(len(candidates), dtype=np.int64)
    suspect_hits = np.zeros(len(candidates), dtype=np.int64)
    for i in range(0, len(candidates), CHUNK):
        chunk = candidates[i : i + CHUNK]
        people[i : i + CHUNK], total[i : i + CHUNK] = score_lines(chunk, P, Q, starts)
        if suspect_segs is not None:
            suspect_hits[i : i + CHUNK] = score_lines(chunk, *suspect_segs)[0]
        if progress is not None:
            progress(min(i + CHUNK, len(candidates)), len(candidates))

    # 誤判定らしい線が通る候補は使わない（全部に通ってしまう場合だけ、通る本数が最少のものを残す）
    allowed = suspect_hits == suspect_hits.min()
    idx_allowed = np.flatnonzero(allowed)
    # 1段目: 高速評価の上位だけ残す（揺れによる見かけの交差を含むので、人数を重視して選ぶ）
    #   人数が同じ線は大量にあるので、同じあたりの線ばかりにならないよう位置を散らして残す
    separation = max(10.0, 0.2 * np.sqrt(area.polygon_area()))
    pre_order = idx_allowed[np.lexsort(((total - people)[idx_allowed], -people[idx_allowed]))]
    all_mids = (candidates[:, 0:2] + candidates[:, 2:4]) / 2
    pool = _spread(pre_order, all_mids, separation / 8, RESCORE)
    candidates = candidates[pool]
    # 2段目: 実際のカウント処理と同じ方法で数え直す
    normal_tracks = {t: tracks[t] for t in normal_xy}
    repeat = max(1, round(fps * repeat_sec))
    people, duplicates, passes = exact_scores(candidates, normal_tracks, smooth_window, margin, repeat)

    # 3段目: 上位の信頼度を出し、「人数 × 信頼度」で順位を付ける。
    # 同じ立ち位置で傾きだけ違う線ばかりにならないよう、位置を少し散らして EVALUATE 本選ぶ
    mids = (candidates[:, 0:2] + candidates[:, 2:4]) / 2
    prelim = _spread(np.lexsort((duplicates, -(people - duplicates))), mids, separation / 4, EVALUATE)
    evals = {}
    for k in prelim:
        if people[k] > 0:
            evals[int(k)] = evaluate_line(
                CountLine(*candidates[k]), tracks, tracks_xy, width, height, fps, margin,
                suspects=suspects, area_polygon=poly, repeat_sec=repeat_sec,
            )
    if not evals:
        return []
    keys = list(evals)
    final = np.array([people[k] * evals[k].reliability / 100 for k in keys])
    lengths = np.array([np.hypot(*(candidates[k, 2:4] - candidates[k, 0:2])) for k in keys])
    order = [keys[i] for i in np.lexsort((lengths, [duplicates[k] for k in keys], -final))]

    chosen: list[int] = []
    for k in order:
        if all(np.hypot(*(mids[k] - mids[j])) >= separation for j in chosen):
            chosen.append(k)
            if len(chosen) >= top_k:
                break

    results = []
    for rank, k in enumerate(chosen, start=1):
        line, ev = candidates[k], evals[k]
        n_people, n_dup, n_pass = int(people[k]), int(duplicates[k]), int(passes[k])
        # 水平・垂直に近い線は、評価が下がらなければ水平・垂直にそろえる（見やすさのため）
        alt = snapped(line, poly, width, height)
        if alt is not None:
            p2, d2, s2 = exact_scores(alt[None, :], normal_tracks, smooth_window, margin, repeat)
            ev2 = evaluate_line(
                CountLine(*alt), tracks, tracks_xy, width, height, fps, margin, suspects=suspects, area_polygon=poly,
                repeat_sec=repeat_sec,
            )
            if int(p2[0]) * ev2.reliability >= n_people * ev.reliability:
                line, ev = alt, ev2
                n_people, n_dup, n_pass = int(p2[0]), int(d2[0]), int(s2[0])
        results.append(
            Suggestion(
                area_index=area_index,
                rank=rank,
                line=CountLine(*(round(float(v)) for v in line)),
                people=n_people,
                duplicates=n_dup,
                area_people=area_people,
                passes=n_pass,
                reliability=round(ev.reliability, 1),
                grade=ev.grade,
                lower=ev.lower,
                upper=ev.upper,
                reasons=ev.reasons,
            )
        )
    return results


def save_suggestions_csv(path: Path, suggestions: list[Suggestion], areas: list[Area]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "area", "kind", "rank", "x1", "y1", "x2", "y2", "people", "passes", "duplicates",
                "reliability", "grade", "estimate_low", "estimate_high", "score", "area_people", "coverage", "reasons",
            ]
        )
        for s in suggestions:
            kind = areas[s.area_index].kind_label if s.area_index < len(areas) else ""
            ln = s.line
            w.writerow(
                [
                    f"E{s.area_index + 1}", kind, s.rank, round(ln.x1), round(ln.y1), round(ln.x2), round(ln.y2),
                    s.people, s.passes, s.duplicates, f"{s.reliability:.0f}", s.grade, s.lower, s.upper,
                    f"{s.final_score:.1f}", s.area_people, f"{s.coverage:.3f}", "・".join(s.reasons),
                ]
            )
