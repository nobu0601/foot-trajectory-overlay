"""エリアごとに、最も多く人を数えられるカウント線の候補を提案する。

手順:
  1. エリア内のランダムな点を通り、ランダムな向きで、エリアの縁から縁まで横断する線を
     n_candidates 本（既定 20,000 本）作る
  2. 全ての線について、またいだ人数とまたいだ回数を数える
  3. 評価値 = またいだ人数 − 二重カウント（同じ人が1秒以内に線をまたぎ直した回数）
     まず全候補を高速な方法で評価し、上位の候補を実際のカウント処理と同じ方法で数え直す
  4. 評価値の高い順に、互いに離れた位置にある線を top_k 本（既定3本）選ぶ
同じエリア・同じ軌跡なら毎回同じ結果になるよう、乱数の種は固定している。
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .areas import Area
from .counting import CountLine, count_lines
from .drawing import smooth
from .tracking import TrackPoint

DEFAULT_CANDIDATES = 20000
RESCORE = 400  # 実際のカウント処理で数え直す候補の数
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
    """各線について (またいだ人数, またいだ回数の合計) を返す。"""
    A, B = lines[:, None, 0:2], lines[:, None, 2:4]  # (L, 1, 2)
    ab = B - A

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    d1 = cross(ab, P[None] - A)
    d2 = cross(ab, Q[None] - A)
    pq = (Q - P)[None]
    d3 = cross(pq, A - P[None])
    d4 = cross(pq, B - P[None])
    hit = ((d1 * d2) < 0) & ((d3 * d4) < 0)  # (L, M)
    per_person = np.add.reduceat(hit.astype(np.int32), starts, axis=1)  # (L, 人数)
    return (per_person > 0).sum(axis=1), per_person.sum(axis=1)


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
) -> list[Suggestion]:
    """1つのエリアについて、カウント線の候補を最大 top_k 本返す。

    20,000 本すべてを高速な方法（間引いた軌跡との交差判定）でふるいにかけ、
    上位 RESCORE 本だけを実際のカウント処理と同じ方法で数え直して順位を決める。
    """
    tracks_xy = {tid: smooth(pts, smooth_window) for tid, pts in tracks.items()}
    candidates = generate_candidates(area, width, height, n_candidates, seed)
    if len(candidates) == 0:
        return []
    poly = area.points()
    margin = 10
    bbox = (poly[:, 0].min() - margin, poly[:, 1].min() - margin, poly[:, 0].max() + margin, poly[:, 1].max() + margin)
    step = max(2.0, 0.004 * max(width, height))
    segs = _segments_near(tracks_xy, bbox, step)
    if segs is None:
        return []
    P, Q, starts = segs

    people = np.zeros(len(candidates), dtype=np.int64)
    total = np.zeros(len(candidates), dtype=np.int64)
    for i in range(0, len(candidates), CHUNK):
        people[i : i + CHUNK], total[i : i + CHUNK] = score_lines(candidates[i : i + CHUNK], P, Q, starts)
        if progress is not None:
            progress(min(i + CHUNK, len(candidates)), len(candidates))

    # 1段目: 高速評価の上位だけ残す（揺れによる見かけの交差を含むので、人数を重視して選ぶ）
    pre_order = np.lexsort((total - people, -people))
    pool = pre_order[: min(RESCORE, len(pre_order))]
    candidates = candidates[pool]
    # 2段目: 実際のカウント処理と同じ方法で数え直す
    people, duplicates, passes = exact_scores(candidates, tracks, smooth_window, margin, max(1, round(fps * REPEAT_SEC)))

    score = people - duplicates
    lengths = np.hypot(candidates[:, 2] - candidates[:, 0], candidates[:, 3] - candidates[:, 1])
    # 評価値が高い順。同点なら二重カウントが少ない線、さらに短い線（余計な場所を含まない）を優先
    order = np.lexsort((lengths, duplicates, -score))

    separation = max(10.0, 0.2 * np.sqrt(area.polygon_area()))
    mids = (candidates[:, 0:2] + candidates[:, 2:4]) / 2
    chosen: list[int] = []
    for idx in order:
        if people[idx] == 0:
            break
        if all(np.hypot(*(mids[idx] - mids[j])) >= separation for j in chosen):
            chosen.append(int(idx))
            if len(chosen) >= top_k:
                break

    return [
        Suggestion(
            area_index=area_index,
            rank=rank,
            line=CountLine(*(round(float(v)) for v in candidates[idx])),
            people=int(people[idx]),
            duplicates=int(duplicates[idx]),
            area_people=area_people,
            passes=int(passes[idx]),
        )
        for rank, idx in enumerate(chosen, start=1)
    ]


def save_suggestions_csv(path: Path, suggestions: list[Suggestion], areas: list[Area]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["area", "kind", "rank", "x1", "y1", "x2", "y2", "people", "passes", "duplicates", "score", "area_people", "coverage"])
        for s in suggestions:
            kind = areas[s.area_index].kind_label if s.area_index < len(areas) else ""
            ln = s.line
            w.writerow(
                [
                    f"E{s.area_index + 1}",
                    kind,
                    s.rank,
                    round(ln.x1),
                    round(ln.y1),
                    round(ln.x2),
                    round(ln.y2),
                    s.people,
                    s.passes,
                    s.duplicates,
                    s.score,
                    s.area_people,
                    f"{s.coverage:.3f}",
                ]
            )
