"""カウント線をまたいだ人数を数える。

線は P1→P2 の線分。線の左右を A 側 / B 側と呼び、
A→B と B→A の向き別に、またいだ回数を数える。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .drawing import smooth
from .tracking import TrackPoint

DEFAULT_SMOOTH = 5


@dataclass
class CountLine:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def p1(self) -> np.ndarray:
        return np.array([self.x1, self.y1], dtype=np.float64)

    @property
    def p2(self) -> np.ndarray:
        return np.array([self.x2, self.y2], dtype=np.float64)

    def normal(self) -> np.ndarray:
        """B 側を指す単位法線。"""
        d = self.p2 - self.p1
        n = np.array([-d[1], d[0]])
        length = np.hypot(*n)
        return n / length if length else n

    def signed_distance(self, xy: np.ndarray) -> np.ndarray:
        """各点が線から何px離れているか。負なら A 側、正なら B 側。"""
        return (xy - self.p1) @ self.normal()

    def length(self) -> float:
        return float(np.hypot(*(self.p2 - self.p1)))


@dataclass
class Crossing:
    line_index: int
    track_id: int
    frame: int
    direction: str  # "A->B" または "B->A"
    x: float
    y: float


@dataclass
class LineCount:
    a_to_b: int = 0
    b_to_a: int = 0

    @property
    def total(self) -> int:
        return self.a_to_b + self.b_to_a


def segments_intersect(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """線分 pq と線分 ab が交わるか（端点で接する場合も含む）。"""

    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    d1, d2 = cross(a, b, p), cross(a, b, q)
    d3, d4 = cross(p, q, a), cross(p, q, b)
    if ((d1 > 0) != (d2 > 0) or d1 == 0 or d2 == 0) and ((d3 > 0) != (d4 > 0) or d3 == 0 or d4 == 0):
        # 同一直線上で重ならないケースを除く
        if d1 == d2 == 0:
            return max(min(p[0], q[0]), min(a[0], b[0])) <= min(max(p[0], q[0]), max(a[0], b[0])) and max(
                min(p[1], q[1]), min(a[1], b[1])
            ) <= min(max(p[1], q[1]), max(a[1], b[1]))
        return True
    return False


def find_crossings(
    line: CountLine,
    line_index: int,
    track_id: int,
    points: list[TrackPoint],
    xy: np.ndarray,
    margin: float,
) -> list[Crossing]:
    """1人分の軌跡が線をまたいだ瞬間をすべて返す。

    足元が線の上で小刻みに揺れても何度も数えないよう、線から margin px 以上
    離れて初めて「その側にいる」と確定させ、確定した側が入れ替わったときだけ数える。
    さらに、線分の端より外側を回り込んだだけの場合は数えない。
    """
    if line.length() == 0 or len(xy) < 2:
        return []
    dist = line.signed_distance(xy)
    crossings: list[Crossing] = []
    side = 0  # -1: A 側, +1: B 側, 0: 未確定
    side_index = 0
    for i, d in enumerate(dist):
        if abs(d) <= margin:
            continue
        new_side = 1 if d > 0 else -1
        if side != 0 and new_side != side:
            hit = _first_hit(xy, side_index, i, line)
            if hit is not None:
                crossings.append(
                    Crossing(
                        line_index=line_index,
                        track_id=track_id,
                        frame=points[i].frame,
                        direction="A->B" if new_side > 0 else "B->A",
                        x=float(hit[0]),
                        y=float(hit[1]),
                    )
                )
        side = new_side
        side_index = i
    return crossings


def _first_hit(xy: np.ndarray, start: int, end: int, line: CountLine) -> np.ndarray | None:
    """xy[start..end] の折れ線が線分と最初に交わる位置。交わらなければ None。"""
    a, b = line.p1, line.p2
    for j in range(start, end):
        p, q = xy[j], xy[j + 1]
        if segments_intersect(p, q, a, b):
            dp, dq = line.signed_distance(np.stack([p, q]))
            t = dp / (dp - dq) if dp != dq else 0.0
            return p + (q - p) * t
    return None


def count_lines(
    tracks: dict[int, list[TrackPoint]],
    lines: list[CountLine],
    smooth_window: int = DEFAULT_SMOOTH,
    margin: float = 3.0,
) -> tuple[list[LineCount], list[Crossing]]:
    """全ての線について、向き別の通過人数と、通過イベントの一覧を返す。

    軌跡は重畳画像と同じ幅で移動平均してから判定する（画像で見た線と結果を一致させるため）。
    """
    smoothed = {tid: smooth(pts, smooth_window) for tid, pts in tracks.items()}
    counts = [LineCount() for _ in lines]
    crossings: list[Crossing] = []
    for li, line in enumerate(lines):
        for tid in sorted(tracks):
            for c in find_crossings(line, li, tid, tracks[tid], smoothed[tid], margin):
                crossings.append(c)
                if c.direction == "A->B":
                    counts[li].a_to_b += 1
                else:
                    counts[li].b_to_a += 1
    return counts, crossings


# ---- 読み込み・保存 ----


def load_tracks_csv(path: Path) -> dict[int, list[TrackPoint]]:
    tracks: dict[int, list[TrackPoint]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tid = int(row["track_id"])
            tracks.setdefault(tid, []).append(TrackPoint(int(row["frame"]), float(row["foot_x"]), float(row["foot_y"])))
    for pts in tracks.values():
        pts.sort(key=lambda p: p.frame)
    return tracks


def estimate_fps(tracks_csv: Path, default: float = 30.0) -> float:
    """meta.json がない古い結果用に、tracks.csv の frame と time_sec から fps を求める。"""
    with tracks_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_sec"])
            if t > 0:
                return int(row["frame"]) / t
    return default


def load_meta(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_lines(path: Path) -> list[CountLine]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [CountLine(**ln) for ln in data.get("lines", [])]


def save_lines(path: Path, lines: list[CountLine]) -> None:
    data = {"lines": [asdict(ln) for ln in lines]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_counts_csv(path: Path, lines: list[CountLine], counts: list[LineCount]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["line", "x1", "y1", "x2", "y2", "A->B", "B->A", "total"])
        for i, (ln, c) in enumerate(zip(lines, counts), start=1):
            w.writerow([f"L{i}", round(ln.x1), round(ln.y1), round(ln.x2), round(ln.y2), c.a_to_b, c.b_to_a, c.total])


def save_crossings_csv(path: Path, crossings: list[Crossing], fps: float) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["line", "track_id", "frame", "time_sec", "direction", "x", "y"])
        for c in sorted(crossings, key=lambda c: (c.line_index, c.frame, c.track_id)):
            w.writerow(
                [f"L{c.line_index + 1}", c.track_id, c.frame, f"{c.frame / fps:.3f}", c.direction, f"{c.x:.1f}", f"{c.y:.1f}"]
            )


def parse_line_arg(text: str) -> CountLine:
    """"x1,y1,x2,y2" 形式の文字列を CountLine にする。"""
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"カウント線は x1,y1,x2,y2 の4つの数で指定してください: {text}")
    return CountLine(*parts)
