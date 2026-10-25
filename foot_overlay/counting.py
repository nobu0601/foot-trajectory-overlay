"""カウント線をまたいだ人数を数える。

線は P1（始点）→P2（終点）の線分。始点→終点の向きに立って左側を A 側、右側を B 側と呼び
（画像の座標は y が下向き）、A→B を「IN」、B→A を「OUT」として向き別に数える。
IN と OUT を入れ替えたいときは、始点と終点を入れ替える（CountLine.reversed）。
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
    name: str = ""  # 線の名前（例: 改札口）。空なら「L1」のように番号で呼ぶ

    def label(self, index: int) -> str:
        """表示用の名前（名前がなければ L1, L2…）。"""
        return self.name or f"L{index + 1}"

    def reversed(self) -> "CountLine":
        """始点と終点を入れ替えた線（IN と OUT が入れ替わる）。"""
        return CountLine(self.x2, self.y2, self.x1, self.y1, self.name)

    def in_direction(self) -> tuple[float, float]:
        """IN として数える移動の向き（単位ベクトル、画像の座標で y は下向き）。"""
        n = self.normal()
        return float(n[0]), float(n[1])

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
    direction: str  # "IN"（A側→B側）または "OUT"（B側→A側）
    x: float
    y: float


@dataclass
class LineCount:
    in_count: int = 0
    out_count: int = 0

    @property
    def total(self) -> int:
        return self.in_count + self.out_count


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
    # 線から margin より離れた点だけで「どちら側にいるか」を確定させ、確定した側が変わった所だけ調べる
    # （1点ずつ回すと遅いので、側の入れ替わりを配列でまとめて見つける）
    confirmed = np.flatnonzero(np.abs(dist) > margin)
    if len(confirmed) < 2:
        return []
    sides = np.sign(dist[confirmed])
    changes = np.flatnonzero(sides[1:] != sides[:-1])
    crossings: list[Crossing] = []
    for c in changes:
        start, i = int(confirmed[c]), int(confirmed[c + 1])
        hit = _first_hit(xy, start, i, line)
        if hit is not None:
            crossings.append(
                Crossing(
                    line_index=line_index,
                    track_id=track_id,
                    frame=points[i].frame,
                    direction="IN" if sides[c + 1] > 0 else "OUT",
                    x=float(hit[0]),
                    y=float(hit[1]),
                )
            )
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
                if c.direction == "IN":
                    counts[li].in_count += 1
                else:
                    counts[li].out_count += 1
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


def save_counts_csv(path: Path, lines: list[CountLine], counts: list[LineCount], evaluations=None) -> None:
    """線ごとの人数。evaluations（evaluate.LineEvaluation の一覧）があれば信頼度・推定範囲・重複も書く。"""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["line", "x1", "y1", "x2", "y2", "IN", "OUT", "total"]
        if evaluations is not None:
            head += ["duplicates", "estimate_low", "estimate_high", "reliability", "grade", "reasons"]
        w.writerow(head)
        for i, (ln, c) in enumerate(zip(lines, counts)):
            row = [ln.label(i), round(ln.x1), round(ln.y1), round(ln.x2), round(ln.y2), c.in_count, c.out_count, c.total]
            if evaluations is not None and i < len(evaluations):
                ev = evaluations[i]
                row += [ev.duplicates, ev.lower, ev.upper, f"{ev.reliability:.0f}", ev.grade, "・".join(ev.reasons)]
            w.writerow(row)


def save_crossings_csv(path: Path, crossings: list[Crossing], fps: float, lines: list[CountLine] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["line", "track_id", "frame", "time_sec", "direction", "x", "y"])
        for c in sorted(crossings, key=lambda c: (c.line_index, c.frame, c.track_id)):
            name = lines[c.line_index].label(c.line_index) if lines and c.line_index < len(lines) else f"L{c.line_index + 1}"
            w.writerow([name, c.track_id, c.frame, f"{c.frame / fps:.3f}", c.direction, f"{c.x:.1f}", f"{c.y:.1f}"])


# ---- カウント線の書き出し・読み込み（JSON） ----

LINES_JSON_FORMAT = "foot-trajectory-overlay/count-lines"


def export_lines_json(path: Path, lines: list[CountLine], width: int, height: int, video: str = "") -> None:
    """カウント線の情報（名前・始点終点の座標・IN の向き）を JSON に書き出す。

    座標は画像のピクセル（左上が原点、x は右向き、y は下向き）。
    IN = 始点→終点を向いて左側から右側へ通過すること。in_direction はその移動の向き（単位ベクトル）。
    """
    data = {
        "format": LINES_JSON_FORMAT,
        "version": 1,
        "video": video,
        "image": {"width": width, "height": height},
        "coordinates": "pixel; origin top-left; x right, y down",
        "in_rule": "IN = crossing from the left side to the right side of start->end",
        "lines": [
            {
                "name": ln.label(i),
                "start": {"x": round(ln.x1, 1), "y": round(ln.y1, 1)},
                "end": {"x": round(ln.x2, 1), "y": round(ln.y2, 1)},
                "in_direction": {"x": round(ln.in_direction()[0], 4), "y": round(ln.in_direction()[1], 4)},
            }
            for i, ln in enumerate(lines)
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def scale_lines(lines: list[CountLine], from_w: int, from_h: int, to_w: int, to_h: int) -> list[CountLine]:
    """画像の大きさが違う動画へ線を移すときに、座標を大きさの比で合わせる（名前と向きはそのまま）。"""
    sx = to_w / from_w if from_w else 1.0
    sy = to_h / from_h if from_h else 1.0
    return [CountLine(ln.x1 * sx, ln.y1 * sy, ln.x2 * sx, ln.y2 * sy, ln.name) for ln in lines]


def import_lines_json(path: Path, width: int, height: int) -> tuple[list[CountLine], str]:
    """export_lines_json で書き出した JSON を読み込む。

    書き出したときと画像の大きさが違えば、座標を大きさの比で拡大・縮小する。
    戻り値の2つ目は利用者向けの注意書き（なければ空）。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != LINES_JSON_FORMAT or "lines" not in data:
        raise ValueError("このツールで書き出したカウント線の JSON ではありません。")
    img = data.get("image") or {}
    sx = width / img["width"] if img.get("width") else 1.0
    sy = height / img["height"] if img.get("height") else 1.0
    lines = []
    for d in data["lines"]:
        ln = CountLine(
            float(d["start"]["x"]) * sx,
            float(d["start"]["y"]) * sy,
            float(d["end"]["x"]) * sx,
            float(d["end"]["y"]) * sy,
            str(d.get("name", "")),
        )
        # 向きは始点・終点の並びで決まる。in_direction と逆なら入れ替える（手で編集された JSON のため）
        want = d.get("in_direction")
        if want and (ln.in_direction()[0] * want["x"] + ln.in_direction()[1] * want["y"]) < 0:
            ln = ln.reversed()
        lines.append(ln)
    note = ""
    if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
        note = f"画像の大きさが違うため（{img.get('width')}×{img.get('height')} → {width}×{height}）、座標を拡大・縮小しました。"
    return lines, note


def parse_line_arg(text: str) -> CountLine:
    """"x1,y1,x2,y2" 形式の文字列を CountLine にする。"""
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"カウント線は x1,y1,x2,y2 の4つの数で指定してください: {text}")
    return CountLine(*parts)
