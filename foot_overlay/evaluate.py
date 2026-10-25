"""カウント線の信頼度を、目視の正解なしで評価する。

線の周りの足元軌跡に残る「数え間違いの痕跡」を8つの指標で数え、
信頼度（0〜100点）・評価（A/B/C）・推定範囲（下限〜上限）を出す。
指標の考え方は提案書「カウント線の信頼度評価 提案書」に合わせている。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .counting import CountLine, Crossing, find_crossings
from .tracking import TrackPoint

REPEAT_SEC = 1.0  # この秒数以内のまたぎ直しは二重カウント（画面で変えられる。既定値）
DWELL_SEC = 2.0  # 帯の中にこれ以上いたら「滞留」
SHALLOW_DEG = 30.0  # 線となす角がこれ未満なら「浅い角度」
END_FRACTION = 0.10  # 線の両端からこの割合以内なら「端ぎりぎり」
CROWD_PEOPLE = 3  # 帯の中に同時にこの人数以上いたら「混雑」
BAND_RATIO = 0.03  # 帯の幅（画像の長辺に対する割合）
DWELL_BAND_RATIO = 0.01  # 滞留を測る幅
FRAGMENT_BAND_RATIO = 0.05  # 途切れを探す幅（物陰で途切れる位置は線から少し離れていることが多い）

# 指標ごとの「危険の目安」と重み。目安で割って 0〜1 にそろえ、重みを掛ける
METRICS = {
    "stability": {"label": "位置ずらしで人数がぶれる", "threshold": 0.15, "weight": 1.0,
                  "hint": "人の流れがそろっている場所へ線を動かす"},
    "fragment": {"label": "線の近くで追跡が途切れている", "threshold": 0.10, "weight": 1.0,
                 "hint": "人の重なりや物陰の少ない場所へ線を動かす"},
    "bypass": {"label": "線の端の外をすり抜ける人がいる", "threshold": 0.10, "weight": 0.8,
               "hint": "線を通路の端まで伸ばす"},
    "duplicate": {"label": "同じ人を短時間に重ねて数えている", "threshold": 0.05, "weight": 1.0,
                  "hint": "立ち止まりや足元の揺れが少ない場所へ線を動かす"},
    "shallow": {"label": "線に対して浅い角度で通る人が多い", "threshold": 0.20, "weight": 0.5,
                "hint": "線を人の流れに直角に近づける"},
    "near_end": {"label": "線の端ぎりぎりを通る人が多い", "threshold": 0.20, "weight": 0.3,
                 "hint": "線を少し長くする"},
    "dwell": {"label": "線の近くで立ち止まる人がいる", "threshold": 0.10, "weight": 0.5,
              "hint": "立ち止まる場所から線をずらす"},
    "crowd": {"label": "混雑した瞬間の通過が多い", "threshold": 0.30, "weight": 0.5,
              "hint": "人がばらけて通る場所へ線を動かす"},
    "suspect": {"label": "誤判定らしい追跡線が通っている", "threshold": 0.05, "weight": 1.0,
                "hint": "誤判定らしい線（灰色の点線）が通らない位置へ線を動かす"},
}


@dataclass
class LineEvaluation:
    passes: int  # 通過回数 N（誤判定らしい線の通過も含む）
    people: int  # 通過した人数
    metrics: dict[str, float]  # 指標ごとの値（0〜1の割合）
    reliability: float  # 信頼度 0〜100
    grade: str  # A / B / C
    lower: int  # 推定範囲の下限
    upper: int  # 推定範囲の上限
    reasons: list[str] = field(default_factory=list)  # 信頼度が低い理由（上位2つ）
    suspect_passes: int = 0  # 誤判定らしい線の通過回数
    duplicates: int = 0  # 同じ人を短時間（repeat_sec 秒以内）に重ねて数えた回数
    hints: list[str] = field(default_factory=list)  # その直し方
    break_points: list[tuple[float, float]] = field(default_factory=list)  # 線の近くで途切れた位置
    bypass_points: list[tuple[float, float]] = field(default_factory=list)  # すり抜けた位置

    @property
    def range_text(self) -> str:
        return f"{self.lower}〜{self.upper}"

    @property
    def summary(self) -> str:
        """例: 28人（推定 25〜31人、信頼度 82点 B）"""
        return f"{self.passes}人（推定 {self.range_text}人、信頼度 {self.reliability:.0f}点 {self.grade}）"


def grade_of(reliability: float) -> str:
    if reliability >= 85:
        return "A"
    if reliability >= 70:
        return "B"
    return "C"


def reliability_from(metrics: dict[str, float]) -> tuple[float, list[tuple[float, str]]]:
    """指標の値から信頼度と、各指標の減点の大きさを出す。"""
    score = 1.0
    penalties = []
    for key, value in metrics.items():
        spec = METRICS[key]
        risk = min(1.0, value / spec["threshold"]) if spec["threshold"] else 0.0
        factor = spec["weight"] * risk * 0.5  # 1つの指標で最大でも半分まで
        score *= 1.0 - factor
        penalties.append((factor, key))
    return 100.0 * score, sorted(penalties, reverse=True)


def perturbed_lines(line: CountLine, width: int, height: int) -> list[CountLine]:
    """位置ずらし安定性に使う10本: 平行移動 ±1%・±2%、回転 ±5°・±10°、長さ ±10%。"""
    p1, p2 = line.p1, line.p2
    mid = (p1 + p2) / 2
    n = line.normal()
    size = max(width, height)
    out = []
    for frac in (-0.02, -0.01, 0.01, 0.02):
        shift = n * size * frac
        out.append(CountLine(*(p1 + shift), *(p2 + shift)))
    for deg in (-10, -5, 5, 10):
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        q1, q2 = mid + rot @ (p1 - mid), mid + rot @ (p2 - mid)
        out.append(CountLine(*q1, *q2))
    for scale in (0.9, 1.1):
        q1, q2 = mid + (p1 - mid) * scale, mid + (p2 - mid) * scale
        out.append(CountLine(*q1, *q2))
    return out


def _near_line(line: CountLine, xy: np.ndarray, band: float) -> np.ndarray:
    """各点が「帯」（線分から band px 以内、線分の範囲内）に入っているか。"""
    d = line.p2 - line.p1
    length2 = float(d @ d) or 1.0
    t = ((xy - line.p1) @ d) / length2
    dist = np.abs(line.signed_distance(xy))
    return (t >= 0) & (t <= 1) & (dist <= band)


def _extensions(line: CountLine, size: int, area_polygon: np.ndarray | None) -> list[CountLine]:
    """すり抜けを調べる、線の両端の延長部分。

    線がエリアの中にあれば、エリアの縁まで延ばす（その外は別の通路なので見ない）。
    そうでなければ、線の長さ分（最低でも画像の長辺の10%）だけ延ばす。
    """
    d = line.p2 - line.p1
    length = float(np.hypot(*d)) or 1.0
    unit = d / length
    if area_polygon is not None and len(area_polygon) >= 3:
        from .suggest import clip_line_to_polygon

        seg = clip_line_to_polygon((line.p1 + line.p2) / 2, unit, np.asarray(area_polygon, dtype=np.float64))
        if seg is not None:
            ta = float((seg[0] - line.p1) @ unit)  # 線の始点から見た、エリアの縁までの距離（負なら手前側）
            tb = float((seg[1] - line.p1) @ unit)
            lo, hi = min(ta, tb), max(ta, tb)
            out = []
            if lo < -1:
                out.append(CountLine(*(line.p1 + unit * lo), *line.p1))
            if hi > length + 1:
                out.append(CountLine(*line.p2, *(line.p1 + unit * hi)))
            return out
    ext = max(length, 0.1 * size)
    return [CountLine(*(line.p1 - unit * ext), *line.p1), CountLine(*line.p2, *(line.p2 + unit * ext))]


def _segment_t(line: CountLine, x: float, y: float) -> float:
    d = line.p2 - line.p1
    return float(((np.array([x, y]) - line.p1) @ d) / (float(d @ d) or 1.0))


def evaluate_line(
    line: CountLine,
    tracks: dict[int, list[TrackPoint]],
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    fps: float,
    margin: float,
    stability: bool = True,
    suspects: set[int] | None = None,
    area_polygon: np.ndarray | None = None,
    repeat_sec: float = REPEAT_SEC,
) -> LineEvaluation:
    """1本のカウント線の信頼度を評価する。

    tracks_xy は重畳画像・カウントと同じく移動平均した足元座標。
    """
    size = max(width, height)
    band = BAND_RATIO * size
    frag_band = FRAGMENT_BAND_RATIO * size
    extension = _extensions(line, size, area_polygon)
    pad = max(frag_band, max((e.length() for e in extension), default=0.0)) + 0.15 * line.length()
    x0, x1 = min(line.x1, line.x2) - pad, max(line.x1, line.x2) + pad
    y0, y1 = min(line.y1, line.y2) - pad, max(line.y1, line.y2) + pad
    # 線の付近を通る人だけを調べる（遠くの人は結果に関係しないので速くなる）
    near_ids = [
        tid
        for tid, xy in tracks_xy.items()
        if len(xy) and xy[:, 0].max() >= x0 and xy[:, 0].min() <= x1 and xy[:, 1].max() >= y0 and xy[:, 1].min() <= y1
    ]

    def crossings_of(ln: CountLine, ids=near_ids) -> list[Crossing]:
        out: list[Crossing] = []
        for tid in ids:
            out += find_crossings(ln, 0, tid, tracks[tid], tracks_xy[tid], margin)
        return out

    crossings = crossings_of(line)
    passes = len(crossings)
    crossed_ids = sorted({c.track_id for c in crossings})
    people = len(crossed_ids)
    n = max(passes, 1)
    repeat_frames = max(1, round(fps * repeat_sec))

    # 4. 二重カウント: 同じ人が repeat_frames 以内にまたぎ直した回数
    duplicates = 0
    last: dict[int, int] = {}
    for c in sorted(crossings, key=lambda c: (c.track_id, c.frame)):
        if c.track_id in last and c.frame - last[c.track_id] <= repeat_frames:
            duplicates += 1
        last[c.track_id] = c.frame

    # 2. 途切れ: 帯の中で始まる・終わる追跡（画面の縁・動画の最初と最後は除く）
    edge = 0.02 * size
    first_frame = min((pts[0].frame for pts in tracks.values() if pts), default=0)
    last_frame = max((pts[-1].frame for pts in tracks.values() if pts), default=0)
    break_points: list[tuple[float, float]] = []
    for tid in near_ids:
        xy, pts = tracks_xy[tid], tracks[tid]
        for idx, at_video_edge in ((0, pts[0].frame <= first_frame + 1), (-1, pts[-1].frame >= last_frame - 1)):
            if at_video_edge:
                continue
            x, y = xy[idx]
            if x <= edge or x >= width - edge or y >= height - edge:
                continue  # 画面の縁で出入りしただけ
            if _near_line(line, xy[idx : idx + 1] if idx == 0 else xy[-1:], frag_band)[0]:
                break_points.append((float(x), float(y)))
    fragments = len(break_points)

    # 3. すり抜け: 線の延長部分だけを横切った人
    d = line.p2 - line.p1
    crossed = set(crossed_ids)
    bypass_points: list[tuple[float, float]] = []
    bypass_ids = set()
    for ext in extension:
        for c in crossings_of(ext):
            if c.track_id not in crossed and c.track_id not in bypass_ids:
                bypass_ids.add(c.track_id)
                bypass_points.append((c.x, c.y))
    bypass = len(bypass_ids)

    # 5. 浅い角度 / 6. 端ぎりぎり: 通過ごとに、前後0.5秒の進行方向と通過位置を見る
    half = max(1, round(fps * 0.5))
    line_dir = d / (np.hypot(*d) or 1.0)
    shallow = near_end = 0
    for c in crossings:
        pts = tracks[c.track_id]
        frames = np.array([p.frame for p in pts])
        i = int(np.searchsorted(frames, c.frame))
        a, b = max(0, i - half), min(len(pts) - 1, i + half)
        move = tracks_xy[c.track_id][b] - tracks_xy[c.track_id][a]
        norm = np.hypot(*move)
        if norm > 0:
            sin_angle = abs(move[0] * line_dir[1] - move[1] * line_dir[0]) / norm
            if math.degrees(math.asin(min(1.0, sin_angle))) < SHALLOW_DEG:
                shallow += 1
        t = _segment_t(line, c.x, c.y)
        if t < END_FRACTION or t > 1 - END_FRACTION:
            near_end += 1

    # 7. 滞留: 通過した人のうち、線のすぐそば（画像の1%以内）に DWELL_SEC 以上いた人。
    #    帯（3%）で測ると、普通に歩いて通り抜けるだけでも2秒近くかかるので狭くする
    dwell_frames = fps * DWELL_SEC
    dwell_band = DWELL_BAND_RATIO * size
    dwellers = sum(int(_near_line(line, tracks_xy[tid], dwell_band).sum()) >= dwell_frames for tid in crossed_ids)

    # 8. 混雑: 通過の瞬間に、帯の中に CROWD_PEOPLE 人以上いた通過
    crowded = 0
    if crossings:
        in_band: dict[int, int] = {}
        for tid in near_ids:
            inside = _near_line(line, tracks_xy[tid], band)
            for f in (p.frame for p, ok in zip(tracks[tid], inside) if ok):
                in_band[f] = in_band.get(f, 0) + 1
        crowded = sum(in_band.get(c.frame, 0) >= CROWD_PEOPLE for c in crossings)

    # 1. 位置ずらし安定性
    counts = [passes]
    if stability:
        counts += [len(crossings_of(ln)) for ln in perturbed_lines(line, width, height)]
    instability = (max(counts) - min(counts)) / n if passes else 0.0

    suspects = suspects or set()
    suspect_passes = sum(c.track_id in suspects for c in crossings)
    metrics = {
        "stability": instability,
        "fragment": fragments / (passes + fragments) if passes + fragments else 0.0,
        "bypass": bypass / (passes + bypass) if passes + bypass else 0.0,
        "duplicate": duplicates / n,
        "shallow": shallow / n,
        "near_end": near_end / n,
        "dwell": dwellers / max(people, 1),
        "crowd": crowded / n,
        "suspect": suspect_passes / n,
    }
    reliability, penalties = reliability_from(metrics)
    reliability = round(reliability)  # 表示する点数で評価（A/B/C）を決める（84.9点が「85点B」にならないように）
    if passes == 0:
        reliability = 0.0  # 誰も通らない線は評価できない
    top = [key for factor, key in penalties if factor >= 0.05][:2]
    lower = max(0, min(passes - duplicates, min(counts)))
    upper = max(passes + math.ceil(fragments / 2) + bypass, max(counts))
    return LineEvaluation(
        passes=passes,
        people=people,
        metrics=metrics,
        reliability=reliability,
        grade=grade_of(reliability) if passes else "-",
        lower=lower,
        upper=upper,
        reasons=[METRICS[k]["label"] for k in top],
        hints=[METRICS[k]["hint"] for k in top],
        break_points=break_points,
        bypass_points=bypass_points,
        suspect_passes=suspect_passes,
        duplicates=duplicates,
    )
