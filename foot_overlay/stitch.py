"""途切れた追跡線をつなぐ。

人が重なったり柱の陰に隠れたりすると、追跡が途切れて同じ人が別の追跡IDになる。
そのままだと1人が2人に数えられたり、線の手前と奥で別人扱いになって数え漏れたりする。

ある線が終わってから少し後に、終わった位置・歩く向きと速さから見て自然な位置で
別の線が始まっていれば、同じ人とみなして1本にまとめる。間の見えなかった区間は
直線で補う（補わないと、つないだ所が「瞬間移動」に見えてしまうため）。

画面の縁で終わった線（画面の外へ出て行った人）や、縁から始まった線はつながない。
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass

import numpy as np

from .tracking import TrackPoint

KEEP_MIN_POINTS = 3  # 解析結果にはこの回数以上映った線を残す（短い切れ端もつなぐ材料にするため）
MAX_GAP_SEC = 2.0  # これ以上長く見えなかった線はつながない
EDGE_RATIO = 0.03  # 画面の縁からこの割合以内で終わる・始まる線は、出入りした人とみなしてつながない
NEAR_RATIO = 0.03  # 予想した位置からのずれの許容（画像の長辺に対する割合）
SPEED_SLACK = 0.5  # 見えなかった時間が長いほど、ずれの許容を「速さ × 時間 × これ」だけ広げる
VELOCITY_SEC = 0.5  # 歩く向きと速さは、線の端のこの秒数分の動きから求める


@dataclass
class StitchSettings:
    """途切れた追跡線のつなぎ方（② 確認の画面で変えられる。meta.json に保存）。"""

    on: bool = True
    max_gap_sec: float = MAX_GAP_SEC

    @classmethod
    def from_dict(cls, d: dict | None) -> "StitchSettings":
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base

    def to_dict(self) -> dict:
        return asdict(self)


def _end_state(pts: list[TrackPoint], fps: float, at_end: bool):
    """線の端の (フレーム, 位置, 1フレームあたりの速度)。端の数点を平均してぶれを抑える。"""
    k = max(2, round(VELOCITY_SEC * fps))
    seg = pts[-k:] if at_end else pts[:k]
    xy = np.array([[p.x, p.y] for p in seg])
    frames = np.array([p.frame for p in seg], dtype=float)
    span = frames[-1] - frames[0]
    vel = (xy[-1] - xy[0]) / span if span > 0 else np.zeros(2)
    tip = xy[-min(3, len(xy)) :].mean(axis=0) if at_end else xy[: min(3, len(xy))].mean(axis=0)
    frame = seg[-1].frame if at_end else seg[0].frame
    return frame, tip, vel


def _near_edge(p: np.ndarray, width: int, height: int) -> bool:
    m = EDGE_RATIO * max(width, height)
    return p[0] <= m or p[0] >= width - 1 - m or p[1] <= m or p[1] >= height - 1 - m


def fill_gaps(pts: list[TrackPoint], max_gap: int) -> list[TrackPoint]:
    """見えなかったフレーム（max_gap フレーム以内）を直線で補う。"""
    out: list[TrackPoint] = []
    for p in pts:
        if out:
            q = out[-1]
            gap = p.frame - q.frame
            if 1 < gap <= max_gap:
                for f in range(q.frame + 1, p.frame):
                    t = (f - q.frame) / gap
                    out.append(TrackPoint(f, q.x + (p.x - q.x) * t, q.y + (p.y - q.y) * t))
        out.append(p)
    return out


def stitch_tracks(
    tracks: dict[int, list[TrackPoint]],
    width: int,
    height: int,
    fps: float,
    settings: StitchSettings | None = None,
) -> tuple[dict[int, list[TrackPoint]], int]:
    """途切れた追跡線をつなぐ。戻り値は (つないだ後の追跡線, つないだ箇所の数)。

    つないだ線のIDは、先に始まった線のIDを使う。
    """
    cfg = settings or StitchSettings()
    if not cfg.on or len(tracks) < 2:
        return tracks, 0
    max_gap = max(1, round(cfg.max_gap_sec * fps))
    size = max(width, height)
    ids = [t for t in sorted(tracks) if len(tracks[t]) >= 2]
    ends = {t: _end_state(tracks[t], fps, True) for t in ids}
    starts = {t: _end_state(tracks[t], fps, False) for t in ids}

    # 始まりのフレーム順に並べ、終わりから max_gap 以内に始まる線だけを比べる
    by_start = sorted(ids, key=lambda t: starts[t][0])
    start_frames = [starts[t][0] for t in by_start]
    pairs = []
    for a in ids:
        fe, pe, ve = ends[a]
        if _near_edge(pe, width, height):
            continue  # 画面の外へ出て行った人
        lo = bisect.bisect_right(start_frames, fe)
        hi = bisect.bisect_right(start_frames, fe + max_gap)
        for b in by_start[lo:hi]:
            if b == a:
                continue
            fs, ps, vs = starts[b]
            if _near_edge(ps, width, height):
                continue  # 画面の外から入ってきた人
            gap = fs - fe
            predicted = pe + ve * gap
            dist = float(np.hypot(*(ps - predicted)))
            allowed = NEAR_RATIO * size + SPEED_SLACK * float(np.hypot(*ve)) * gap
            if dist > allowed:
                continue
            # どちらもはっきり歩いているのに向きが逆なら別人
            if np.hypot(*ve) * fps > 0.02 * size and np.hypot(*vs) * fps > 0.02 * size and float(ve @ vs) < 0:
                continue
            pairs.append((dist / allowed + 0.3 * gap / max_gap, a, b))

    # ずれの小さい組から順に、1本の線の終わりと始まりを1回ずつだけ使ってつなぐ
    next_of: dict[int, int] = {}
    prev_of: dict[int, int] = {}
    for _, a, b in sorted(pairs):
        if a in next_of or b in prev_of:
            continue
        # 輪にならないように（a の線の先頭が b なら、つなぐと輪になる）
        head = a
        while head in prev_of:
            head = prev_of[head]
        if head == b:
            continue
        next_of[a] = b
        prev_of[b] = a

    out: dict[int, list[TrackPoint]] = {}
    for t in sorted(tracks):
        if t in prev_of:
            continue  # 前の線につながれた
        pts = list(tracks[t])
        cur = t
        while cur in next_of:
            cur = next_of[cur]
            pts += tracks[cur]
        out[t] = fill_gaps(pts, max_gap)
    return out, len(next_of)
