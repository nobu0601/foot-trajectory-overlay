"""人の流れを基準に、カウント線の候補（流れに直角・エリアの縁から縁まで）を作る。

  1. 流れの向き: エリアをマスに分け、マスごとに通った人の進行方向を「軸」として平均する
     （行きと帰りが打ち消し合わないよう、角度を2倍にして平均する）
  2. 中心線: エリアの真ん中を通る線（エリアを細めていって最後に残る線）。枝分かれも含む
  3. 立ち位置: 中心線の上に1pxおきに置く
  4. 各立ち位置で、そこでの流れに直角な向きを基準に ±max_tilt° を1°きざみで変えた線を、
     エリアの縁から縁まで引く（画面の下端・左右の端の近くは除く）
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .areas import Area

MAX_TILT_DEG = 20
CELL_RATIO = 0.02  # 流れの向きを調べるマスの大きさ（画像の長辺に対する割合）
EDGE_RATIO = 0.01  # 画面の下端・左右の端からこの距離以内には線を引かない


# ---------- 1. 流れの向き ----------


def axis_field(tracks_xy: dict[int, np.ndarray], width: int, height: int) -> tuple[float, np.ndarray, np.ndarray]:
    """マスごとの流れの軸（ラジアン, 0〜π）と、向きのそろい具合（0〜1）。"""
    cell = max(4.0, CELL_RATIO * max(width, height))
    gw, gh = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    c2 = np.zeros((gh, gw), np.float32)
    s2 = np.zeros((gh, gw), np.float32)
    wsum = np.zeros((gh, gw), np.float32)
    for xy in tracks_xy.values():
        if len(xy) < 2:
            continue
        d = xy[1:] - xy[:-1]
        length = np.hypot(d[:, 0], d[:, 1])
        ok = length > 0.5
        if not ok.any():
            continue
        mid = (xy[1:] + xy[:-1])[ok] / 2
        ang = np.arctan2(d[ok, 1], d[ok, 0])
        gx = np.clip((mid[:, 0] / cell).astype(int), 0, gw - 1)
        gy = np.clip((mid[:, 1] / cell).astype(int), 0, gh - 1)
        np.add.at(c2, (gy, gx), (np.cos(2 * ang) * length[ok]).astype(np.float32))
        np.add.at(s2, (gy, gx), (np.sin(2 * ang) * length[ok]).astype(np.float32))
        np.add.at(wsum, (gy, gx), length[ok].astype(np.float32))
    # となりのマスとならして、通った人が少ないマスでも向きが決まるようにする
    c2 = cv2.GaussianBlur(c2, (5, 5), 1.0)
    s2 = cv2.GaussianBlur(s2, (5, 5), 1.0)
    wsum = cv2.GaussianBlur(wsum, (5, 5), 1.0)
    axis = (0.5 * np.arctan2(s2, c2)) % np.pi
    strength = np.where(wsum > 1e-6, np.hypot(c2, s2) / np.maximum(wsum, 1e-6), 0.0)
    return cell, axis, strength


def dominant_axes(tracks_xy: dict[int, np.ndarray], area: Area, width: int, height: int) -> list[float]:
    """エリア全体の主な流れの軸。交差点のように2方向あれば2つ返す。"""
    mask = area.mask(width, height)
    hist = np.zeros(36)  # 5°きざみ（軸なので0〜180°）
    for xy in tracks_xy.values():
        if len(xy) < 2:
            continue
        pts = np.round(xy).astype(int)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        inside[inside] = mask[pts[inside, 1], pts[inside, 0]] > 0
        seg_in = inside[1:] & inside[:-1]
        d = (xy[1:] - xy[:-1])[seg_in]
        if not len(d):
            continue
        length = np.hypot(d[:, 0], d[:, 1])
        deg = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
        np.add.at(hist, (deg // 5).astype(int) % 36, length)
    if hist.sum() == 0:
        return []
    smoothed = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3
    first = int(np.argmax(smoothed))
    axes = [math.radians(first * 5 + 2.5)]
    # 1つ目から45°以上離れていて、1つ目の30%以上の強さがあれば2つ目の流れとみなす
    far = np.array([min(abs(i - first), 36 - abs(i - first)) >= 9 for i in range(36)])
    if far.any():
        second = int(np.argmax(np.where(far, smoothed, -1)))
        if smoothed[second] >= 0.3 * smoothed[first]:
            axes.append(math.radians(second * 5 + 2.5))
    return axes


# ---------- 2. 中心線 ----------


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 法で、図形を幅1画素の芯（中心線）になるまで細める。"""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9, p1 = img[1:-1, :-2], img[:-2, :-2], img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            remove = (p1 == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & (c1 == 0) & (c2 == 0)
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
    return img[1:-1, 1:-1]


def centerline_points(area: Area, width: int, height: int) -> np.ndarray:
    """エリアの中心線（枝分かれを含む）の上に、約1pxおきに並べた点。戻り値は (N, 2)。"""
    mask = area.mask(width, height)
    if mask.sum() == 0:
        return np.zeros((0, 2))
    down = max(1, int(math.ceil(max(width, height) / 480)))  # 細める処理を軽くするため縮小する
    small = cv2.resize(mask, (max(1, width // down), max(1, height // down)), interpolation=cv2.INTER_NEAREST)
    skel = thin(small)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        cy, cx = np.argwhere(mask).mean(axis=0)
        return np.array([[cx, cy]])
    # 芯の画素どうしをつなぎ、その間を1pxおきに埋める（縮小した分を戻すため）
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    pts = np.stack([xs, ys], axis=1).astype(np.float64) * down + down / 2
    out = [pts]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):  # 各組を1回だけ見る
            j = index.get((int(y) + dy, int(x) + dx))
            if j is None:
                continue
            a, b = pts[i], pts[j]
            n = int(np.hypot(*(b - a)))
            if n > 1:
                t = np.arange(1, n)[:, None] / n
                out.append(a + (b - a) * t)
    return np.concatenate(out)


def _skeleton_tangent(points: np.ndarray, radius: float) -> np.ndarray:
    """各点のまわりの中心線の向き（ラジアン）。流れが分からない場所の代わりに使う。"""
    angles = np.zeros(len(points))
    for i, p in enumerate(points):
        near = points[np.hypot(*(points - p).T) <= radius]
        if len(near) < 3:
            continue
        c = near - near.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        angles[i] = math.atan2(vt[0, 1], vt[0, 0]) % math.pi
    return angles


# ---------- 3〜4. 候補線 ----------


def clip_to_frame(p1: np.ndarray, p2: np.ndarray, x0: float, x1: float, y0: float, y1: float):
    """線分を長方形の中に切り詰める（Liang-Barsky 法）。はみ出し部分がすべてなら None。"""
    d = p2 - p1
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], p1[0] - x0), (d[0], x1 - p1[0]), (-d[1], p1[1] - y0), (d[1], y1 - p1[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return p1 + d * t0, p1 + d * t1


def flow_candidates(
    area: Area,
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    n_max: int,
    max_tilt: int = MAX_TILT_DEG,
) -> np.ndarray:
    """流れに直角（±max_tilt°）で、エリアの縁から縁まで引いた候補線。戻り値は (N, 4)。"""
    from .suggest import clip_line_to_polygon  # 循環 import を避ける

    poly = area.points()
    stations = centerline_points(area, width, height)
    if len(stations) == 0:
        return np.zeros((0, 4))
    cell, axis, strength = axis_field(tracks_xy, width, height)
    gx = np.clip((stations[:, 0] / cell).astype(int), 0, axis.shape[1] - 1)
    gy = np.clip((stations[:, 1] / cell).astype(int), 0, axis.shape[0] - 1)
    flow = axis[gy, gx]
    weak = strength[gy, gx] < 0.3
    if weak.any():  # 流れの向きがはっきりしない場所は、中心線の向きを流れの向きとみなす
        flow[weak] = _skeleton_tangent(stations, radius=3 * cell)[weak]
    bases = [flow + math.pi / 2]  # 流れに直角
    for extra in dominant_axes(tracks_xy, area, width, height)[1:]:
        # 交差点のように2方向の流れがあれば、2つ目の流れに直角な線も作る
        bases.append(np.full(len(stations), extra + math.pi / 2))

    tilts = np.radians(np.arange(-max_tilt, max_tilt + 1))
    per_station = len(bases) * len(tilts)
    if len(stations) * per_station > n_max:  # 本数の上限に合わせて立ち位置を等間隔に間引く
        keep = np.linspace(0, len(stations) - 1, max(1, n_max // per_station)).round().astype(int)
        stations, bases = stations[keep], [b[keep] for b in bases]

    edge = EDGE_RATIO * max(width, height)
    frame = (edge, width - 1 - edge, 0.0, height - 1 - edge)  # 上端は足元が切れないので除外しない
    lines = []
    for k, p in enumerate(stations):
        if not (frame[0] <= p[0] <= frame[1] and p[1] <= frame[3]):
            continue
        for base in bases:
            for tilt in tilts:
                a = base[k] + tilt
                seg = clip_line_to_polygon(p, np.array([math.cos(a), math.sin(a)]), poly)
                if seg is None:
                    continue
                seg = clip_to_frame(seg[0], seg[1], *frame)
                if seg is not None and np.hypot(*(seg[1] - seg[0])) >= 5:
                    lines.append(np.concatenate(seg))
    return np.array(lines, dtype=np.float64).reshape(-1, 4)


def snapped(line: np.ndarray, poly: np.ndarray, width: int, height: int, tol_deg: float = 3.0) -> np.ndarray | None:
    """水平・垂直から tol_deg° 以内の線を、水平・垂直にそろえた線（そろえる必要がなければ None）。"""
    from .suggest import clip_line_to_polygon

    p1, p2 = line[0:2], line[2:4]
    deg = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
    target = next((t for t in (0, 90, 180) if 0 < abs(deg - t) <= tol_deg), None)
    if target is None:
        return None
    a = math.radians(target)
    mid = (p1 + p2) / 2
    seg = clip_line_to_polygon(mid, np.array([math.cos(a), math.sin(a)]), poly)
    if seg is None:
        return None
    edge = EDGE_RATIO * max(width, height)
    seg = clip_to_frame(seg[0], seg[1], edge, width - 1 - edge, 0.0, height - 1 - edge)
    return None if seg is None else np.concatenate(seg)
