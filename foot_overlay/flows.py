"""出入口と人流から、カウント線の候補を作る（③ 候補の「出入口と人流から」方式）。

  1. 出入口: 全員の追跡線の「最初に検知した地点」と「最後に検知した地点」を集め、
     近いものどうしをまとめて（密度によるクラスタ分け）出入口とする
  2. 人流: 出入口 A で現れて出入口 B で消えた人を「A↔B の人流」に分ける。
     行き（A→B）と帰り（B→A）は同じ人流にまとめる（向きは IN/OUT で分かれる）
  3. 流れの向き: 人流の人の移動ベクトル（少し先の位置 − 少し前の位置）を、マスごとに
     A→B の向きにそろえて平均する。線の形ではなく、実際に動いた向きから求める
  4. 候補線: 人流の代表経路の上に立ち位置を並べ、そこでの流れの向きと min_angle 度以上で
     交わる向きの線を、人流の範囲の縁から縁まで引く。線全体で見ても流れと
     min_angle 度以上で交わるものだけ残す
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .flowlines import EDGE_RATIO, clip_to_frame

GATE_EPS_RATIO = 0.06  # 出入口としてまとめる距離（画像の長辺に対する割合）
GATE_ASSIGN_FACTOR = 2.0  # まとまりから外れた地点も、この倍の距離以内なら近い出入口に入れる
MIN_MOVE_RATIO = 0.05  # これより動かなかった人（立ち止まっていた人など）は人流に使わない
MIN_FLOW_SHARE = 0.03  # 動いた人のこの割合未満（かつ3人未満）の人流は出さない
MAX_FLOWS = 10
CELL_RATIO = 0.02  # 流れの向きを調べるマスの大きさ
STEP_RATIO = 0.006  # 代表経路の上に立ち位置を置く間隔
END_SKIP = 0.12  # 代表経路の両端（出入口のすぐ近く）のこの割合には線を引かない
ANGLE_STEP = 2  # 線の向きを変える刻み（度）
MIN_ANGLE = 60.0  # 流れとこの角度以上で交わる線だけを候補にする（画面で変えられる）
PATH_POINTS = 50


@dataclass
class Gateway:
    x: float
    y: float
    radius: float
    count: int  # この出入口で現れた・消えた回数


@dataclass
class Flow:
    a: int  # 出入口の番号（a < b）
    b: int
    track_ids: list[int] = field(default_factory=list)
    backward_ids: list[int] = field(default_factory=list)  # b → a の向きに歩いた人（残りは a → b）
    path: list[list[float]] = field(default_factory=list)  # 代表経路（a → b の向き）
    corridor: list[list[float]] = field(default_factory=list)  # 人流が通る範囲（多角形）

    @property
    def people(self) -> int:
        return len(self.track_ids)

    @property
    def forward(self) -> int:
        return self.people - len(self.backward_ids)

    @property
    def backward(self) -> int:
        return len(self.backward_ids)

    def oriented_paths(self, tracks_xy: dict[int, np.ndarray]) -> list[np.ndarray]:
        """人流の人の軌跡を、すべて a → b の向きにそろえて返す。"""
        back = set(self.backward_ids)
        return [tracks_xy[t][::-1] if t in back else tracks_xy[t] for t in self.track_ids if t in tracks_xy]

    @property
    def label(self) -> str:
        return f"G{self.a + 1}↔G{self.b + 1}"

    def corridor_points(self) -> np.ndarray:
        return np.asarray(self.corridor, dtype=np.float64).reshape(-1, 2)

    def contains(self, x: float, y: float) -> bool:
        if len(self.corridor) < 3:
            return False
        return cv2.pointPolygonTest(self.corridor_points().astype(np.float32), (float(x), float(y)), False) >= 0


@dataclass
class FlowSettings:
    min_angle: float = MIN_ANGLE

    @classmethod
    def from_dict(cls, d: dict | None) -> "FlowSettings":
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 1. 出入口 ----------


def _dbscan(points: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """密度によるクラスタ分け（DBSCAN）。近くに min_pts 個以上ある点を核にして広げる。-1 はどこにも入らない点。"""
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    # eps の大きさのマスに分けて、隣り合うマスの点だけ距離を比べる（点が多くても速い）
    keys = np.floor(points / eps).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for i, (kx, ky) in enumerate(keys):
        cells.setdefault((kx, ky), []).append(i)

    def neighbors(i: int) -> np.ndarray:
        kx, ky = keys[i]
        cand = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cells.get((kx + dx, ky + dy), ())]
        cand = np.array(cand, dtype=int)
        d = np.hypot(*(points[cand] - points[i]).T)
        return cand[d <= eps]

    nbrs = [neighbors(i) for i in range(n)]
    core = np.array([len(nb) >= min_pts for nb in nbrs])
    cluster = 0
    for i in range(n):
        if not core[i] or labels[i] >= 0:
            continue
        labels[i] = cluster
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in nbrs[j]:
                if labels[k] < 0:
                    labels[k] = cluster
                    stack.append(k)
        cluster += 1
    return labels


def _tip(xy: np.ndarray, at_end: bool) -> np.ndarray:
    k = min(3, len(xy))
    return xy[-k:].mean(axis=0) if at_end else xy[:k].mean(axis=0)


def find_gateways(
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    exclude: set[int] | None = None,
    cut_start: set[int] | None = None,
    cut_end: set[int] | None = None,
) -> tuple[list[Gateway], dict[int, tuple[int, int]]]:
    """出入口の一覧と、追跡IDごとの (現れた出入口, 消えた出入口)（どこにも入らなければ -1）。

    cut_start / cut_end: 動画の始まり・終わりで切れている線（録画の前から・後まで画面にいた人）。
    その端は出入口ではないので、出入口を探すのに使わない。
    """
    size = max(width, height)
    exclude = exclude or set()
    cut_start = cut_start or set()
    cut_end = cut_end or set()
    ids = [
        t for t, xy in tracks_xy.items()
        if t not in exclude and len(xy) >= 5 and np.hypot(*(xy[-1] - xy[0])) >= MIN_MOVE_RATIO * size
    ]
    if not ids:
        return [], {}
    pts = np.array([_tip(tracks_xy[t], False) for t in ids] + [_tip(tracks_xy[t], True) for t in ids])
    usable = np.array([t not in cut_start for t in ids] + [t not in cut_end for t in ids])
    eps = GATE_EPS_RATIO * size
    labels = np.full(len(pts), -1, dtype=int)
    labels[usable] = _dbscan(pts[usable], eps, max(3, round(0.03 * len(ids))))
    n_clusters = labels.max() + 1
    if n_clusters <= 0:
        return [], {t: (-1, -1) for t in ids}
    # 多い順に G1, G2, … と番号を付け直す
    counts = np.bincount(labels[labels >= 0], minlength=n_clusters)
    order = np.argsort(-counts, kind="stable")
    renum = np.empty(n_clusters, dtype=int)
    renum[order] = np.arange(n_clusters)
    labels = np.where(labels >= 0, renum[np.maximum(labels, 0)], -1)
    centers = np.array([pts[labels == g].mean(axis=0) for g in range(n_clusters)])
    # まとまりから少し外れた地点は、近い出入口に入れる
    for i in np.flatnonzero((labels < 0) & usable):
        d = np.hypot(*(centers - pts[i]).T)
        if d.min() <= GATE_ASSIGN_FACTOR * eps:
            labels[i] = int(np.argmin(d))
    gateways = []
    for g in range(n_clusters):
        members = pts[labels == g]
        radius = float(np.percentile(np.hypot(*(members - centers[g]).T), 90)) if len(members) else eps
        gateways.append(Gateway(float(centers[g][0]), float(centers[g][1]), max(radius, eps / 3), int(len(members))))
    n = len(ids)
    ends = {t: (int(labels[i]), int(labels[n + i])) for i, t in enumerate(ids)}
    return gateways, ends


# ---------- 2. 人流 ----------


def _resample(xy: np.ndarray, n: int) -> np.ndarray:
    """道のりに沿って等間隔の n 点に並べ直す。"""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(xy[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)


def _cell_grid(width: int, height: int) -> tuple[float, int, int]:
    cell = max(4.0, CELL_RATIO * max(width, height))
    return cell, int(math.ceil(width / cell)), int(math.ceil(height / cell))


def _corridor_mask(paths: list[np.ndarray], width: int, height: int, share: float = 0.1) -> np.ndarray:
    """paths の人が通ったマス（share の割合以上の人が通ったマス）を塗り、少し広げて穴をふさいだ範囲（0/1）。"""
    cell, gw, gh = _cell_grid(width, height)
    visits = np.zeros(gh * gw, np.int32)
    for xy in paths:
        gx = np.clip((xy[:, 0] / cell).astype(int), 0, gw - 1)
        gy = np.clip((xy[:, 1] / cell).astype(int), 0, gh - 1)
        np.add.at(visits, np.unique(gy * gw + gx), 1)
    need = max(1, math.ceil(share * len(paths)))
    grid = (visits.reshape(gh, gw) >= need).astype(np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.resize(grid, (gw * int(cell), gh * int(cell)), interpolation=cv2.INTER_NEAREST)[:height, :width]
    if mask.shape != (height, width):
        mask = cv2.copyMakeBorder(mask, 0, height - mask.shape[0], 0, width - mask.shape[1], cv2.BORDER_CONSTANT, 0)
    k = max(3, int(cell) | 1)
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


def walk_mask(tracks_xy: dict[int, np.ndarray], width: int, height: int) -> np.ndarray:
    """全員が歩く範囲（通路）。候補線は人流の向きに合わせて、この範囲の縁から縁まで引く。"""
    return _corridor_mask(list(tracks_xy.values()), width, height, share=0.02)


def region_polygon(mask: np.ndarray, path: list[list[float]], width: int, height: int) -> list[list[float]]:
    """mask のうち、代表経路の点を最も多く含むひと続きの範囲の輪郭。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    pts = np.array(path, np.float32)
    best = max(contours, key=lambda c: sum(cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in pts))
    return cv2.approxPolyDP(best, 0.004 * max(width, height), True).reshape(-1, 2).astype(float).tolist()


def find_flows(
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    exclude: set[int] | None = None,
    cut_start: set[int] | None = None,
    cut_end: set[int] | None = None,
) -> tuple[list[Gateway], list[Flow]]:
    """出入口と、出入口の組ごとの人流（人数の多い順、最大 MAX_FLOWS 件）。"""
    gateways, ends = find_gateways(tracks_xy, width, height, exclude, cut_start, cut_end)
    groups: dict[tuple[int, int], list[int]] = {}
    for t, (s, e) in ends.items():
        if s >= 0 and e >= 0 and s != e:
            groups.setdefault((min(s, e), max(s, e)), []).append(t)
    need = max(3, round(MIN_FLOW_SHARE * len(ends)))
    flows = []
    for (a, b), ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(ids) < need or len(flows) >= MAX_FLOWS:
            continue
        flow = Flow(a=a, b=b, track_ids=sorted(ids), backward_ids=sorted(t for t in ids if ends[t][0] == b))
        paths = flow.oriented_paths(tracks_xy)
        flow.path = np.median(np.stack([_resample(p, PATH_POINTS) for p in paths]), axis=0).round(1).tolist()
        # 代表経路の点を最も多く含む範囲を、その人流の範囲とする
        flow.corridor = region_polygon(_corridor_mask(paths, width, height), flow.path, width, height)
        if flow.corridor:
            flows.append(flow)
    return gateways, flows


# ---------- 3. 流れの向き（移動ベクトル） ----------


def direction_field(
    paths: list[np.ndarray], width: int, height: int, fps: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """マスごとの (移動ベクトルの和, 人数)。ベクトルは a → b の向きにそろえた単位ベクトルの和。"""
    cell, gw, gh = _cell_grid(width, height)
    vec = np.zeros((gh, gw, 2))
    cnt = np.zeros((gh, gw))
    k = max(1, round(0.25 * fps))  # 0.25秒前と後の位置の差（足元の細かな揺れを拾わないように）
    for xy in paths:
        if len(xy) <= 2 * k:
            continue
        d = xy[2 * k :] - xy[: -2 * k]
        mid = xy[k:-k]
        norm = np.hypot(*d.T)
        ok = norm > 1e-6
        if not ok.any():
            continue
        u = d[ok] / norm[ok, None]
        gx = np.clip((mid[ok, 0] / cell).astype(int), 0, gw - 1)
        gy = np.clip((mid[ok, 1] / cell).astype(int), 0, gh - 1)
        np.add.at(vec, (gy, gx), u)
        np.add.at(cnt, (gy, gx), 1)
    # となりのマスと少しならす（人の少ないマスの向きが乱れないように）
    vec = cv2.blur(vec.astype(np.float32), (3, 3))
    cnt = cv2.blur(cnt.astype(np.float32), (3, 3))
    return cell, vec, cnt


def _field_at(points: np.ndarray, cell: float, vec: np.ndarray, cnt: np.ndarray):
    gh, gw = cnt.shape
    gx = np.clip((points[:, 0] / cell).astype(int), 0, gw - 1)
    gy = np.clip((points[:, 1] / cell).astype(int), 0, gh - 1)
    return vec[gy, gx], cnt[gy, gx]


def crossing_angle(direction: np.ndarray, flow_vec: np.ndarray) -> np.ndarray:
    """線の向きと流れの向きがなす角（0〜90度）。"""
    n = np.hypot(*flow_vec.T) if flow_vec.ndim == 2 else np.hypot(*flow_vec)
    n = np.maximum(n, 1e-9)
    cross = np.abs(direction[0] * flow_vec[..., 1] - direction[1] * flow_vec[..., 0]) / n
    return np.degrees(np.arcsin(np.clip(cross, 0, 1)))


# ---------- 4. 候補線 ----------


def flow_candidates(
    flow: Flow,
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    fps: float,
    min_angle: float = MIN_ANGLE,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """人流と min_angle 度以上で交わる候補線 (N, 4)。

    線の長さは region（全員が歩く通路。walk_mask）の縁から縁まで。省略時は人流の範囲の縁から縁まで。
    """
    paths = flow.oriented_paths(tracks_xy)
    if not paths or len(flow.path) < 2:
        return np.zeros((0, 4))
    size = max(width, height)
    cell, vec, cnt = direction_field(paths, width, height, fps)
    if region is not None:
        mask = region
    else:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(flow.corridor_points()).astype(np.int32)], 1)

    path = np.array(flow.path, dtype=float)
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    step = max(3.0, STEP_RATIO * size)
    ts = np.arange(END_SKIP * s[-1], (1 - END_SKIP) * s[-1], step)
    centers = np.stack([np.interp(ts, s, path[:, 0]), np.interp(ts, s, path[:, 1])], axis=1)
    tangent = np.gradient(centers, axis=0) if len(centers) > 1 else np.zeros_like(centers)

    max_tilt = max(0.0, 90.0 - float(min_angle))
    tilts = np.radians(np.arange(-max_tilt, max_tilt + 1e-9, ANGLE_STEP))
    reach = np.arange(0, size, 1.5)  # 中心から縁まで、1.5px ずつ確かめる
    margin = EDGE_RATIO * size
    x0, x1, y0, y1 = margin, width - 1 - margin, margin, height - 1 - margin
    pad = 0.5 * cell  # 範囲の縁ぎりぎりを歩いた人も数えられるよう、少しはみ出して引く
    lines = []
    for c, tan in zip(centers, tangent):
        v, n = _field_at(c[None], cell, vec, cnt)
        d = v[0] if n[0] > 0 and np.hypot(*v[0]) > 1e-6 else tan
        if np.hypot(*d) < 1e-9:
            continue
        d = d / np.hypot(*d)
        normal = np.array([-d[1], d[0]])
        for a in tilts:
            ca, sa = math.cos(a), math.sin(a)
            u = np.array([normal[0] * ca - normal[1] * sa, normal[0] * sa + normal[1] * ca])
            tips = []
            for sign in (1, -1):
                p = c + sign * reach[:, None] * u
                inside = (p[:, 0] >= 0) & (p[:, 0] < width) & (p[:, 1] >= 0) & (p[:, 1] < height)
                px = np.clip(p[:, 0].astype(int), 0, width - 1)
                py = np.clip(p[:, 1].astype(int), 0, height - 1)
                ok = inside & (mask[py, px] > 0)
                stop = int(np.argmin(ok)) if not ok.all() else len(ok)
                tips.append(c + sign * (reach[max(stop - 1, 0)] + pad) * u)
            clipped = clip_to_frame(tips[0], tips[1], x0, x1, y0, y1)
            if clipped is None:
                continue
            p1, p2 = clipped
            length = float(np.hypot(*(p2 - p1)))
            if length < 0.02 * size:
                continue
            # 線全体で見ても、通る人の流れと min_angle 度以上で交わっているか（移動ベクトルで確かめる）
            samples = p1 + np.linspace(0.05, 0.95, 12)[:, None] * (p2 - p1)
            sv, sn = _field_at(samples, cell, vec, cnt)
            used = sn > 0
            if not used.any():
                continue
            angles = crossing_angle(u, sv[used])
            weight = sn[used]
            if float(np.sum(weight * (angles >= min_angle)) / np.sum(weight)) < 0.8:
                continue
            if crossing_angle(u, sv[used].sum(axis=0)) < min_angle:
                continue
            lines.append([p1[0], p1[1], p2[0], p2[1]])
    if not lines:
        return np.zeros((0, 4))
    arr = np.round(np.array(lines), 1)
    return np.unique(arr, axis=0)


# ---------- 保存 ----------


def save_flows(path: Path, gateways: list[Gateway], flows: list[Flow]) -> None:
    data = {"gateways": [asdict(g) for g in gateways], "flows": [asdict(f) for f in flows]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_flows(path: Path) -> tuple[list[Gateway], list[Flow]]:
    if not path.is_file():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        gateways = [Gateway(**g) for g in data.get("gateways", [])]
        flows = [Flow(**f) for f in data.get("flows", [])]
    except (ValueError, TypeError, KeyError):
        return [], []
    return gateways, flows
