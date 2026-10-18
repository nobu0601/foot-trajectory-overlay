"""エスカレーター・階段・通路などの「エリア」。

画像だけからエスカレーターや階段を見分けることはせず、人の足元軌跡が集中している
場所をエリアとして自動で見つける。種類（エスカレーター等）は利用者が選び、形も手で直せる。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

AREA_KINDS = {
    "escalator": "エスカレーター",
    "stairs": "階段",
    "passage": "通路",
    "other": "その他",
}


@dataclass
class Area:
    kind: str = "passage"
    polygon: list[list[float]] = field(default_factory=list)  # [[x, y], ...]

    @property
    def kind_label(self) -> str:
        return AREA_KINDS.get(self.kind, self.kind)

    def points(self) -> np.ndarray:
        return np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)

    def contains(self, x: float, y: float) -> bool:
        if len(self.polygon) < 3:
            return False
        return cv2.pointPolygonTest(self.points().astype(np.float32), (float(x), float(y)), False) >= 0

    def polygon_area(self) -> float:
        if len(self.polygon) < 3:
            return 0.0
        return float(abs(cv2.contourArea(self.points().astype(np.float32))))

    def mask(self, width: int, height: int) -> np.ndarray:
        m = np.zeros((height, width), dtype=np.uint8)
        if len(self.polygon) >= 3:
            cv2.fillPoly(m, [np.round(self.points()).astype(np.int32)], 1)
        return m


def people_in_area(area: Area, tracks_xy: dict[int, np.ndarray], width: int, height: int) -> int:
    """エリアの中を1回でも通った人（追跡ID）の数。"""
    m = area.mask(width, height)
    n = 0
    for xy in tracks_xy.values():
        pts = np.round(xy).astype(int)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        pts = pts[inside]
        if len(pts) and m[pts[:, 1], pts[:, 0]].any():
            n += 1
    return n


def detect_areas(
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    max_areas: int = 6,
) -> list[Area]:
    """足元軌跡が集中している場所をエリアとして見つける。

    1. 軌跡を太めの線で描き、画素ごとに「何人が通ったか」を数える
    2. 一定人数以上が通った画素をつなげて塊にする
    3. 塊の輪郭を多角形にする（通った人が多い順に max_areas 個まで）
    """
    if not tracks_xy:
        return []
    down = max(1, round(max(width, height) / 400))  # 計算を軽くするため縮小して数える
    gw, gh = max(1, width // down), max(1, height // down)
    brush = max(3, round(0.03 * max(gw, gh)))  # 足元の位置のばらつきを吸収する太さ

    density = np.zeros((gh, gw), dtype=np.int32)
    for xy in tracks_xy.values():
        canvas = np.zeros((gh, gw), dtype=np.uint8)
        pts = np.round(xy / down).astype(np.int32)
        cv2.polylines(canvas, [pts], False, 1, brush)
        density += canvas

    n_tracks = len(tracks_xy)
    # 全体の5%以上（最低1人）が通った画素を「通り道」とみなす。
    # 人が少ない動画では通った場所全体、人が多い動画では主な流れだけがエリアになる
    threshold = max(1, round(0.05 * n_tracks))
    mask = (density >= threshold).astype(np.uint8)
    # 隙間を埋めてから、通り道の幅より小さいトゲだけを削る（細い通路まで消さないように）
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (brush * 2 + 1, brush * 2 + 1))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (brush // 2 * 2 + 1, brush // 2 * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_pixels = 0.003 * gw * gh
    candidates: list[tuple[int, Area]] = []
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_pixels:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        while len(poly) > 16:  # 手で直しやすいよう頂点数を抑える
            eps *= 1.5
            poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        polygon = [[float(min(x * down, width - 1)), float(min(y * down, height - 1))] for x, y in poly]
        people = int(density[labels == label].max())
        if people < max(2, 0.1 * n_tracks):
            continue  # ほとんど人が通らない場所はエリアにしない
        candidates.append((people, Area(kind="passage", polygon=polygon)))

    candidates.sort(key=lambda c: -c[0])
    return [a for _, a in candidates[:max_areas]]


def load_areas(path: Path) -> list[Area]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Area(kind=a.get("kind", "passage"), polygon=a.get("polygon", [])) for a in data.get("areas", [])]


def save_areas(path: Path, areas: list[Area]) -> None:
    data = {"areas": [asdict(a) for a in areas]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
