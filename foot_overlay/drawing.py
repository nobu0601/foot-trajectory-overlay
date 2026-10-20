"""足元軌跡の整形と、最初のフレームへの重畳描画。"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .tracking import TrackPoint


def filter_short_tracks(tracks: dict[int, list[TrackPoint]], min_points: int) -> dict[int, list[TrackPoint]]:
    """検出回数が少ないID（誤検出・一瞬だけ映った人）を除く。"""
    return {tid: pts for tid, pts in tracks.items() if len(pts) >= min_points}


def smooth(points: list[TrackPoint], window: int) -> np.ndarray:
    """移動平均で足元座標のブレをならす。戻り値は (N, 2) の配列。"""
    xy = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    if window <= 1 or len(xy) < 3:
        return xy
    window = min(window, len(xy))
    pad = window // 2
    # 端は端の値で埋めて、始点・終点が内側へ縮まないようにする
    padded = np.pad(xy, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """IDごとに見分けやすい色（BGR）を決める。黄金比で色相をずらす。"""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_grid(img: np.ndarray, step: int) -> None:
    """カウント線の座標を読み取りやすいよう、座標付きの格子を描く。"""
    h, w = img.shape[:2]
    color = (200, 200, 200)
    for x in range(step, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for y in range(step, h, step):
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


SUSPECT_COLOR = (150, 150, 150)  # 誤判定らしい追跡線の色（灰色）


def draw_dashed_polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int, dash: int) -> None:
    """折れ線を点線で描く（道のりに沿って dash px 描いて dash px あける）。"""
    if len(pts) < 2:
        return
    seg = np.diff(pts.astype(np.float64), axis=0)
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    on = (dist // dash) % 2 == 0
    for i in range(len(pts) - 1):
        if on[i]:
            cv2.line(img, tuple(int(v) for v in pts[i]), tuple(int(v) for v in pts[i + 1]), color, thickness, cv2.LINE_AA)


def default_line_width(width: int, height: int) -> int:
    """画像サイズに合わせた追跡線の標準の太さ。"""
    return max(1, round(1.5 * max(width, height) / 1280))


def outline_width(thickness: int) -> int:
    """追跡線の黒い縁取りの太さ。細い線では縁取りをしない（線が太って見えないように）。"""
    if thickness <= 1:
        return 0
    return thickness + (1 if thickness <= 3 else 2)


def draw_overlay(
    first_frame: np.ndarray,
    tracks: dict[int, list[TrackPoint]],
    smooth_window: int = 5,
    dim: float = 0.35,
    show_ids: bool = True,
    grid: int = 0,
    line_width: int | None = None,
    suspects: frozenset[int] | set[int] = frozenset(),
) -> np.ndarray:
    """最初のフレームに全員の足元軌跡を重ねた画像を返す。

    - 線: 足元の通り道（IDごとに色分け）
    - 丸: 追跡の開始位置
    - 矢印: 追跡の終了位置と進行方向

    line_width を省略すると、画像サイズに合った太さで描く。
    """
    img = first_frame.copy()
    if dim > 0:
        # 背景を少し暗くして線を目立たせる
        img = cv2.addWeighted(img, 1.0 - dim, np.zeros_like(img), dim, 0)
    if grid > 0:
        draw_grid(img, grid)

    h, w = img.shape[:2]
    scale = max(w, h) / 1280
    thickness = line_width if line_width else default_line_width(w, h)
    radius = max(2, thickness + 1)
    font_scale = 0.5 * max(scale, 0.8)
    text_th = max(1, round(1.5 * max(scale, 0.8)))  # 文字の太さは追跡線の太さと切り離す

    for tid in sorted(tracks):
        xy = smooth(tracks[tid], smooth_window)
        pts = np.round(xy).astype(np.int32)
        if tid in suspects:
            # 誤判定らしい線は灰色の点線で、色付きの線と区別する
            draw_dashed_polyline(img, pts, SUSPECT_COLOR, thickness, dash=max(6, 4 * thickness))
            if show_ids:
                cv2.putText(img, str(tid), (int(pts[0][0]) + 4, int(pts[0][1]) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, SUSPECT_COLOR, text_th, cv2.LINE_AA)
            continue
        color = color_for_id(tid)

        # 黒い縁取りを先に描いて、明るい背景でも線が見えるようにする（太さ1では縁取りなし）
        outline = outline_width(thickness)
        if outline:
            cv2.polylines(img, [pts], False, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

        start = tuple(int(v) for v in pts[0])
        if outline:
            cv2.circle(img, start, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, start, radius, color, -1, cv2.LINE_AA)

        _draw_end_arrow(img, pts, color, thickness, radius)

        if show_ids:
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_th + 2)
            # 画面端で番号が切れないよう、画像内に収める
            org = (
                min(max(start[0] + radius + 2, 0), w - tw - 2),
                min(max(start[1] - radius - 2, th + 2), h - 2),
            )
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_th + 2, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_th, cv2.LINE_AA)

    return img


def _draw_end_arrow(img, pts: np.ndarray, color, thickness: int, radius: int) -> None:
    """終点に、直近の移動方向を向いた矢印を描く。"""
    end = pts[-1]
    min_len = max(10, radius * 3)  # 細い線でも矢印の向きが分かる長さにする
    # 終点から十分離れた点まで遡って方向を決める（短すぎると向きが不安定）
    for prev in pts[-2::-1]:
        if np.hypot(*(end - prev)) >= min_len:
            direction = (end - prev) / np.hypot(*(end - prev))
            tail = end - direction * min_len
            p1 = tuple(int(v) for v in np.round(tail))
            p2 = tuple(int(v) for v in end)
            outline = outline_width(thickness)
            if outline:
                cv2.arrowedLine(img, p1, p2, (0, 0, 0), outline, cv2.LINE_AA, tipLength=0.6)
            cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.6)
            return
    # ほとんど動いていない場合は四角で終点を示す
    x, y = int(end[0]), int(end[1])
    cv2.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, -1)
