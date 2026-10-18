"""1つの結果フォルダ（動画名_overlay/）を扱う。画面にもコマンドにも依存しない部分。

結果フォルダの中身:
  first_frame.png   最初のフレーム（下絵）
  tracks.csv        足元座標
  meta.json         動画情報と表示設定
  overlay.png       足元軌跡の重畳画像
  lines.json        カウント線
  counts.csv        線ごとの通過人数
  crossings.csv     1回ごとの通過記録
  count_result.png  重畳画像＋カウント線＋人数
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .counting import (
    DEFAULT_SMOOTH,
    CountLine,
    Crossing,
    LineCount,
    count_lines,
    estimate_fps,
    load_lines,
    load_meta,
    load_tracks_csv,
    save_counts_csv,
    save_crossings_csv,
    save_lines,
)
from .drawing import default_line_width, draw_overlay
from .export import save_image, save_tracks_csv
from .tracking import TrackingResult, TrackPoint

FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_LINE_PX = 5

# カウント線の色（BGR）。軌跡の色と区別しやすい明るい色
LINE_COLORS_BGR = [
    (0, 255, 255),  # 黄
    (255, 255, 255),  # 白
    (255, 255, 0),  # 水色
    (255, 0, 255),  # 桃
    (0, 165, 255),  # 橙
    (0, 255, 0),  # 緑
]


def line_color(i: int) -> tuple[int, int, int]:
    return LINE_COLORS_BGR[i % len(LINE_COLORS_BGR)]


def write_result(
    out_dir: Path,
    video: Path,
    result: TrackingResult,
    tracks: dict[int, list[TrackPoint]],
    smooth: int,
    min_points: int,
    view: dict | None = None,
) -> None:
    """追跡結果を結果フォルダに書き出す（overlay.png は Session.save で描く）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(out_dir / "first_frame.png", result.first_frame)
    save_tracks_csv(out_dir / "tracks.csv", tracks, result.fps)
    h, w = result.first_frame.shape[:2]
    meta = {
        "video": video.name,
        "width": w,
        "height": h,
        "fps": result.fps,
        "frames": result.total_frames,
        "smooth": smooth,
        "min_points": min_points,
        "view": view or {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # 前回のカウント結果が残っていると混乱するので消す
    for name in ("lines.json", "counts.csv", "crossings.csv", "count_result.png"):
        (out_dir / name).unlink(missing_ok=True)


def read_image(path: Path) -> np.ndarray:
    """日本語パスでも読めるよう、バイト列から画像を読む。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"画像を読み込めませんでした: {path}")
    return img


class Session:
    """結果フォルダに対する、表示設定とカウント線の編集状態。"""

    def __init__(self, result_dir: Path):
        self.dir = result_dir
        tracks_path = result_dir / "tracks.csv"
        frame_path = result_dir / "first_frame.png"
        if not tracks_path.is_file() or not frame_path.is_file():
            raise FileNotFoundError(f"解析結果（tracks.csv / first_frame.png）が見つかりません: {result_dir}")
        self.first_frame = read_image(frame_path)
        self.meta = load_meta(result_dir / "meta.json")
        self.fps = float(self.meta.get("fps") or estimate_fps(tracks_path))
        self.smooth = int(self.meta.get("smooth", DEFAULT_SMOOTH))
        self.tracks = load_tracks_csv(tracks_path)

        h, w = self.first_frame.shape[:2]
        self.width, self.height = w, h
        self.scale = max(w, h) / 1280
        self.margin = max(3.0, 0.004 * max(w, h))

        view = self.meta.get("view", {})
        self.line_width: int = int(view.get("line_width") or default_line_width(w, h))
        self.show_ids: bool = bool(view.get("show_ids", True))
        self.grid: int = int(view.get("grid", 100))
        self.dim: float = float(view.get("dim", 0.35))
        self._overlay_cache: tuple[tuple, np.ndarray] | None = None

        self.lines: list[CountLine] = load_lines(result_dir / "lines.json")
        self.counts: list[LineCount] = []
        self.crossings: list[Crossing] = []
        self.recount()

    @property
    def video_name(self) -> str:
        return self.meta.get("video", self.dir.name)

    # ---- カウント線 ----

    def recount(self) -> None:
        self.counts, self.crossings = count_lines(self.tracks, self.lines, self.smooth, self.margin)

    def add_line(self, line: CountLine) -> bool:
        if line.length() < MIN_LINE_PX:
            return False
        self.lines.append(line)
        self.recount()
        return True

    def remove_line(self, index: int) -> None:
        if 0 <= index < len(self.lines):
            self.lines.pop(index)
            self.recount()

    def undo(self) -> None:
        self.remove_line(len(self.lines) - 1)

    def clear(self) -> None:
        self.lines.clear()
        self.recount()

    def set_lines(self, lines: list[CountLine]) -> None:
        self.lines = list(lines)
        self.recount()

    # ---- 描画 ----

    def overlay(self) -> np.ndarray:
        """足元軌跡の重畳画像（表示設定が変わったときだけ描き直す）。"""
        key = (self.line_width, self.show_ids, self.grid, self.dim)
        if self._overlay_cache is None or self._overlay_cache[0] != key:
            img = draw_overlay(
                self.first_frame,
                self.tracks,
                smooth_window=self.smooth,
                dim=self.dim,
                show_ids=self.show_ids,
                grid=self.grid,
                line_width=self.line_width,
            )
            self._overlay_cache = (key, img)
        return self._overlay_cache[1]

    def render(self) -> np.ndarray:
        """重畳画像にカウント線と通過位置を描いた画像。"""
        img = self.overlay().copy()
        thick = max(2, round(3 * self.scale))
        for i, line in enumerate(self.lines):
            draw_count_line(img, line, f"L{i + 1}", line_color(i), thick, self.scale)
        # 通過した位置の印（線の上に描いて、どこで数えたか見えるようにする）
        for c in self.crossings:
            center = (round(c.x), round(c.y))
            cv2.circle(img, center, thick + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, center, thick, (0, 0, 255), -1, cv2.LINE_AA)
        return img

    def render_with_summary(self) -> np.ndarray:
        """保存用: 画像の下に線ごとの人数を書き足す。"""
        img = self.render()
        row_h = max(20, round(26 * self.scale))
        font_scale = max(0.45, 0.6 * self.scale)
        rows = [(f"{self.video_name}", (220, 220, 220))] if self.video_name.isascii() else []
        if not self.lines:
            rows.append(("No count lines", (200, 200, 200)))
        for i, c in enumerate(self.counts):
            rows.append((f"L{i + 1}   A->B: {c.a_to_b}   B->A: {c.b_to_a}   Total: {c.total}", line_color(i)))
        footer = np.full((row_h * (len(rows) + 1) - row_h // 2, self.width, 3), 32, dtype=np.uint8)
        for r, (text, color) in enumerate(rows):
            cv2.putText(footer, text, (10, row_h * (r + 1) - row_h // 4), FONT, font_scale, color, 1, cv2.LINE_AA)
        return np.vstack([img, footer])

    # ---- 保存 ----

    def save(self) -> None:
        self.meta["view"] = {
            "line_width": self.line_width,
            "show_ids": self.show_ids,
            "grid": self.grid,
            "dim": self.dim,
        }
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        save_image(self.dir / "overlay.png", self.overlay())
        save_lines(self.dir / "lines.json", self.lines)
        save_counts_csv(self.dir / "counts.csv", self.lines, self.counts)
        save_crossings_csv(self.dir / "crossings.csv", self.crossings, self.fps)
        save_image(self.dir / "count_result.png", self.render_with_summary())

    def summary_lines(self) -> list[str]:
        if not self.lines:
            return ["カウント線: なし"]
        return [f"L{i}: A→B {c.a_to_b}人 / B→A {c.b_to_a}人 / 合計 {c.total}人" for i, c in enumerate(self.counts, 1)]


def draw_count_line(img: np.ndarray, line: CountLine, label: str, color, thick: int, scale: float) -> None:
    """カウント線と、A側・B側の目印、A→B の向きの矢印を描く。"""
    p1 = (round(line.x1), round(line.y1))
    p2 = (round(line.x2), round(line.y2))
    cv2.line(img, p1, p2, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, thick + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p, thick + 1, color, -1, cv2.LINE_AA)

    mid = (line.p1 + line.p2) / 2
    n = line.normal()
    off = max(18.0, 22 * scale)
    tail = tuple(int(v) for v in np.round(mid - n * off * 0.55))
    head = tuple(int(v) for v in np.round(mid + n * off * 0.55))
    cv2.arrowedLine(img, tail, head, (0, 0, 0), thick + 2, cv2.LINE_AA, tipLength=0.45)
    cv2.arrowedLine(img, tail, head, color, max(1, thick - 1), cv2.LINE_AA, tipLength=0.45)

    font_scale = max(0.5, 0.6 * scale)
    _put_centered(img, "A", mid - n * off, font_scale, color, thick)
    _put_centered(img, "B", mid + n * off, font_scale, color, thick)
    # 線の番号は始点の外側に置く
    d = line.p1 - line.p2
    d = d / (np.hypot(*d) or 1)
    _put_centered(img, label, line.p1 + d * off, font_scale, color, thick)


def _put_centered(img, text: str, pos: np.ndarray, font_scale: float, color, thick: int) -> None:
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thick)
    x = int(min(max(pos[0] - tw / 2, 2), w - tw - 2))
    y = int(min(max(pos[1] + th / 2, th + 2), h - 2))
    cv2.putText(img, text, (x, y), FONT, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, max(1, thick - 1), cv2.LINE_AA)
