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
from dataclasses import asdict
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
from .areas import AREA_KINDS, Area, detect_areas, load_areas, people_in_area, save_areas
from .drawing import default_line_width, draw_overlay, smooth
from .export import save_image, save_tracks_csv
from .suggest import DEFAULT_CANDIDATES, Suggestion, save_suggestions_csv, suggest_lines
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


AREA_COLOR = (255, 200, 80)  # 水色系（BGR）
AREA_HIGHLIGHT = (80, 255, 255)
SUGGEST_COLOR = (60, 255, 60)  # 緑
SUGGEST_SELECTED = (255, 255, 255)


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
    for name in (
        "lines.json",
        "counts.csv",
        "crossings.csv",
        "count_result.png",
        "areas.json",
        "suggestions.json",
        "suggestions.csv",
        "suggestions.png",
    ):
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
        # 足元の位置の細かな揺れで二重に数えないよう、線からこの距離以上離れて初めて「渡った」とみなす
        self.margin = max(3.0, 0.01 * max(w, h))

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

        self.areas: list[Area] = load_areas(result_dir / "areas.json")
        self.suggestions: list[Suggestion] = load_suggestions(result_dir / "suggestions.json")
        self.show_areas = True
        self._tracks_xy: dict[int, np.ndarray] | None = None

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

    # ---- エリア・候補提案 ----

    @property
    def tracks_xy(self) -> dict[int, np.ndarray]:
        """移動平均した足元座標（重畳画像・カウントと同じもの）。"""
        if self._tracks_xy is None:
            self._tracks_xy = {tid: smooth(pts, self.smooth) for tid, pts in self.tracks.items()}
        return self._tracks_xy

    def auto_detect_areas(self) -> None:
        self.areas = detect_areas(self.tracks_xy, self.width, self.height)
        self.suggestions = []

    def area_people(self, index: int) -> int:
        return people_in_area(self.areas[index], self.tracks_xy, self.width, self.height)

    def add_area(self, polygon: list[list[float]], kind: str = "passage") -> bool:
        area = Area(kind=kind, polygon=[[float(x), float(y)] for x, y in polygon])
        if len(polygon) < 3 or area.polygon_area() < 25:
            return False
        self.areas.append(area)
        return True

    def remove_area(self, index: int) -> None:
        if 0 <= index < len(self.areas):
            self.areas.pop(index)
            # 消したエリアの候補を除き、後ろのエリア番号を詰める
            kept = []
            for sg in self.suggestions:
                if sg.area_index == index:
                    continue
                if sg.area_index > index:
                    sg.area_index -= 1
                kept.append(sg)
            self.suggestions = kept

    def area_changed(self, index: int) -> None:
        """エリアの形が変わったら、そのエリアの古い候補は消す。"""
        self.suggestions = [sg for sg in self.suggestions if sg.area_index != index]

    def compute_suggestions(
        self,
        n_candidates: int = DEFAULT_CANDIDATES,
        top_k: int = 3,
        progress=None,
        should_stop=None,
    ) -> list[Suggestion]:
        """全エリアについて候補線を計算して返す（self には反映しない。別スレッドから呼べる）。

        progress(エリア番号, 評価済み本数, 全本数) で進み具合を通知する。
        """
        results: list[Suggestion] = []
        for i, area in enumerate(list(self.areas)):
            if should_stop is not None and should_stop():
                break
            people = self.area_people(i)

            def area_progress(done, total, i=i):
                if progress is not None:
                    progress(i, done, total)
                if should_stop is not None and should_stop():
                    raise SuggestionCancelled()

            results += suggest_lines(
                area,
                i,
                self.tracks,
                self.width,
                self.height,
                people,
                smooth_window=self.smooth,
                margin=self.margin,
                fps=self.fps,
                n_candidates=n_candidates,
                top_k=top_k,
                progress=area_progress,
            )
        return results

    def adopt_suggestion(self, index: int) -> bool:
        """候補をカウント線として追加する。"""
        if not 0 <= index < len(self.suggestions):
            return False
        return self.add_line(self.suggestions[index].line)

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

    def render(self, highlight_area: int | None = None, selected_suggestion: int | None = None) -> np.ndarray:
        """重畳画像に、エリア・候補線・カウント線・通過位置を描いた画像。"""
        img = self.overlay().copy()
        thick = max(2, round(3 * self.scale))
        if self.show_areas:
            self.draw_areas(img, highlight=highlight_area)
            self.draw_suggestions(img, thick, selected=selected_suggestion)
        for i, line in enumerate(self.lines):
            draw_count_line(img, line, f"L{i + 1}", line_color(i), thick, self.scale)
        # 通過した位置の印（線の上に描いて、どこで数えたか見えるようにする）
        for c in self.crossings:
            center = (round(c.x), round(c.y))
            cv2.circle(img, center, thick + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, center, thick, (0, 0, 255), -1, cv2.LINE_AA)
        return img

    def draw_areas(self, img: np.ndarray, highlight: int | None = None) -> None:
        """エリアを半透明で塗り、輪郭と番号を描く。"""
        if not self.areas:
            return
        fill = img.copy()
        for i, area in enumerate(self.areas):
            if len(area.polygon) >= 3:
                cv2.fillPoly(fill, [np.round(area.points()).astype(np.int32)], AREA_COLOR)
        cv2.addWeighted(fill, 0.18, img, 0.82, 0, dst=img)
        font_scale = max(0.5, 0.6 * self.scale)
        for i, area in enumerate(self.areas):
            pts = np.round(area.points()).astype(np.int32)
            if len(pts) < 2:
                continue
            color = AREA_HIGHLIGHT if i == highlight else AREA_COLOR
            width = max(2, round(2 * self.scale)) + (1 if i == highlight else 0)
            cv2.polylines(img, [pts], True, (0, 0, 0), width + 2, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, color, width, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), width + 2, color, -1, cv2.LINE_AA)
            label = f"E{i + 1} {area.kind}"
            top = pts[np.argmin(pts[:, 1])]
            _put_centered(img, label, top + np.array([0, -14 * max(self.scale, 1)]), font_scale, color, 2)

    def draw_suggestions(self, img: np.ndarray, thick: int, selected: int | None = None) -> None:
        """カウント線の候補を点線で描く。"""
        font_scale = max(0.45, 0.55 * self.scale)
        for k, sg in enumerate(self.suggestions):
            color = SUGGEST_SELECTED if k == selected else SUGGEST_COLOR
            ln = sg.line
            p1, p2 = np.array([ln.x1, ln.y1], float), np.array([ln.x2, ln.y2], float)
            _dashed_line(img, p1, p2, color, thick + (1 if k == selected else 0), dash=max(8, round(12 * self.scale)))
            mid = (p1 + p2) / 2
            _put_centered(img, f"E{sg.area_index + 1}-{sg.rank}", mid + np.array([0, -12 * max(self.scale, 1)]), font_scale, color, 2)

    def render_suggestions_image(self) -> np.ndarray:
        """保存用: エリアと候補線と各候補の人数。"""
        img = self.overlay().copy()
        thick = max(2, round(3 * self.scale))
        self.draw_areas(img)
        self.draw_suggestions(img, thick)
        rows = []
        for sg in self.suggestions:
            kind = self.areas[sg.area_index].kind if sg.area_index < len(self.areas) else ""
            rows.append(
                (
                    f"E{sg.area_index + 1}-{sg.rank} ({kind})  people: {sg.people}/{sg.area_people}"
                    f" ({sg.coverage:.0%})  duplicates: {sg.duplicates}",
                    SUGGEST_COLOR,
                )
            )
        return self._with_footer(img, rows or [("No suggestions", (200, 200, 200))])

    def _with_footer(self, img: np.ndarray, rows: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
        row_h = max(20, round(26 * self.scale))
        font_scale = max(0.45, 0.6 * self.scale)
        footer = np.full((row_h * (len(rows) + 1) - row_h // 2, self.width, 3), 32, dtype=np.uint8)
        for r, (text, color) in enumerate(rows):
            cv2.putText(footer, text, (10, row_h * (r + 1) - row_h // 4), FONT, font_scale, color, 1, cv2.LINE_AA)
        return np.vstack([img, footer])

    def render_with_summary(self) -> np.ndarray:
        """保存用: 画像の下に線ごとの人数を書き足す。"""
        show = self.show_areas
        self.show_areas = False  # カウント結果の画像にはエリアと候補を描かない
        try:
            img = self.render()
        finally:
            self.show_areas = show
        rows = [(f"{self.video_name}", (220, 220, 220))] if self.video_name.isascii() else []
        if not self.lines:
            rows.append(("No count lines", (200, 200, 200)))
        for i, c in enumerate(self.counts):
            rows.append((f"L{i + 1}   A->B: {c.a_to_b}   B->A: {c.b_to_a}   Total: {c.total}", line_color(i)))
        return self._with_footer(img, rows)

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
        save_areas(self.dir / "areas.json", self.areas)
        save_suggestions(self.dir / "suggestions.json", self.suggestions)
        if self.suggestions:
            save_suggestions_csv(self.dir / "suggestions.csv", self.suggestions, self.areas)
            save_image(self.dir / "suggestions.png", self.render_suggestions_image())
        else:
            (self.dir / "suggestions.csv").unlink(missing_ok=True)
            (self.dir / "suggestions.png").unlink(missing_ok=True)

    def summary_lines(self) -> list[str]:
        if not self.lines:
            return ["カウント線: なし"]
        return [f"L{i}: A→B {c.a_to_b}人 / B→A {c.b_to_a}人 / 合計 {c.total}人" for i, c in enumerate(self.counts, 1)]


class SuggestionCancelled(Exception):
    """候補線の計算を利用者が中止した。"""


def load_suggestions(path: Path) -> list[Suggestion]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Suggestion(
            area_index=d["area_index"],
            rank=d["rank"],
            line=CountLine(**d["line"]),
            people=d["people"],
            duplicates=d["duplicates"],
            area_people=d["area_people"],
            passes=d.get("passes", 0),
        )
        for d in data.get("suggestions", [])
    ]


def save_suggestions(path: Path, suggestions: list[Suggestion]) -> None:
    data = {"suggestions": [asdict(sg) for sg in suggestions]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dashed_line(img, p1: np.ndarray, p2: np.ndarray, color, thick: int, dash: int) -> None:
    length = float(np.hypot(*(p2 - p1)))
    if length < 1:
        return
    d = (p2 - p1) / length
    for start in np.arange(0, length, dash * 2):
        a = p1 + d * start
        b = p1 + d * min(start + dash, length)
        pa, pb = tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b))
        cv2.line(img, pa, pb, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)


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
