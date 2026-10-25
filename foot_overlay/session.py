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
    export_lines_json,
    import_lines_json,
    load_lines,
    load_meta,
    load_tracks_csv,
    save_counts_csv,
    save_crossings_csv,
    save_lines,
)
from .areas import AREA_KINDS, Area, detect_areas, load_areas, people_in_area, save_areas
from .drawing import default_line_width, draw_overlay, filter_short_tracks, smooth
from .evaluate import REPEAT_SEC, LineEvaluation, evaluate_line
from .export import save_image, save_tracks_csv
from .suspects import SuspectSettings, find_suspects
from .textdraw import put_texts, text_size
from .suggest import DEFAULT_CANDIDATES, Suggestion, save_suggestions_csv, suggest_lines
from .tracking import TrackingResult, TrackPoint, split_at_frame_edges

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
        "count_lines.json",
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
        h, w = self.first_frame.shape[:2]
        self.width, self.height = w, h
        # 足元が画面外に出た点は使わない（古い解析結果にも効くよう、読み込み時に処理する）
        min_points = int(self.meta.get("min_points", 10))
        self.tracks = filter_short_tracks(split_at_frame_edges(load_tracks_csv(tracks_path), w, h), min_points)
        self.scale = max(w, h) / 1280
        # 足元の位置の細かな揺れで二重に数えないよう、線からこの距離以上離れて初めて「渡った」とみなす
        self.margin = max(3.0, 0.01 * max(w, h))

        view = self.meta.get("view", {})
        self.line_width: int = int(view.get("line_width") or default_line_width(w, h))
        self.show_ids: bool = bool(view.get("show_ids", True))
        self.grid: int = int(view.get("grid", 100))
        self.dim: float = float(view.get("dim", 0.35))
        # 表示用（絞り込みあり）と保存用（全員）の2枚を別々に覚えておく
        self._overlay_cache: dict[bool, tuple[tuple, np.ndarray]] = {}

        self._tracks_xy: dict[int, np.ndarray] | None = None
        # 誤判定らしい追跡線。利用者が「誤判定ではない」と戻したIDは suspect_ok に入る
        self._suspects_found: dict[int, list[str]] | None = None
        self.suspect_ok: set[int] = {int(t) for t in self.meta.get("suspect_ok", [])}
        self.suspect_settings = SuspectSettings.from_dict(self.meta.get("suspect_settings"))
        # 同じ人がこの秒数以内に線をまたぎ直したら「重複」とみなす（画面で変えられる）
        self.repeat_sec: float = float(self.meta.get("repeat_sec", REPEAT_SEC))

        self.lines: list[CountLine] = load_lines(result_dir / "lines.json")
        self.counts: list[LineCount] = []
        self.crossings: list[Crossing] = []
        self.evaluations: list[LineEvaluation] = []
        self.recount()

        self.areas: list[Area] = load_areas(result_dir / "areas.json")
        self.suggestions: list[Suggestion] = load_suggestions(result_dir / "suggestions.json")
        self.show_areas = True
        if self.areas and self.lines:
            self.recount()  # エリアが分かったので、すり抜けをエリアの縁まで見て評価し直す

        # 表示する時間帯（秒）。表示だけの絞り込みで、人数の集計や候補提案には使わない
        self.time_start: float = 0.0
        self.time_end: float = self.duration

    @property
    def video_name(self) -> str:
        return self.meta.get("video", self.dir.name)

    @property
    def duration(self) -> float:
        """動画の長さ（秒）。"""
        last = max((pts[-1].frame for pts in self.tracks.values() if pts), default=0)
        frames = max(int(self.meta.get("frames") or 0), last + 1)
        return frames / self.fps if self.fps else 0.0

    def is_time_filtered(self) -> bool:
        return self.time_start > 0 or self.time_end < self.duration - 1e-6

    def visible_tracks(self) -> dict[int, list[TrackPoint]]:
        """表示する時間帯に映っていた人の追跡線（線は途中で切らず、その人の全体を描く）。"""
        if not self.is_time_filtered():
            return self.tracks
        f0, f1 = self.time_start * self.fps, self.time_end * self.fps
        return {tid: pts for tid, pts in self.tracks.items() if pts and pts[0].frame <= f1 and pts[-1].frame >= f0}

    # ---- カウント線 ----

    def recount(self) -> None:
        self.counts, self.crossings = count_lines(self.tracks, self.lines, self.smooth, self.margin)
        self.evaluations = [self.evaluate(line) for line in self.lines]

    def evaluate(self, line: CountLine) -> LineEvaluation:
        """カウント線の信頼度・推定範囲（evaluate.py）。"""
        mid = (line.p1 + line.p2) / 2
        area = next((a for a in getattr(self, "areas", []) if a.contains(*mid)), None)
        return evaluate_line(
            line,
            self.tracks,
            self.tracks_xy,
            self.width,
            self.height,
            self.fps,
            self.margin,
            suspects=self.suspects,
            area_polygon=area.points() if area is not None else None,
            repeat_sec=self.repeat_sec,
        )

    # ---- 誤判定らしい追跡線 ----

    @property
    def suspects_found(self) -> dict[int, list[str]]:
        """検知した誤判定らしい追跡ID → 理由（利用者が戻したものも含む）。"""
        if self._suspects_found is None:
            self._suspects_found = find_suspects(
                self.tracks_xy, self.width, self.height, self.fps, self.suspect_settings
            )
        return self._suspects_found

    @property
    def suspects(self) -> set[int]:
        """誤判定らしいとして扱う追跡ID（利用者が「誤判定ではない」と戻したものを除く）。"""
        return set(self.suspects_found) - self.suspect_ok

    def set_suspect_settings(self, settings: SuspectSettings) -> None:
        """誤判定の判定基準を変えて、検知し直す。"""
        self.suspect_settings = settings
        self._suspects_found = None
        self._overlay_cache.clear()
        self.recount()

    def set_repeat_sec(self, seconds: float) -> None:
        """何秒以内のまたぎ直しを重複とみなすかを変えて、評価し直す。"""
        self.repeat_sec = max(0.1, float(seconds))
        self.recount()

    # ---- カウント線の編集 ----

    def rename_line(self, index: int, name: str) -> None:
        if 0 <= index < len(self.lines):
            self.lines[index].name = name.strip()

    def reverse_line(self, index: int) -> None:
        """IN と OUT を入れ替える（始点と終点を入れ替える）。"""
        if 0 <= index < len(self.lines):
            self.lines[index] = self.lines[index].reversed()
            self.recount()

    def move_endpoint(self, index: int, which: int, x: float, y: float, recount: bool = True) -> None:
        """線の始点（which=0）か終点（which=1）を動かす。ドラッグ中は recount=False で軽くする。"""
        if not 0 <= index < len(self.lines):
            return
        ln = self.lines[index]
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)
        if which == 0:
            ln.x1, ln.y1 = x, y
        else:
            ln.x2, ln.y2 = x, y
        if recount:
            self.recount()

    def export_lines(self, path: Path) -> None:
        export_lines_json(path, self.lines, self.width, self.height, self.video_name)

    def import_lines(self, path: Path, replace: bool = True) -> str:
        """JSON からカウント線を読み込む。戻り値は利用者向けの注意書き（なければ空）。"""
        lines, note = import_lines_json(path, self.width, self.height)
        self.lines = lines if replace else self.lines + lines
        self.recount()
        return note

    def set_suspect_ok(self, track_id: int, ok: bool) -> None:
        """ok=True: 誤判定ではないとして普通の線に戻す。False: 誤判定らしい線に戻す。"""
        if ok:
            self.suspect_ok.add(track_id)
        else:
            self.suspect_ok.discard(track_id)
        self.recount()

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
        self.recount()  # すり抜けの評価はエリアの縁まで見るので、エリアが変わったら評価し直す

    def area_people(self, index: int) -> int:
        return people_in_area(self.areas[index], self.tracks_xy, self.width, self.height)

    def add_area(self, polygon: list[list[float]], kind: str = "passage") -> bool:
        area = Area(kind=kind, polygon=[[float(x), float(y)] for x, y in polygon])
        if len(polygon) < 3 or area.polygon_area() < 25:
            return False
        self.areas.append(area)
        self.recount()
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
            self.recount()

    def area_changed(self, index: int) -> None:
        """エリアの形が変わったら、そのエリアの古い候補は消す。"""
        self.suggestions = [sg for sg in self.suggestions if sg.area_index != index]
        self.recount()

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
                suspects=self.suspects,
                repeat_sec=self.repeat_sec,
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

    def overlay(self, all_tracks: bool = False) -> np.ndarray:
        """足元軌跡の重畳画像（表示設定が変わったときだけ描き直す）。

        all_tracks=True なら時間帯の絞り込みをせず全員を描く（保存用）。
        """
        filtered = not all_tracks and self.is_time_filtered()
        time_key = (self.time_start, self.time_end) if filtered else None
        suspects = frozenset(self.suspects)
        key = (self.line_width, self.show_ids, self.grid, self.dim, time_key, suspects)
        cached = self._overlay_cache.get(filtered)
        if cached is None or cached[0] != key:
            img = draw_overlay(
                self.first_frame,
                self.visible_tracks() if filtered else self.tracks,
                smooth_window=self.smooth,
                dim=self.dim,
                show_ids=self.show_ids,
                grid=self.grid,
                line_width=self.line_width,
                suspects=suspects,
            )
            self._overlay_cache[filtered] = (key, img)
        return self._overlay_cache[filtered][1]

    def render(
        self,
        highlight_area: int | None = None,
        selected_suggestion: int | None = None,
        all_tracks: bool = False,
        selected_line: int | None = None,
        highlight_track: int | None = None,
        show_lines: bool = True,
    ) -> np.ndarray:
        """重畳画像に、エリア・候補線・カウント線・通過位置を描いた画像。

        show_areas（属性）と show_lines で、エリア・候補線とカウント線を描くかを選べる（画面の手順ごとに切り替える）。
        """
        img = self.overlay(all_tracks).copy()
        thick = max(2, round(3 * self.scale))
        if self.show_areas:
            self.draw_areas(img, highlight=highlight_area)
            self.draw_suggestions(img, thick, selected=selected_suggestion)
        if highlight_track is not None and highlight_track in self.tracks_xy:
            # 一覧で選んだ追跡線を白く太く示す
            pts = np.round(self.tracks_xy[highlight_track]).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 0), thick + 4, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, (255, 255, 255), thick + 2, cv2.LINE_AA)
        if not show_lines:
            return img
        for i, line in enumerate(self.lines):
            draw_count_line(img, line, line.label(i), line_color(i), thick, self.scale, selected=i == selected_line)
        if selected_line is not None and selected_line < len(self.evaluations):
            # 選んだ線の「途切れ」（橙の×）と「すり抜け」（紫の丸）の位置を示す
            ev = self.evaluations[selected_line]
            r = thick + 4
            for x, y in ev.break_points:
                c = (round(x), round(y))
                for color, w in (((0, 0, 0), thick + 3), ((0, 140, 255), thick + 1)):
                    cv2.line(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), color, w, cv2.LINE_AA)
                    cv2.line(img, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), color, w, cv2.LINE_AA)
            for x, y in ev.bypass_points:
                c = (round(x), round(y))
                cv2.circle(img, c, r, (0, 0, 0), thick + 3, cv2.LINE_AA)
                cv2.circle(img, c, r, (255, 80, 200), thick + 1, cv2.LINE_AA)
        # 通過した位置の印（数えた位置）。画面がごちゃつかないよう、線を選んでいるときはその線の分だけ描く
        for c in self.crossings:
            if selected_line is not None and not all_tracks and c.line_index != selected_line:
                continue
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
        size = max(14, round(16 * max(self.scale, 0.9)))
        labels = []
        for i, area in enumerate(self.areas):
            pts = np.round(area.points()).astype(np.int32)
            if len(pts) < 2:
                continue
            color = AREA_HIGHLIGHT if i == highlight else AREA_COLOR
            width = max(2, round(2 * self.scale)) + (1 if i == highlight else 0)
            cv2.polylines(img, [pts], True, (0, 0, 0), width + 2, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, color, width, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), width + 3, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, tuple(int(v) for v in p), width + 2, color, -1, cv2.LINE_AA)
            top = pts[np.argmin(pts[:, 1])]
            text = f"E{i + 1} {area.kind_label}"
            tw, th = text_size(text, size)
            labels.append((text, (min(max(top[0] - tw / 2, 2), self.width - tw - 2), max(2, top[1] - th - 10)), size, color))
        put_texts(img, labels)

    def draw_suggestions(self, img: np.ndarray, thick: int, selected: int | None = None) -> None:
        """カウント線の候補を点線で描く。"""
        size = max(13, round(14 * max(self.scale, 0.9)))
        labels = []
        for k, sg in enumerate(self.suggestions):
            color = SUGGEST_SELECTED if k == selected else SUGGEST_COLOR
            ln = sg.line
            p1, p2 = np.array([ln.x1, ln.y1], float), np.array([ln.x2, ln.y2], float)
            _dashed_line(img, p1, p2, color, thick + (1 if k == selected else 0), dash=max(8, round(12 * self.scale)))
            mid = (p1 + p2) / 2
            labels.append((f"E{sg.area_index + 1}-{sg.rank}", (mid[0] + 6, mid[1] - size - 4), size, color))
        put_texts(img, labels)

    def render_suggestions_image(self) -> np.ndarray:
        """保存用: エリアと候補線と各候補の人数。"""
        img = self.overlay(all_tracks=True).copy()
        thick = max(2, round(3 * self.scale))
        self.draw_areas(img)
        self.draw_suggestions(img, thick)
        rows = []
        for sg in self.suggestions:
            kind = self.areas[sg.area_index].kind if sg.area_index < len(self.areas) else ""
            label = self.areas[sg.area_index].kind_label if sg.area_index < len(self.areas) else kind
            rows.append(
                (
                    f"E{sg.area_index + 1}-{sg.rank}（{label}）  数えた人 {sg.people}/{sg.area_people}人"
                    f"（{sg.coverage:.0%}）  信頼度 {sg.reliability:.0f}点 {sg.grade}",
                    SUGGEST_COLOR,
                )
            )
        return self._with_footer(img, rows or [("候補なし", (200, 200, 200))])

    def _with_footer(self, img: np.ndarray, rows: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
        size = max(14, round(18 * self.scale))
        row_h = round(size * 1.6)
        footer = np.full((row_h * len(rows) + row_h // 2, self.width, 3), 32, dtype=np.uint8)
        put_texts(footer, [(text, (10, row_h // 3 + r * row_h), size, color) for r, (text, color) in enumerate(rows)], outline=False)
        return np.vstack([img, footer])

    def render_with_summary(self) -> np.ndarray:
        """保存用: 画像の下に線ごとの人数を書き足す。"""
        show = self.show_areas
        self.show_areas = False  # カウント結果の画像にはエリアと候補を描かない
        try:
            img = self.render(all_tracks=True)
        finally:
            self.show_areas = show
        rows = [(self.video_name, (220, 220, 220))]
        if not self.lines:
            rows.append(("カウント線なし", (200, 200, 200)))
        for i, c in enumerate(self.counts):
            text = f"{self.lines[i].label(i)}   IN {c.in_count}人   OUT {c.out_count}人   合計 {c.total}人"
            if i < len(self.evaluations):
                ev = self.evaluations[i]
                text += f"   （推定 {ev.range_text}人・信頼度 {ev.reliability:.0f}点 {ev.grade}）"
            rows.append((text, line_color(i)))
        return self._with_footer(img, rows)

    # ---- 保存 ----

    def save(self) -> None:
        self.meta["suspect_ok"] = sorted(self.suspect_ok)
        self.meta["suspect_settings"] = self.suspect_settings.to_dict()
        self.meta["repeat_sec"] = self.repeat_sec
        self.meta["view"] = {
            "line_width": self.line_width,
            "show_ids": self.show_ids,
            "grid": self.grid,
            "dim": self.dim,
        }
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        save_image(self.dir / "overlay.png", self.overlay(all_tracks=True))
        save_lines(self.dir / "lines.json", self.lines)
        save_counts_csv(self.dir / "counts.csv", self.lines, self.counts, self.evaluations)
        save_crossings_csv(self.dir / "crossings.csv", self.crossings, self.fps, self.lines)
        export_lines_json(self.dir / "count_lines.json", self.lines, self.width, self.height, self.video_name)
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
        out = []
        for i, c in enumerate(self.counts):
            text = f"{self.lines[i].label(i)}: IN {c.in_count}人 / OUT {c.out_count}人 / 合計 {c.total}人"
            if i < len(self.evaluations):
                ev = self.evaluations[i]
                text += f"（推定 {ev.range_text}人、信頼度 {ev.reliability:.0f}点 {ev.grade}）"
                if ev.duplicates:
                    text += f" 重複 {ev.duplicates}回"
                if ev.reasons:
                    text += f" 理由: {'・'.join(ev.reasons)}"
            out.append(text)
        return out


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
            reliability=d.get("reliability", 0.0),
            grade=d.get("grade", ""),
            lower=d.get("lower", 0),
            upper=d.get("upper", 0),
            reasons=d.get("reasons", []),
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


def draw_count_line(img: np.ndarray, line: CountLine, label: str, color, thick: int, scale: float,
                    selected: bool = False) -> None:
    """カウント線と、IN の向きの矢印（「IN」の文字つき）と線の名前を描く。"""
    p1 = (round(line.x1), round(line.y1))
    p2 = (round(line.x2), round(line.y2))
    if selected:  # 選んでいる線は白い縁で目立たせる
        cv2.line(img, p1, p2, (255, 255, 255), thick + 7, cv2.LINE_AA)
    cv2.line(img, p1, p2, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)
    # 端点（ドラッグで動かせる所）は大きめの丸
    for p in (p1, p2):
        cv2.circle(img, p, thick + 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p, thick + 2, (255, 255, 255) if selected else color, -1, cv2.LINE_AA)

    mid = (line.p1 + line.p2) / 2
    n = line.normal()  # IN として数える移動の向き
    off = max(20.0, 26 * scale)
    tail = tuple(int(v) for v in np.round(mid - n * off * 0.6))
    head = tuple(int(v) for v in np.round(mid + n * off * 0.6))
    cv2.arrowedLine(img, tail, head, (0, 0, 0), thick + 3, cv2.LINE_AA, tipLength=0.45)
    cv2.arrowedLine(img, tail, head, color, max(2, thick), cv2.LINE_AA, tipLength=0.45)

    size = max(14, round(16 * max(scale, 0.9)))
    d = line.p1 - line.p2
    d = d / (np.hypot(*d) or 1)
    items = []
    for text, pos in (("IN", mid + n * off * 1.25), (label, line.p1 + d * off)):
        tw, th = text_size(text, size)
        h, w = img.shape[:2]
        x = min(max(pos[0] - tw / 2, 2), w - tw - 2)
        y = min(max(pos[1] - th / 2 - 2, 2), h - th - 4)
        items.append((text, (x, y), size, color))
    put_texts(img, items)


def _put_centered(img, text: str, pos: np.ndarray, font_scale: float, color, thick: int) -> None:
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thick)
    x = int(min(max(pos[0] - tw / 2, 2), w - tw - 2))
    y = int(min(max(pos[1] + th / 2, th + 2), h - 2))
    cv2.putText(img, text, (x, y), FONT, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, max(1, thick - 1), cv2.LINE_AA)
