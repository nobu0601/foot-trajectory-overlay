"""足元軌跡カウントツールの画面。

  python -m foot_overlay.app [動画 または 結果フォルダ]

1. 「動画を開いて解析」で mp4 を選ぶと、全フレームを解析して足元軌跡を重ねた画像を表示する
2. 右側のスライダーで追跡線の太さなどをいつでも変えられる
3. 画像を2回クリックするとカウント線が引け、線ごとの通過人数が表に出る
結果は 動画名_overlay/ フォルダに自動保存される。
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .areas import AREA_KINDS
from .export import save_image
from .counting import CountLine
from .drawing import filter_short_tracks
from .session import Session, SuggestionCancelled, line_color, write_result
from .tracking import TrackingCancelled, open_video, track_video

APP_TITLE = "足元軌跡カウントツール"
VIDEO_TYPES = [("動画ファイル", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"), ("すべてのファイル", "*.*")]
MODELS = {
    "標準（速い）": "yolo11n.pt",
    "高精度（約3倍遅い）": "yolo11s.pt",
}
MIN_POINTS = 10  # この回数未満しか検出されなかったIDはノイズとして除く
SMOOTH = 5
MAX_LINE_WIDTH = 15
VERTEX_PICK_PX = 10  # 頂点をつかめる距離（画面上のpx）
KIND_KEYS = list(AREA_KINDS)
KIND_LABELS = [AREA_KINDS[k] for k in KIND_KEYS]


def result_dir_for(video: Path) -> Path:
    return video.with_name(f"{video.stem}_overlay")


def open_in_file_manager(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 (Windows のエクスプローラーで開く)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class App(tk.Tk):
    def __init__(self, initial: Path | None = None):
        super().__init__()
        self.title(APP_TITLE)
        # 画面に収まる大きさで開く
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1440, sw - 60)}x{min(920, sh - 100)}+20+10")
        self.minsize(900, 600)

        self.session: Session | None = None
        self.pending: tuple[float, float] | None = None  # 引きかけの線の始点（画像座標）
        self.view_scale = 1.0
        self.view_offset = (0, 0)
        self.photo: ImageTk.PhotoImage | None = None
        self.rendered = None  # 最後に描いた画像（BGR）
        self.render_dirty = True
        self.redraw_job: str | None = None
        self.save_job: str | None = None
        self.stop_event: threading.Event | None = None
        self.selected_area: int | None = None
        self.selected_line: int | None = None
        self.selected_suspect: int | None = None
        self.selected_suggestion: int | None = None
        self.area_draft: list[tuple[float, float]] | None = None  # 手で追加中のエリアの頂点
        self.drag: tuple[int, int] | None = None  # ドラッグ中の (エリア番号, 頂点番号)
        self.cursor: tuple[float, float] | None = None
        self.messages: queue.Queue = queue.Queue()

        self._build_ui()
        self._set_busy(False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if initial is not None:
            self.after(200, lambda: self.open_path(initial))

    # ================= 画面の組み立て =================

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.btn_video = ttk.Button(toolbar, text="動画を開いて解析…", command=self.choose_video)
        self.btn_video.pack(side=tk.LEFT)
        self.btn_result = ttk.Button(toolbar, text="解析結果を開く…", command=self.choose_result)
        self.btn_result.pack(side=tk.LEFT, padx=(6, 0))
        self.btn_batch = ttk.Button(toolbar, text="一括処理…", command=self.open_batch)
        self.btn_batch.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, text="検出モデル:").pack(side=tk.LEFT, padx=(16, 4))
        self.model_var = tk.StringVar(value=next(iter(MODELS)))
        self.model_box = ttk.Combobox(toolbar, textvariable=self.model_var, values=list(MODELS), state="readonly", width=18)
        self.model_box.pack(side=tk.LEFT)
        self.btn_cancel = ttk.Button(toolbar, text="中止", command=self.cancel_analysis)
        self.btn_cancel.pack(side=tk.LEFT, padx=(16, 0))
        self.btn_folder = ttk.Button(toolbar, text="保存フォルダを開く", command=self.open_result_folder)
        self.btn_folder.pack(side=tk.RIGHT)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.schedule_redraw())
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.bind("<Return>", lambda e: self.finish_area_draft())
        self.bind("<Escape>", lambda e: self.cancel_or_undo())
        self.bind("<Control-z>", lambda e: self.undo_line())

        side = ttk.Frame(body, padding=8, width=400)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        self._build_view_panel(side)
        self.notebook = ttk.Notebook(side)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        count_tab = ttk.Frame(self.notebook, padding=8)
        area_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(count_tab, text="カウント線")
        self.notebook.add(area_tab, text="エリア・候補提案")
        suspect_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(suspect_tab, text="誤判定の疑い")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self._build_count_panel(count_tab)
        self._build_area_panel(area_tab)
        self._build_suspect_panel(suspect_tab)

        statusbar = ttk.Frame(self, padding=(8, 4))
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="「動画を開いて解析…」から mp4 ファイルを選んでください。")
        ttk.Label(statusbar, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(statusbar, length=260, maximum=100)
        self.progress.pack(side=tk.RIGHT)

    def _build_view_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="表示", padding=8)
        box.pack(fill=tk.X)

        self.width_var = tk.IntVar(value=2)
        self.width_label = ttk.Label(box, text="追跡線の太さ: 2")
        self.width_label.pack(anchor=tk.W)
        self.width_scale = ttk.Scale(
            box, from_=1, to=MAX_LINE_WIDTH, orient=tk.HORIZONTAL, command=self.on_width_change
        )
        self.width_scale.pack(fill=tk.X, pady=(0, 6))

        self.dim_label = ttk.Label(box, text="背景の暗さ: 35%")
        self.dim_label.pack(anchor=tk.W)
        self.dim_scale = ttk.Scale(box, from_=0, to=80, orient=tk.HORIZONTAL, command=self.on_dim_change)
        self.dim_scale.pack(fill=tk.X, pady=(0, 6))

        # 表示する時間帯（人数の集計や候補提案には影響しない）
        self.time_label = ttk.Label(box, text="表示する時間帯: 全時間")
        self.time_label.pack(anchor=tk.W)
        time_row = ttk.Frame(box)
        time_row.pack(fill=tk.X)
        ttk.Label(time_row, text="開始").grid(row=0, column=0, sticky=tk.W)
        self.start_scale = ttk.Scale(time_row, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_time_change)
        self.start_scale.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Label(time_row, text="終了").grid(row=1, column=0, sticky=tk.W)
        self.end_scale = ttk.Scale(time_row, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_time_change)
        self.end_scale.grid(row=1, column=1, sticky=tk.EW, padx=4)
        time_row.columnconfigure(1, weight=1)
        time_btns = ttk.Frame(box)
        time_btns.pack(fill=tk.X, pady=(2, 6))
        self.btn_all_time = ttk.Button(time_btns, text="全時間を表示", command=self.reset_time_range)
        self.btn_all_time.pack(side=tk.LEFT)
        self.btn_save_view = ttk.Button(time_btns, text="表示中の画像を保存…", command=self.save_view_image)
        self.btn_save_view.pack(side=tk.RIGHT)

        checks = ttk.Frame(box)
        checks.pack(fill=tk.X)
        self.ids_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(checks, text="追跡ID番号", variable=self.ids_var, command=self.on_view_toggle).grid(
            row=0, column=0, sticky=tk.W
        )
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(checks, text="座標の格子（100px）", variable=self.grid_var, command=self.on_view_toggle).grid(
            row=0, column=1, sticky=tk.W, padx=(12, 0)
        )
        self.areas_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(checks, text="エリアと候補線", variable=self.areas_var, command=self.on_view_toggle).grid(
            row=1, column=0, sticky=tk.W
        )

    def _build_count_panel(self, box) -> None:
        ttk.Label(
            box,
            text="画像を2回クリックで線を追加\n右クリック / Esc: 引きかけの線を取り消し\n"
            "Ctrl+Z: 最後の線を削除\nA→B は線の中央の矢印の向きの通過人数",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        columns = ("line", "ab", "ba", "total", "range", "rel")
        self.table = ttk.Treeview(box, columns=columns, show="headings", height=6, selectmode="browse")
        for col, text, width in zip(
            columns, ("線", "A→B", "B→A", "合計", "推定範囲", "信頼度"), (40, 50, 50, 50, 75, 70)
        ):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True)
        self.table.bind("<<TreeviewSelect>>", self.on_line_select)
        self.line_detail_var = tk.StringVar(value="線を選ぶと、信頼度の内訳と直し方を表示します。")
        ttk.Label(box, textvariable=self.line_detail_var, justify=tk.LEFT, wraplength=350).pack(
            anchor=tk.W, fill=tk.X, pady=(6, 0)
        )

        buttons = ttk.Frame(box)
        buttons.pack(fill=tk.X, pady=(6, 0))
        self.btn_delete = ttk.Button(buttons, text="選んだ線を削除", command=self.delete_selected_line)
        self.btn_delete.pack(side=tk.LEFT)
        self.btn_clear = ttk.Button(buttons, text="すべて削除", command=self.clear_lines)
        self.btn_clear.pack(side=tk.RIGHT)

    def _build_area_panel(self, box) -> None:
        ttk.Label(
            box,
            text="人がよく通る場所（エスカレーター・階段・通路など）\n"
            "・形を直す: 頂点をドラッグ\n・手で追加: 頂点を順にクリック → Enter で確定",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(6, 4))
        self.btn_detect = ttk.Button(row, text="自動検出", command=self.detect_areas)
        self.btn_detect.pack(side=tk.LEFT)
        self.btn_add_area = ttk.Button(row, text="手で追加", command=self.start_area_draft)
        self.btn_add_area.pack(side=tk.LEFT, padx=4)
        self.btn_del_area = ttk.Button(row, text="削除", command=self.delete_area)
        self.btn_del_area.pack(side=tk.LEFT)

        cols = ("area", "kind", "people")
        self.area_table = ttk.Treeview(box, columns=cols, show="headings", height=3, selectmode="browse")
        for col, text, width in zip(cols, ("エリア", "種類", "通った人"), (60, 150, 90)):
            self.area_table.heading(col, text=text)
            self.area_table.column(col, width=width, anchor=tk.CENTER)
        self.area_table.pack(fill=tk.X)
        self.area_table.bind("<<TreeviewSelect>>", self.on_area_select)

        kind_row = ttk.Frame(box)
        kind_row.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(kind_row, text="選んだエリアの種類:").pack(side=tk.LEFT)
        self.kind_var = tk.StringVar()
        self.kind_box = ttk.Combobox(kind_row, textvariable=self.kind_var, values=KIND_LABELS, state="readonly", width=14)
        self.kind_box.pack(side=tk.LEFT, padx=4)
        self.kind_box.bind("<<ComboboxSelected>>", self.on_kind_change)

        ttk.Separator(box).pack(fill=tk.X, pady=4)
        n_row = ttk.Frame(box)
        n_row.pack(fill=tk.X, pady=(4, 4))
        ttk.Label(n_row, text="評価する候補線（各エリア）:").pack(side=tk.LEFT)
        self.n_var = tk.IntVar(value=20000)
        ttk.Spinbox(n_row, from_=1000, to=100000, increment=1000, textvariable=self.n_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(n_row, text="本").pack(side=tk.LEFT)
        self.btn_suggest = ttk.Button(box, text="カウント線の候補を提案（各エリア上位3本）", command=self.start_suggest)
        self.btn_suggest.pack(fill=tk.X)

        cols = ("cand", "people", "rate", "passes", "dup", "rel")
        self.suggest_table = ttk.Treeview(box, columns=cols, show="headings", height=6, selectmode="browse")
        for col, text, width in zip(
            cols, ("候補", "数えた人", "割合", "通過", "二重", "信頼度"), (55, 60, 45, 50, 40, 70)
        ):
            self.suggest_table.heading(col, text=text)
            self.suggest_table.column(col, width=width, anchor=tk.CENTER)
        self.suggest_table.pack(fill=tk.X, pady=(6, 0))
        self.suggest_table.bind("<<TreeviewSelect>>", self.on_suggest_select)
        self.suggest_table.bind("<Double-Button-1>", lambda e: self.adopt_suggestion())
        self.btn_adopt = ttk.Button(box, text="選んだ候補をカウント線に採用", command=self.adopt_suggestion)
        self.btn_adopt.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(
            box,
            text="順位: 数えた人 × 信頼度。誤判定らしい線が通る位置は除外\n"
            "割合: エリアを通った人のうち数えた割合 / 通過: 往復は2回",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

    def _build_suspect_panel(self, box) -> None:
        ttk.Label(
            box,
            text="追跡線の様子から誤判定らしい線を見つけ、灰色の点線で表示します。\n"
            "カウント線の候補は、これらの線が通らない位置に引きます。\n"
            "誤判定ではない線は、選んで「誤判定ではない」に戻せます。",
            justify=tk.LEFT,
            wraplength=350,
        ).pack(anchor=tk.W, pady=(0, 6))
        cols = ("id", "reason", "state")
        self.suspect_table = ttk.Treeview(box, columns=cols, show="headings", height=10, selectmode="browse")
        for col, text, width in zip(cols, ("追跡ID", "理由", "扱い"), (60, 190, 90)):
            self.suspect_table.heading(col, text=text)
            self.suspect_table.column(col, width=width, anchor=tk.CENTER)
        self.suspect_table.pack(fill=tk.BOTH, expand=True)
        self.suspect_table.bind("<<TreeviewSelect>>", self.on_suspect_select)
        self.btn_toggle_suspect = ttk.Button(box, text="選んだ線を「誤判定ではない」に戻す", command=self.toggle_suspect)
        self.btn_toggle_suspect.pack(fill=tk.X, pady=(6, 0))
        self.suspect_count_var = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.suspect_count_var).pack(anchor=tk.W, pady=(4, 0))

    def refresh_suspect_table(self) -> None:
        self.suspect_table.delete(*self.suspect_table.get_children())
        s = self.session
        if s is None:
            self.suspect_count_var.set("")
            return
        from .suspects import reason_text

        found = s.suspects_found
        for tid in sorted(found):
            state = "誤判定ではない" if tid in s.suspect_ok else "誤判定らしい"
            self.suspect_table.insert("", tk.END, iid=str(tid), values=(tid, reason_text(found[tid]), state))
        if self.selected_suspect is not None and str(self.selected_suspect) in self.suspect_table.get_children():
            self.suspect_table.selection_set(str(self.selected_suspect))
        self.suspect_count_var.set(f"誤判定らしい線: {len(s.suspects)}本 / 全{len(s.tracks)}本")
        self._update_toggle_label()

    def _update_toggle_label(self) -> None:
        s = self.session
        if s is not None and self.selected_suspect in s.suspect_ok:
            self.btn_toggle_suspect.config(text="選んだ線を「誤判定らしい」に戻す")
        else:
            self.btn_toggle_suspect.config(text="選んだ線を「誤判定ではない」に戻す")

    def on_suspect_select(self, _event=None) -> None:
        sel = self.suspect_table.selection()
        if not sel:
            return
        self.selected_suspect = int(sel[0])
        self._update_toggle_label()
        self.render_dirty = True
        self.schedule_redraw()

    def toggle_suspect(self) -> None:
        s = self.session
        if s is None or self.selected_suspect is None:
            messagebox.showinfo(APP_TITLE, "一覧から線を選んでください。")
            return
        tid = self.selected_suspect
        s.set_suspect_ok(tid, tid not in s.suspect_ok)
        self.refresh_suspect_table()
        self.lines_changed()  # 人数の信頼度が変わるので表を更新して保存する
        if s.suggestions:
            self.status_var.set("誤判定の扱いを変えました。候補線に反映するには、もう一度「候補を提案」してください。")

    # ================= 開く・解析 =================

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(title="解析する動画を選択", filetypes=VIDEO_TYPES)
        if path:
            self.open_video(Path(path))

    def choose_result(self) -> None:
        path = filedialog.askdirectory(title="解析結果のフォルダ（動画名_overlay）を選択")
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        if path.is_dir():
            self.load_result(path)
        elif path.is_file():
            self.open_video(path)
        else:
            messagebox.showerror(APP_TITLE, f"見つかりません:\n{path}")

    def open_video(self, video: Path) -> None:
        out_dir = result_dir_for(video)
        if (out_dir / "tracks.csv").is_file():
            answer = messagebox.askyesnocancel(
                APP_TITLE,
                f"この動画の解析結果がすでにあります。\n{out_dir}\n\n"
                "「はい」: 前回の結果を開く（すぐ開けます）\n「いいえ」: 解析し直す（カウント線は消えます）",
            )
            if answer is None:
                return
            if answer:
                self.load_result(out_dir)
                return
        self.start_analysis(video, out_dir)

    def start_analysis(self, video: Path, out_dir: Path) -> None:
        self.flush_save()
        self.session = None
        self.pending = None
        self.reset_area_state()
        self.render_dirty = True
        self.rendered = read_first_frame(video)  # 解析中は最初のフレームを表示しておく
        self.selected_line = None
        self.selected_suspect = None
        self.refresh_table()
        self.refresh_area_tables()
        self.refresh_suspect_table()
        self.canvas.delete("all")
        self.schedule_redraw()
        self._set_busy(True)
        self.progress["value"] = 0
        self.status_var.set(f"{video.name}: 検出モデルを準備しています…（初回はダウンロードに時間がかかります）")

        model = MODELS[self.model_var.get()]
        self.stop_event = threading.Event()
        started = time.time()

        def progress(done: int, total: int, ids: int) -> None:
            self.messages.put(("progress", video.name, done, total, ids, started))

        def worker() -> None:
            try:
                result = track_video(video, model_name=model, progress=progress, stop_event=self.stop_event)
                tracks = filter_short_tracks(result.tracks, MIN_POINTS)
                write_result(out_dir, video, result, tracks, smooth=SMOOTH, min_points=MIN_POINTS)
                self.messages.put(("done", out_dir, len(result.tracks), len(tracks), time.time() - started))
            except TrackingCancelled:
                self.messages.put(("cancelled",))
            except Exception as e:  # 画面に理由を出す
                self.messages.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self.poll_messages)

    def poll_messages(self) -> None:
        finished = False
        try:
            while True:
                msg = self.messages.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, name, done, total, ids, started = msg
                    percent = done * 100 / total if total else 0
                    self.progress["value"] = percent
                    elapsed = time.time() - started
                    remain = elapsed / done * (total - done) if done else 0
                    self.status_var.set(
                        f"{name}: 解析中 {percent:.0f}%（{done}/{total}フレーム, 追跡ID {ids}件, 残り約{_fmt_sec(remain)}）"
                    )
                elif kind == "done":
                    _, out_dir, n_all, n_drawn, elapsed = msg
                    finished = True
                    self.load_result(out_dir)
                    self.status_var.set(
                        f"解析完了（{_fmt_sec(elapsed)}）: 追跡ID {n_all}件中 {n_drawn}件を表示。"
                        "画像を2回クリックしてカウント線を引いてください。"
                    )
                elif kind == "suggest_progress":
                    _, area_i, n_areas, done, total = msg
                    overall = (area_i + done / total) / n_areas * 100 if total else 0
                    self.progress["value"] = overall
                    self.status_var.set(f"候補線を評価中: エリア E{area_i + 1}/{n_areas}（{done:,}/{total:,}本）")
                elif kind == "suggest_done":
                    finished = True
                    _, suggestions, elapsed = msg
                    if self.session is not None:
                        self.session.suggestions = suggestions
                        self.selected_suggestion = None
                        self.refresh_area_tables()
                        self.render_dirty = True
                        self.schedule_redraw()
                        self.schedule_save()
                    self.status_var.set(
                        f"候補線を {len(suggestions)} 本提案しました（{_fmt_sec(elapsed)}）。"
                        "表で選んでダブルクリックするとカウント線に採用されます。"
                    )
                elif kind == "cancelled":
                    finished = True
                    self.status_var.set("中止しました。")
                elif kind == "error":
                    finished = True
                    self.status_var.set("解析に失敗しました。")
                    messagebox.showerror(APP_TITLE, f"解析に失敗しました。\n\n{msg[1]}")
        except queue.Empty:
            pass
        if finished:
            self._set_busy(False)
            self.progress["value"] = 100 if self.session else 0
        else:
            self.after(100, self.poll_messages)

    def cancel_analysis(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("中止しています…")

    def load_result(self, result_dir: Path) -> None:
        self.flush_save()
        try:
            session = Session(result_dir)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"解析結果を開けませんでした。\n\n{e}")
            return
        self.session = session
        self.pending = None
        self.reset_area_state()
        session.show_areas = self.areas_var.get()
        self.title(f"{APP_TITLE} - {session.video_name}")
        self.width_scale.set(session.line_width)
        self.dim_scale.set(round(session.dim * 100))
        self.ids_var.set(session.show_ids)
        self.grid_var.set(session.grid > 0)
        self._setting_time = True
        duration = max(session.duration, 0.1)
        self.start_scale.configure(to=duration)
        self.end_scale.configure(to=duration)
        self.start_scale.set(0)
        self.end_scale.set(duration)
        self._setting_time = False
        self._update_view_labels()
        self.render_dirty = True
        self.selected_line = None
        self.selected_suspect = None
        self.refresh_table()
        self.refresh_area_tables()
        self.refresh_suspect_table()
        self.schedule_redraw()
        self.schedule_save()
        self.status_var.set(f"{result_dir} を開きました。画像を2回クリックしてカウント線を引いてください。")

    # ================= 表示設定 =================

    def on_width_change(self, _value=None) -> None:
        if self.session is None:
            return
        width = int(round(float(self.width_scale.get())))
        if width != self.session.line_width:
            self.session.line_width = width
            self.view_changed()
        self._update_view_labels()

    def on_dim_change(self, _value=None) -> None:
        if self.session is None:
            return
        dim = round(float(self.dim_scale.get())) / 100
        if dim != self.session.dim:
            self.session.dim = dim
            self.view_changed()
        self._update_view_labels()

    def on_view_toggle(self) -> None:
        if self.session is None:
            return
        self.session.show_ids = self.ids_var.get()
        self.session.grid = 100 if self.grid_var.get() else 0
        self.session.show_areas = self.areas_var.get()
        self.view_changed()

    def _update_view_labels(self) -> None:
        if self.session is None:
            return
        self.width_label.config(text=f"追跡線の太さ: {self.session.line_width}")
        self.dim_label.config(text=f"背景の暗さ: {round(self.session.dim * 100)}%")
        s = self.session
        shown, total = len(s.visible_tracks()), len(s.tracks)
        if s.is_time_filtered():
            text = f"表示する時間帯: {_fmt_clock(s.time_start)}〜{_fmt_clock(s.time_end)}（{shown}/{total}人）"
        else:
            text = f"表示する時間帯: 全時間 0:00〜{_fmt_clock(s.duration)}（{total}人）"
        self.time_label.config(text=text)

    def on_time_change(self, _value=None) -> None:
        if self.session is None or getattr(self, "_setting_time", False):
            return
        start = float(self.start_scale.get())
        end = float(self.end_scale.get())
        if start > end:  # 開始が終了を追い越したら、もう一方も動かして揃える
            self._setting_time = True
            if abs(start - self.session.time_start) > 1e-9:
                self.end_scale.set(start)
                end = start
            else:
                self.start_scale.set(end)
                start = end
            self._setting_time = False
        # 動画の端まで寄せたときは「全時間」と同じ扱いにする
        if end >= self.session.duration - 0.05:
            end = self.session.duration
        if (start, end) != (self.session.time_start, self.session.time_end):
            self.session.time_start, self.session.time_end = start, end
            self.render_dirty = True
            self.schedule_redraw()  # 表示だけの変更なので保存はしない
        self._update_view_labels()

    def reset_time_range(self) -> None:
        if self.session is None:
            return
        self._setting_time = True
        self.start_scale.set(0)
        self.end_scale.set(self.session.duration)
        self._setting_time = False
        self.on_time_change()

    def save_view_image(self) -> None:
        """いま表示している画像（時間帯の絞り込み・カウント線を含む）を保存する。"""
        if self.session is None:
            return
        s = self.session
        name = f"{Path(s.video_name).stem}_view"
        if s.is_time_filtered():
            name += f"_{_fmt_clock(s.time_start).replace(':', 'm')}s-{_fmt_clock(s.time_end).replace(':', 'm')}s"
        path = filedialog.asksaveasfilename(
            title="表示中の画像を保存",
            initialdir=str(s.dir),
            initialfile=name + ".png",
            defaultextension=".png",
            filetypes=[("PNG 画像", "*.png"), ("JPEG 画像", "*.jpg")],
        )
        if not path:
            return
        try:
            save_image(Path(path), s.render(self.selected_area, self.selected_suggestion))
            self.status_var.set(f"保存しました: {path}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"保存に失敗しました。\n\n{e}")

    def open_batch(self) -> None:
        """一括処理の画面を開く。"""
        from .batch_dialog import BatchDialog

        self.flush_save()
        width = self.session.line_width if self.session is not None else 1
        BatchDialog(self, MODELS, self.model_var.get(), width, open_in_file_manager, self.load_result)

    def view_changed(self) -> None:
        self.render_dirty = True
        self.schedule_redraw()
        self.schedule_save()

    # ================= カウント線の操作 =================

    def to_image(self, x: float, y: float) -> tuple[float, float] | None:
        """画面上の座標 → 元画像の座標。画像の外なら None。"""
        if self.session is None:
            return None
        ix = (x - self.view_offset[0]) / self.view_scale
        iy = (y - self.view_offset[1]) / self.view_scale
        if 0 <= ix < self.session.width and 0 <= iy < self.session.height:
            return ix, iy
        return None

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.view_scale + self.view_offset[0], iy * self.view_scale + self.view_offset[1]

    def mode(self) -> str:
        """右側で開いているタブ: "lines"（カウント線）/ "areas"（エリア・候補提案）/ "suspects"（誤判定の疑い）。"""
        try:
            return {1: "areas", 2: "suspects"}.get(self.notebook.index(self.notebook.select()), "lines")
        except tk.TclError:
            return "lines"

    def on_tab_changed(self, _event=None) -> None:
        # タブを切り替えたら、引きかけの線・エリアは取り消す
        self.pending = None
        self.area_draft = None
        self.canvas.delete("pending")
        if self.session is not None and self.mode() == "areas" and not self.session.areas:
            self.status_var.set("「自動検出」でエリアを作るか、「手で追加」でエリアを囲んでください。")

    def on_left_click(self, event) -> None:
        if self.session is None:
            return
        if self.mode() == "areas":
            self.on_area_click(event)
            return
        if self.mode() == "suspects":
            return  # 「誤判定の疑い」タブでは画像のクリックで線を引かない
        point = self.to_image(event.x, event.y)
        if point is None:
            return
        if self.pending is None:
            self.pending = point
            self.draw_pending(event.x, event.y)
            self.status_var.set("終点をクリックしてください（右クリック / Esc で取り消し）")
            return
        x1, y1 = self.pending
        if self.session.add_line(CountLine(round(x1), round(y1), round(point[0]), round(point[1]))):
            self.pending = None
            self.canvas.delete("pending")
            self.lines_changed()
            c = self.session.counts[-1]
            n = len(self.session.lines)
            self.status_var.set(f"L{n} を追加しました: A→B {c.a_to_b}人 / B→A {c.b_to_a}人 / 合計 {c.total}人")

    def on_right_click(self, _event=None) -> None:
        self.cancel_or_undo()

    def cancel_or_undo(self) -> None:
        if self.area_draft is not None:
            if self.area_draft:
                self.area_draft.pop()
            if not self.area_draft:
                self.area_draft = None
                self.status_var.set("エリアの追加を取り消しました。")
            self.draw_area_draft()
            return
        if self.pending is not None:
            self.pending = None
            self.canvas.delete("pending")
            self.status_var.set("線の追加を取り消しました。")

    def undo_line(self) -> None:
        if self.session is not None and self.session.lines:
            self.session.undo()
            self.lines_changed()

    def on_motion(self, event) -> None:
        self.cursor = (event.x, event.y)
        if self.pending is not None:
            self.draw_pending(event.x, event.y)
        if self.area_draft is not None:
            self.draw_area_draft()

    def draw_pending(self, x: float, y: float) -> None:
        """引きかけの線（始点からマウス位置まで）を画面上に仮表示する。"""
        self.canvas.delete("pending")
        if self.pending is None or self.session is None:
            return
        sx, sy = self.to_screen(*self.pending)
        b, g, r = line_color(len(self.session.lines))
        color = f"#{r:02x}{g:02x}{b:02x}"
        self.canvas.create_line(sx, sy, x, y, fill="black", width=5, tags="pending")
        self.canvas.create_line(sx, sy, x, y, fill=color, width=3, tags="pending", dash=(6, 3))
        self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill=color, outline="black", tags="pending")

    def delete_selected_line(self) -> None:
        if self.session is None:
            return
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo(APP_TITLE, "表から削除する線を選んでください。")
            return
        self.session.remove_line(self.table.index(selected[0]))
        self.selected_line = None
        self.lines_changed()

    def clear_lines(self) -> None:
        if self.session is None or not self.session.lines:
            return
        if messagebox.askyesno(APP_TITLE, "カウント線をすべて削除しますか？"):
            self.session.clear()
            self.lines_changed()

    def lines_changed(self) -> None:
        self.render_dirty = True
        self.refresh_table()
        self.schedule_redraw()
        self.schedule_save()

    def refresh_table(self) -> None:
        self.table.delete(*self.table.get_children())
        if self.session is None:
            return
        for i, c in enumerate(self.session.counts, start=1):
            ev = self.session.evaluations[i - 1] if i - 1 < len(self.session.evaluations) else None
            extra = (f"{ev.range_text}人", f"{ev.reliability:.0f}点 {ev.grade}") if ev else ("", "")
            self.table.insert("", tk.END, values=(f"L{i}", c.a_to_b, c.b_to_a, c.total, *extra))
        if self.selected_line is not None and self.selected_line < len(self.session.lines):
            self.table.selection_set(self.table.get_children()[self.selected_line])
        else:
            self.selected_line = None
        self.update_line_detail()

    def on_line_select(self, _event=None) -> None:
        sel = self.table.selection()
        if not sel or self.session is None:
            return
        index = self.table.index(sel[0])
        if index != self.selected_line:
            self.selected_line = index
            self.update_line_detail()
            self.render_dirty = True
            self.schedule_redraw()

    def update_line_detail(self) -> None:
        """選んだカウント線の信頼度の内訳（低い理由と直し方）を表示する。"""
        s = self.session
        if s is None or self.selected_line is None or self.selected_line >= len(s.evaluations):
            self.line_detail_var.set("線を選ぶと、信頼度の内訳と直し方を表示します。")
            return
        ev = s.evaluations[self.selected_line]
        text = f"L{self.selected_line + 1}: {ev.summary}"
        if ev.reasons:
            for reason, hint in zip(ev.reasons, ev.hints):
                text += f"\n・{reason}\n　→ {hint}"
        else:
            text += "\n数え間違いの痕跡はほとんどありません。"
        if ev.break_points or ev.bypass_points:
            text += "\n画像の印: 橙の× = 追跡が途切れた所\n　　　　　紫の○ = 線の端の外をすり抜けた所"
        self.line_detail_var.set(text)
    # ================= エリア・候補提案 =================

    def reset_area_state(self) -> None:
        self.selected_area = None
        self.selected_suggestion = None
        self.area_draft = None
        self.drag = None

    def areas_changed(self) -> None:
        self.refresh_area_tables()
        self.refresh_table()  # カウント線の信頼度はエリアの形でも変わる
        self.render_dirty = True
        self.schedule_redraw()
        self.schedule_save()

    def refresh_area_tables(self) -> None:
        self.area_table.delete(*self.area_table.get_children())
        self.suggest_table.delete(*self.suggest_table.get_children())
        if self.session is None:
            return
        for i, area in enumerate(self.session.areas):
            self.area_table.insert("", tk.END, values=(f"E{i + 1}", area.kind_label, f"{self.session.area_people(i)}人"))
        for sg in self.session.suggestions:
            self.suggest_table.insert(
                "",
                tk.END,
                values=(
                    f"E{sg.area_index + 1}-{sg.rank}",
                    f"{sg.people}人",
                    f"{sg.coverage:.0%}",
                    f"{sg.passes}回",
                    sg.duplicates,
                    f"{sg.reliability:.0f}点 {sg.grade}" if sg.grade else "",
                ),
            )
        if self.selected_area is not None and self.selected_area < len(self.session.areas):
            item = self.area_table.get_children()[self.selected_area]
            self.area_table.selection_set(item)
            self.kind_var.set(self.session.areas[self.selected_area].kind_label)
        else:
            self.selected_area = None
            self.kind_var.set("")

    def detect_areas(self) -> None:
        if self.session is None:
            return
        if self.session.areas and not messagebox.askyesno(
            APP_TITLE, "今のエリアと候補線を消して、エリアを自動検出し直しますか？"
        ):
            return
        self.session.auto_detect_areas()
        self.selected_area = None
        self.selected_suggestion = None
        self.areas_changed()
        n = len(self.session.areas)
        if n:
            self.status_var.set(f"エリアを {n} 件見つけました。表で選んで種類を設定し、形は頂点をドラッグで直せます。")
        else:
            self.status_var.set("人がよく通る場所が見つかりませんでした。「手で追加」でエリアを囲んでください。")

    def start_area_draft(self) -> None:
        if self.session is None:
            return
        self.area_draft = []
        self.status_var.set("エリアの頂点を順にクリックしてください。ダブルクリックか Enter で確定、右クリックで1つ戻す。")

    def finish_area_draft(self) -> None:
        if self.area_draft is None or self.session is None:
            return
        # ダブルクリックで同じ点が重なって入るので取り除く
        pts: list[tuple[float, float]] = []
        for p in self.area_draft:
            if not pts or abs(p[0] - pts[-1][0]) + abs(p[1] - pts[-1][1]) > 3:
                pts.append(p)
        if len(pts) < 3:
            self.status_var.set("エリアには頂点が3つ以上必要です。続けてクリックしてください。")
            return
        if self.session.add_area([list(p) for p in pts]):
            self.area_draft = None
            self.canvas.delete("pending")
            self.selected_area = len(self.session.areas) - 1
            self.areas_changed()
            self.status_var.set(f"E{self.selected_area + 1} を追加しました。種類を選んでください。")

    def draw_area_draft(self) -> None:
        self.canvas.delete("pending")
        if not self.area_draft:
            return
        pts = [self.to_screen(*p) for p in self.area_draft]
        if self.cursor is not None:
            pts.append(self.cursor)
        flat = [v for p in pts for v in p]
        if len(pts) >= 2:
            self.canvas.create_line(*flat, fill="#50c8ff", width=3, dash=(6, 3), tags="pending")
        for x, y in pts[: len(self.area_draft)]:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#50c8ff", outline="black", tags="pending")

    def pick_vertex(self, sx: float, sy: float) -> tuple[int, int] | None:
        """画面座標の近くにあるエリアの頂点 (エリア番号, 頂点番号)。"""
        if self.session is None:
            return None
        best, best_d = None, VERTEX_PICK_PX
        for ai, area in enumerate(self.session.areas):
            for vi, (x, y) in enumerate(area.polygon):
                px, py = self.to_screen(x, y)
                d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = (ai, vi), d
        return best

    def is_busy(self) -> bool:
        return str(self.btn_cancel["state"]) == tk.NORMAL

    def on_area_click(self, event) -> None:
        if self.is_busy():
            return  # 候補線の計算中はエリアを変えない
        if self.area_draft is not None:
            point = self.to_image(event.x, event.y)
            if point is not None:
                self.area_draft.append(point)
                self.draw_area_draft()
            return
        picked = self.pick_vertex(event.x, event.y)
        if picked is not None:
            self.drag = picked
            self.select_area(picked[0])
            return
        point = self.to_image(event.x, event.y)
        if point is not None:
            for i, area in enumerate(self.session.areas):
                if area.contains(*point):
                    self.select_area(i)
                    return

    def on_drag(self, event) -> None:
        if self.drag is None or self.session is None:
            return
        ai, vi = self.drag
        ix = (event.x - self.view_offset[0]) / self.view_scale
        iy = (event.y - self.view_offset[1]) / self.view_scale
        ix = min(max(ix, 0), self.session.width - 1)
        iy = min(max(iy, 0), self.session.height - 1)
        self.session.areas[ai].polygon[vi] = [ix, iy]
        self.render_dirty = True
        self.schedule_redraw()

    def on_release(self, _event=None) -> None:
        if self.drag is None or self.session is None:
            return
        ai, _ = self.drag
        self.drag = None
        had = any(sg.area_index == ai for sg in self.session.suggestions)
        self.session.area_changed(ai)
        self.areas_changed()
        if had:
            self.status_var.set(f"E{ai + 1} の形を変えたので、このエリアの候補線を消しました。もう一度提案してください。")

    def on_double_click(self, _event=None) -> None:
        if self.mode() == "areas" and self.area_draft is not None:
            self.finish_area_draft()

    def select_area(self, index: int) -> None:
        self.selected_area = index
        items = self.area_table.get_children()
        if index < len(items):
            self.area_table.selection_set(items[index])
        self.kind_var.set(self.session.areas[index].kind_label)
        self.render_dirty = True
        self.schedule_redraw()

    def on_area_select(self, _event=None) -> None:
        sel = self.area_table.selection()
        if not sel or self.session is None:
            return
        index = self.area_table.index(sel[0])
        if index != self.selected_area:
            self.select_area(index)

    def on_kind_change(self, _event=None) -> None:
        if self.session is None or self.selected_area is None:
            return
        label = self.kind_var.get()
        if label in KIND_LABELS:
            self.session.areas[self.selected_area].kind = KIND_KEYS[KIND_LABELS.index(label)]
            self.areas_changed()

    def delete_area(self) -> None:
        if self.session is None or self.selected_area is None:
            messagebox.showinfo(APP_TITLE, "表から削除するエリアを選んでください。")
            return
        self.session.remove_area(self.selected_area)
        self.selected_area = None
        self.selected_suggestion = None
        self.areas_changed()

    def start_suggest(self) -> None:
        if self.session is None:
            return
        if self.area_draft is not None:
            self.finish_area_draft()
        if not self.session.areas:
            self.session.auto_detect_areas()
            self.areas_changed()
            if not self.session.areas:
                messagebox.showinfo(APP_TITLE, "エリアが見つかりませんでした。「手で追加」でエリアを囲んでください。")
                return
        try:
            n = int(self.n_var.get())
        except (tk.TclError, ValueError):
            n = 20000
        n = min(max(n, 100), 200000)
        session = self.session
        n_areas = len(session.areas)
        self.stop_event = threading.Event()
        stop = self.stop_event
        started = time.time()
        self._set_busy(True)
        self.progress["value"] = 0

        def progress(area_i: int, done: int, total: int) -> None:
            self.messages.put(("suggest_progress", area_i, n_areas, done, total))

        def worker() -> None:
            try:
                result = session.compute_suggestions(n_candidates=n, progress=progress, should_stop=stop.is_set)
                self.messages.put(("suggest_done", result, time.time() - started))
            except SuggestionCancelled:
                self.messages.put(("cancelled",))
            except Exception as e:  # 画面に理由を出す
                self.messages.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self.poll_messages)

    def on_suggest_select(self, _event=None) -> None:
        sel = self.suggest_table.selection()
        if not sel:
            return
        self.selected_suggestion = self.suggest_table.index(sel[0])
        self.render_dirty = True
        self.schedule_redraw()

    def adopt_suggestion(self) -> None:
        if self.session is None:
            return
        if self.selected_suggestion is None:
            messagebox.showinfo(APP_TITLE, "表から採用する候補を選んでください。")
            return
        sg = self.session.suggestions[self.selected_suggestion]
        if self.session.adopt_suggestion(self.selected_suggestion):
            self.lines_changed()
            c = self.session.counts[-1]
            n = len(self.session.lines)
            self.status_var.set(
                f"候補 E{sg.area_index + 1}-{sg.rank} を L{n} として採用しました: "
                f"A→B {c.a_to_b}人 / B→A {c.b_to_a}人 / 合計 {c.total}人"
            )

    # ================= 描画・保存 =================

    def schedule_redraw(self) -> None:
        # スライダーを動かしている間に描き直しが溜まらないよう、まとめて1回にする
        if self.redraw_job is None:
            self.redraw_job = self.after(15, self.redraw)

    def redraw(self) -> None:
        self.redraw_job = None
        self.canvas.delete("image")
        if self.session is None and self.rendered is None:
            return
        if self.session is not None and (self.render_dirty or self.rendered is None):
            self.rendered = self.session.render(
                self.selected_area,
                self.selected_suggestion,
                selected_line=self.selected_line,
                highlight_track=self.selected_suspect,
            )
            self.render_dirty = False
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        h, w = self.rendered.shape[:2]
        scale = min(cw / w, ch / h)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        self.view_scale = scale
        self.view_offset = ((cw - dw) // 2, (ch - dh) // 2)
        rgb = cv2.cvtColor(self.rendered, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (dw, dh), interpolation=interp)))
        self.canvas.create_image(*self.view_offset, image=self.photo, anchor=tk.NW, tags="image")
        self.canvas.tag_lower("image")

    def schedule_save(self) -> None:
        # 操作のたびに画像を書き出すと重いので、操作が止まってから保存する
        if self.save_job is not None:
            self.after_cancel(self.save_job)
        self.save_job = self.after(600, self.save_now)

    def save_now(self) -> None:
        self.save_job = None
        if self.session is None:
            return
        try:
            self.session.save()
        except Exception as e:
            self.status_var.set(f"保存に失敗しました: {e}")

    def flush_save(self) -> None:
        if self.save_job is not None:
            self.after_cancel(self.save_job)
            self.save_now()

    def open_result_folder(self) -> None:
        if self.session is not None:
            self.flush_save()
            open_in_file_manager(self.session.dir)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for w in (
            self.btn_video,
            self.btn_result,
            self.btn_delete,
            self.btn_clear,
            self.btn_folder,
            self.btn_batch,
            self.btn_save_view,
            self.btn_toggle_suspect,
            self.btn_detect,
            self.btn_add_area,
            self.btn_del_area,
            self.btn_suggest,
            self.btn_adopt,
        ):
            w.config(state=state)
        self.model_box.config(state=tk.DISABLED if busy else "readonly")
        self.btn_cancel.config(state=tk.NORMAL if busy else tk.DISABLED)

    def on_close(self) -> None:
        if self.stop_event is not None and not self.stop_event.is_set() and self.is_busy():
            if not messagebox.askyesno(APP_TITLE, "処理中です。中止して終了しますか？"):
                return
            self.stop_event.set()
        self.flush_save()
        self.destroy()


def read_first_frame(video: Path):
    """動画の最初のフレーム（読めなければ None）。"""
    try:
        cap, tmp = open_video(video)
    except Exception:
        return None
    ok, frame = cap.read()
    cap.release()
    if tmp is not None:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return frame if ok else None


def _fmt_clock(sec: float) -> str:
    """秒を 分:秒 の形にする（例: 83.4 → 1:23）。"""
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _fmt_sec(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}秒"
    if sec < 3600:
        return f"{sec // 60}分{sec % 60:02d}秒"
    return f"{sec // 3600}時間{sec % 3600 // 60:02d}分"


def _enable_high_dpi() -> None:
    """Windows の高解像度画面で文字や画像がぼやけないようにする。"""
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _enable_high_dpi()
    app = App(Path(argv[0]) if argv else None)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
