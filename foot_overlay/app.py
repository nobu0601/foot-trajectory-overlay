"""足元軌跡カウントツールの画面（手順ごとのステップ式）。

  python -m foot_overlay.app [動画 または 結果フォルダ]

右側の手順に沿って進める:
  ① 解析      動画を選ぶと、全フレームで人物を追跡し、足元の通り道（追跡線）を重ねる
  ② 確認      見やすさの調整と、誤判定らしい追跡線の確認・判定基準の調整
  ③ 候補      エリアを決めて、数えやすいカウント線の候補を自動で探す（任意）
  ④ カウント線 線を引く・端をドラッグで調整・名前と IN/OUT の向きを決める
  ⑤ 書き出し  カウント線を JSON で保存・読み込み、画像を保存
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
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk

from .areas import AREA_KINDS
from .counting import CountLine
from .drawing import filter_short_tracks
from .export import save_image
from .session import Session, SuggestionCancelled, line_color, write_result
from .suspects import SuspectSettings, reason_text
from .resume import progress_info
from .tracking import TrackingCancelled, open_video, track_video

APP_TITLE = "足元軌跡カウントツール"
VIDEO_TYPES = [("動画ファイル", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV"), ("すべてのファイル", "*.*")]
MODELS = {
    "標準（速い）": "yolo11n.pt",
    "高精度（約3倍遅い）": "yolo11s.pt",
}
MIN_POINTS = 10  # この回数未満しか検出されなかったIDはノイズとして除く
SMOOTH = 5
MAX_LINE_WIDTH = 15
PICK_PX = 12  # 頂点・線の端をつかめる距離（画面上のpx）
KIND_KEYS = list(AREA_KINDS)
KIND_LABELS = [AREA_KINDS[k] for k in KIND_KEYS]

# 色（画面の部品用）
ACCENT = "#1f6feb"
ACCENT_LIGHT = "#dbe9ff"
WARN_BG = "#fde4cc"
MUTED = "#5b6470"

# 手順: (キー, ボタンの文字, 見出し, 説明)
STEPS = [
    ("analyze", "①解析", "① 動画を解析する",
     "動画を選ぶと、全フレームで人を追跡し、足元の通り道（追跡線）を最初のフレームに重ねます。"),
    ("review", "②確認", "② 追跡線を確認する",
     "見やすさを調整し、誤判定らしい線（灰色の点線）を確認します。判定の基準もここで変えられます。"),
    ("suggest", "③候補", "③ カウント線の候補を探す（任意）",
     "人がよく通る場所（エリア）を決めると、数えやすい線の候補を自動で探します。使わずに④へ進んでも構いません。"),
    ("lines", "④線", "④ カウント線を決める",
     "画像を2回クリックして線を引きます。線の端の丸はドラッグで動かせます。名前と IN の向きもここで決めます。"),
    ("export", "⑤出力", "⑤ 結果を書き出す",
     "カウント線を JSON で保存・読み込みしたり、画像を保存したりします。人数などは自動で保存されています。"),
]
STEP_KEYS = [s[0] for s in STEPS]

# 画像の下に出す「いま見えている印の意味」
LEGENDS = {
    "analyze": "色付きの線 = 1人分の足元の通り道　● = 歩き始め　➜ = 歩き終わりと向き",
    "review": "色付きの線 = 足元の通り道　灰色の点線 = 誤判定らしい線　白く太い線 = 一覧で選んだ線",
    "suggest": "水色の枠 = エリア（角の丸をドラッグで形を直す）　緑の点線 = 候補線（白 = 選んだ候補）",
    "lines": "太い線 = カウント線（端の丸をドラッグで調整）　矢印 = IN の向き　赤い点 = 選んだ線で数えた位置"
             "　橙の× = 追跡が途切れた所　紫の○ = 線の端の外をすり抜けた所",
    "export": "カウント線と、数えた位置（赤い点）。保存される結果画像と同じ内容です",
}


def result_dir_for(video: Path) -> Path:
    return video.with_name(f"{video.stem}_overlay")


def open_in_file_manager(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 (Windows のエクスプローラーで開く)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ScrollFrame(ttk.Frame):
    """縦にスクロールできる入れ物（小さい画面でも右側の操作がすべて届くように）。"""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.inner.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def to_top(self) -> None:
        self.canvas.yview_moveto(0)


class App(tk.Tk):
    def __init__(self, initial: Path | None = None):
        super().__init__()
        self.title(APP_TITLE)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1500, sw - 60)}x{min(940, sh - 90)}+20+10")
        self.minsize(1000, 640)
        self._setup_style()

        self.session: Session | None = None
        self.step = "analyze"
        self.pending: tuple[float, float] | None = None  # 引きかけのカウント線の始点（画像座標）
        self.view_scale = 1.0
        self.view_offset = (0, 0)
        self.photo: ImageTk.PhotoImage | None = None
        self.rendered = None  # 最後に描いた画像（BGR）
        self.render_dirty = True
        self.redraw_job: str | None = None
        self.save_job: str | None = None
        self.suspect_job: str | None = None
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.busy = False
        self.selected_area: int | None = None
        self.selected_line: int | None = None
        self.selected_suspect: int | None = None
        self.selected_suggestion: int | None = None
        self.area_draft: list[tuple[float, float]] | None = None  # 手で追加中のエリアの頂点
        self.vertex_drag: tuple[int, int] | None = None  # ドラッグ中の (エリア番号, 頂点番号)
        self.line_drag: tuple[int, int] | None = None  # ドラッグ中の (線番号, 0=始点/1=終点)
        self.cursor: tuple[float, float] | None = None
        self.messages: queue.Queue = queue.Queue()
        self._setting = False  # 画面の値をプログラムから設定している最中（変更イベントを無視する）
        # 連続設定モード: たくさんの解析結果を1本ずつ開いてカウント線を決めていく
        self.queue: list[Path] | None = None
        self.queue_index = 0
        self.queue_root: Path | None = None

        self._build_ui()
        self.show_step("analyze")
        self._set_busy(False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if initial is not None:
            self.after(200, lambda: self.open_path(initial))

    # ================= 見た目 =================

    def _setup_style(self) -> None:
        """文字を大きめにし、表の行を高くして読みやすくする。"""
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(size=11)
            except tk.TclError:
                pass
        base = tkfont.nametofont("TkDefaultFont")
        family = base.actual("family")
        self.font_title = tkfont.Font(family=family, size=15, weight="bold")
        self.font_bold = tkfont.Font(family=family, size=11, weight="bold")
        self.font_small = tkfont.Font(family=family, size=10)
        self.font_big = tkfont.Font(family=family, size=13, weight="bold")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=28)
        style.configure("Treeview.Heading", font=self.font_bold)
        style.configure("Step.TButton", padding=(2, 8))
        style.configure("StepDone.TButton", padding=(2, 8), foreground="#1a7f37", font=self.font_bold)
        style.configure("StepOn.TButton", padding=(4, 8), font=self.font_bold, background=ACCENT, foreground="white")
        style.map("StepOn.TButton", background=[("active", ACCENT)], foreground=[("active", "white")])
        style.configure("Primary.TButton", padding=(10, 8), font=self.font_big, background=ACCENT, foreground="white")
        style.map("Primary.TButton", background=[("active", "#1a5fd0"), ("disabled", "#9db8e8")])
        style.configure("Guide.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=self.font_title)
        style.configure("Legend.TLabel", background="#202020", foreground="#e8e8e8", padding=(10, 6))
        style.configure("Section.TLabelframe.Label", font=self.font_bold)

    # ================= 画面の組み立て =================

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(12, 8))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side=tk.LEFT)
        self.file_var = tk.StringVar(value="（動画をまだ開いていません）")
        ttk.Label(header, textvariable=self.file_var, style="Guide.TLabel").pack(side=tk.LEFT, padx=16)
        self.btn_folder = ttk.Button(header, text="保存フォルダを開く", command=self.open_result_folder)
        self.btn_folder.pack(side=tk.RIGHT)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left, background="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.legend_var = tk.StringVar()
        ttk.Label(left, textvariable=self.legend_var, style="Legend.TLabel", wraplength=900).pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.bind("<Configure>", lambda e: self.schedule_redraw())
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", lambda e: self.cancel_or_undo())
        self.canvas.bind("<Motion>", self.on_motion)
        self.bind("<Escape>", lambda e: self.cancel_or_undo())
        self.bind("<Return>", lambda e: self.finish_area_draft())
        self.bind("<Next>", lambda e: self.queue_go(1) if self.queue else None)  # PageDown
        self.bind("<Prior>", lambda e: self.queue_go(-1) if self.queue else None)  # PageUp

        side = ttk.Frame(body, padding=(10, 6), width=440)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # 連続設定中だけ表示する帯（どの動画の何本目か、次へ・前へ）
        self.queue_bar = ttk.LabelFrame(side, text="連続設定", padding=8, style="Section.TLabelframe")
        self.queue_var = tk.StringVar()
        ttk.Label(self.queue_bar, textvariable=self.queue_var, font=self.font_bold, wraplength=400).pack(anchor=tk.W)
        qrow = ttk.Frame(self.queue_bar)
        qrow.pack(fill=tk.X, pady=(6, 0))
        self.btn_q_prev = ttk.Button(qrow, text="◀ 前へ", command=lambda: self.queue_go(-1))
        self.btn_q_prev.pack(side=tk.LEFT)
        self.btn_q_next = ttk.Button(qrow, text="保存して次へ ▶", style="Primary.TButton", command=lambda: self.queue_go(1))
        self.btn_q_next.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.btn_q_end = ttk.Button(qrow, text="終了", command=self.queue_finish)
        self.btn_q_end.pack(side=tk.LEFT)
        self.carry_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.queue_bar, text="線がない動画には、前の動画の線を引き継ぐ", variable=self.carry_var).pack(
            anchor=tk.W, pady=(6, 0)
        )
        ttk.Label(self.queue_bar, text="PageDown: 保存して次へ　PageUp: 前へ\n全動画の人数は all_counts.csv に自動でまとめます",
                  style="Guide.TLabel", justify=tk.LEFT).pack(anchor=tk.W)

        steps = ttk.Frame(side)
        steps.pack(fill=tk.X)
        self.steps_frame = steps
        self.step_buttons: dict[str, ttk.Button] = {}
        for i, (key, label, _, _) in enumerate(STEPS):
            b = ttk.Button(steps, text=label, style="Step.TButton", command=lambda k=key: self.show_step(k))
            b.grid(row=0, column=i, sticky=tk.EW, padx=1)
            steps.columnconfigure(i, weight=1)
            self.step_buttons[key] = b

        self.step_title = ttk.Label(side, style="Title.TLabel")
        self.step_title.pack(anchor=tk.W, pady=(12, 2))
        self.step_guide = ttk.Label(side, style="Guide.TLabel", wraplength=410, justify=tk.LEFT)
        self.step_guide.pack(anchor=tk.W, fill=tk.X)

        nav = ttk.Frame(side)
        nav.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        self.btn_back = ttk.Button(nav, text="← 前の手順", command=lambda: self.move_step(-1))
        self.btn_back.pack(side=tk.LEFT)
        self.btn_next = ttk.Button(nav, text="次の手順 →", command=lambda: self.move_step(1))
        self.btn_next.pack(side=tk.RIGHT)

        self.scroll = ScrollFrame(side)
        self.scroll.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        holder = self.scroll.inner
        self.step_frames: dict[str, ttk.Frame] = {}
        for key, builder in (
            ("analyze", self._build_analyze),
            ("review", self._build_review),
            ("suggest", self._build_suggest),
            ("lines", self._build_lines),
            ("export", self._build_export),
        ):
            frame = ttk.Frame(holder, padding=(2, 0, 12, 8))
            builder(frame)
            self.step_frames[key] = frame

        statusbar = ttk.Frame(self, padding=(10, 4))
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="「動画を選んで解析を始める」から始めてください。")
        ttk.Label(statusbar, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        # 中止ボタンは、どの手順にいても押せるよう画面の下に置く
        self.btn_cancel = ttk.Button(statusbar, text="中止", command=self.cancel_job)
        self.btn_cancel.pack(side=tk.RIGHT, padx=(6, 0))
        self.progress = ttk.Progressbar(statusbar, length=260, maximum=100)
        self.progress.pack(side=tk.RIGHT)

    def _section(self, parent, title: str) -> ttk.LabelFrame:
        box = ttk.LabelFrame(parent, text=title, padding=10, style="Section.TLabelframe")
        box.pack(fill=tk.X, pady=(0, 10))
        return box

    # ---------- ① 解析 ----------

    def _build_analyze(self, f) -> None:
        box = self._section(f, "動画を解析する")
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="検出モデル:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=next(iter(MODELS)))
        self.model_box = ttk.Combobox(row, textvariable=self.model_var, values=list(MODELS), state="readonly", width=20)
        self.model_box.pack(side=tk.LEFT, padx=6)
        self.btn_video = ttk.Button(box, text="動画を選んで解析を始める…", style="Primary.TButton", command=self.choose_video)
        self.btn_video.pack(fill=tk.X)
        ttk.Label(box, text="解析は全フレームで行います。長い動画は時間がかかります（進み具合は画面の下に出ます）。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

        box2 = self._section(f, "解析済みの結果・たくさんの動画")
        self.btn_result = ttk.Button(box2, text="解析済みの結果を開く…", command=self.choose_result)
        self.btn_result.pack(fill=tk.X)
        self.btn_batch = ttk.Button(box2, text="まとめて処理する（一括処理）…", command=self.open_batch)
        self.btn_batch.pack(fill=tk.X, pady=(6, 0))
        self.btn_queue = ttk.Button(box2, text="結果を順にカウント線設定する（連続設定）…", command=self.start_queue)
        self.btn_queue.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(box2, text="連続設定: フォルダの中の解析結果を1本ずつ開き、線を引いて「保存して次へ」で進みます。"
                             "全動画の人数は all_counts.csv にまとめます。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

        box3 = self._section(f, "いま開いている結果")
        self.info_var = tk.StringVar(value="まだありません。")
        ttk.Label(box3, textvariable=self.info_var, wraplength=350, justify=tk.LEFT).pack(anchor=tk.W)

    # ---------- ② 確認 ----------

    def _build_review(self, f) -> None:
        box = self._section(f, "見え方")
        self.width_label = ttk.Label(box, text="追跡線の太さ")
        self.width_label.pack(anchor=tk.W)
        self.width_scale = ttk.Scale(box, from_=1, to=MAX_LINE_WIDTH, orient=tk.HORIZONTAL, command=self.on_width_change)
        self.width_scale.pack(fill=tk.X, pady=(0, 6))
        self.dim_label = ttk.Label(box, text="背景の暗さ")
        self.dim_label.pack(anchor=tk.W)
        self.dim_scale = ttk.Scale(box, from_=0, to=80, orient=tk.HORIZONTAL, command=self.on_dim_change)
        self.dim_scale.pack(fill=tk.X, pady=(0, 6))
        self.time_label = ttk.Label(box, text="表示する時間帯")
        self.time_label.pack(anchor=tk.W)
        trow = ttk.Frame(box)
        trow.pack(fill=tk.X)
        ttk.Label(trow, text="開始").grid(row=0, column=0, sticky=tk.W)
        self.start_scale = ttk.Scale(trow, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_time_change)
        self.start_scale.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Label(trow, text="終了").grid(row=1, column=0, sticky=tk.W)
        self.end_scale = ttk.Scale(trow, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_time_change)
        self.end_scale.grid(row=1, column=1, sticky=tk.EW, padx=4)
        trow.columnconfigure(1, weight=1)
        crow = ttk.Frame(box)
        crow.pack(fill=tk.X, pady=(4, 0))
        self.btn_all_time = ttk.Button(crow, text="全時間を表示", command=self.reset_time_range)
        self.btn_all_time.pack(side=tk.LEFT)
        self.ids_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(crow, text="ID番号", variable=self.ids_var, command=self.on_view_toggle).pack(side=tk.LEFT, padx=(12, 0))
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(crow, text="座標の格子", variable=self.grid_var, command=self.on_view_toggle).pack(side=tk.LEFT, padx=(8, 0))

        box2 = self._section(f, "誤判定らしい追跡線")
        ttk.Label(box2, text="灰色の点線で表示します。人数からは外しませんが、③の候補線はこれらが通らない位置に引きます。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))
        cols = ("id", "reason", "state")
        self.suspect_table = ttk.Treeview(box2, columns=cols, show="headings", height=5, selectmode="browse")
        for col, text, width in zip(cols, ("ID", "理由", "扱い"), (50, 200, 120)):
            self.suspect_table.heading(col, text=text)
            self.suspect_table.column(col, width=width, anchor=tk.CENTER)
        self.suspect_table.pack(fill=tk.X)
        self.suspect_table.tag_configure("ok", foreground=MUTED)
        self.suspect_table.bind("<<TreeviewSelect>>", self.on_suspect_select)
        self.btn_toggle_suspect = ttk.Button(box2, text="選んだ線を「誤判定ではない」にする", command=self.toggle_suspect)
        self.btn_toggle_suspect.pack(fill=tk.X, pady=(6, 0))
        self.suspect_count_var = tk.StringVar()
        ttk.Label(box2, textvariable=self.suspect_count_var).pack(anchor=tk.W, pady=(4, 0))

        box3 = self._section(f, "誤判定の判定基準")
        self.sus_vars: dict[str, tk.Variable] = {
            "jump_on": tk.BooleanVar(value=True),
            "jump_factor": tk.DoubleVar(value=6.0),
            "zigzag_on": tk.BooleanVar(value=True),
            "zigzag_straightness": tk.DoubleVar(value=0.3),
            "offpath_on": tk.BooleanVar(value=True),
            "offpath_share": tk.IntVar(value=70),
            "offpath_min_tracks": tk.IntVar(value=10),
        }
        v = self.sus_vars

        def spin(parent, var, lo, hi, step, width=5):
            sb = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=var, width=width,
                             command=self.schedule_suspect_update)
            sb.bind("<KeyRelease>", lambda e: self.schedule_suspect_update())
            return sb

        r1 = ttk.Frame(box3)
        r1.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(r1, text="瞬間移動", variable=v["jump_on"], command=self.schedule_suspect_update, width=16).pack(side=tk.LEFT)
        ttk.Label(r1, text="普通の動きの").pack(side=tk.LEFT)
        spin(r1, v["jump_factor"], 2, 30, 0.5).pack(side=tk.LEFT, padx=3)
        ttk.Label(r1, text="倍以上で飛ぶ").pack(side=tk.LEFT)
        r2 = ttk.Frame(box3)
        r2.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(r2, text="じぐざぐ", variable=v["zigzag_on"], command=self.schedule_suspect_update, width=16).pack(side=tk.LEFT)
        ttk.Label(r2, text="1秒の前進率が").pack(side=tk.LEFT)
        spin(r2, v["zigzag_straightness"], 0.05, 0.95, 0.05).pack(side=tk.LEFT, padx=3)
        ttk.Label(r2, text="未満").pack(side=tk.LEFT)
        r3 = ttk.Frame(box3)
        r3.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(r3, text="人が歩かない場所", variable=v["offpath_on"], command=self.schedule_suspect_update, width=16).pack(side=tk.LEFT)
        ttk.Label(r3, text="線の").pack(side=tk.LEFT)
        spin(r3, v["offpath_share"], 30, 100, 5).pack(side=tk.LEFT, padx=3)
        ttk.Label(r3, text="%以上").pack(side=tk.LEFT)
        r4 = ttk.Frame(box3)
        r4.pack(fill=tk.X, pady=2)
        ttk.Label(r4, text="", width=18).pack(side=tk.LEFT)
        ttk.Label(r4, text="（追跡線が").pack(side=tk.LEFT)
        spin(r4, v["offpath_min_tracks"], 3, 500, 1).pack(side=tk.LEFT, padx=3)
        ttk.Label(r4, text="本以上の動画だけ）").pack(side=tk.LEFT)
        ttk.Label(box3, text="前進率 = 1秒に進んだ距離 ÷ 歩いた道のり（まっすぐ歩くと 1）。\n"
                             "人が歩かない場所 = 他に誰も通らない所。数値を変えるとすぐ反映されます。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        self.btn_sus_reset = ttk.Button(box3, text="既定に戻す", command=self.reset_suspect_settings)
        self.btn_sus_reset.pack(anchor=tk.E, pady=(6, 0))

    # ---------- ③ 候補 ----------

    def _build_suggest(self, f) -> None:
        box = self._section(f, "1. エリアを決める")
        ttk.Label(box, text="「自動で見つける」を押すか、「手で囲む」→ 画像の上で角を順にクリック → Enter。"
                            "形は角の丸をドラッグで直せます。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(0, 6))
        self.btn_detect = ttk.Button(row, text="自動で見つける", command=self.detect_areas)
        self.btn_detect.pack(side=tk.LEFT)
        self.btn_add_area = ttk.Button(row, text="手で囲む", command=self.start_area_draft)
        self.btn_add_area.pack(side=tk.LEFT, padx=4)
        self.btn_del_area = ttk.Button(row, text="削除", command=self.delete_area)
        self.btn_del_area.pack(side=tk.LEFT)
        cols = ("area", "kind", "people")
        self.area_table = ttk.Treeview(box, columns=cols, show="headings", height=3, selectmode="browse")
        for col, text, width in zip(cols, ("エリア", "種類", "通った人"), (70, 170, 110)):
            self.area_table.heading(col, text=text)
            self.area_table.column(col, width=width, anchor=tk.CENTER)
        self.area_table.pack(fill=tk.X)
        self.area_table.bind("<<TreeviewSelect>>", self.on_area_select)
        krow = ttk.Frame(box)
        krow.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(krow, text="選んだエリアの種類:").pack(side=tk.LEFT)
        self.kind_var = tk.StringVar()
        self.kind_box = ttk.Combobox(krow, textvariable=self.kind_var, values=KIND_LABELS, state="readonly", width=14)
        self.kind_box.pack(side=tk.LEFT, padx=4)
        self.kind_box.bind("<<ComboboxSelected>>", self.on_kind_change)

        box2 = self._section(f, "2. 候補を探す")
        nrow = ttk.Frame(box2)
        nrow.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(nrow, text="評価する候補線（各エリア・最大）:").pack(side=tk.LEFT)
        self.n_var = tk.IntVar(value=20000)
        ttk.Spinbox(nrow, from_=1000, to=100000, increment=1000, textvariable=self.n_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(nrow, text="本").pack(side=tk.LEFT)
        self.btn_suggest = ttk.Button(box2, text="候補を探す（各エリア上位3本）", style="Primary.TButton", command=self.start_suggest)
        self.btn_suggest.pack(fill=tk.X)
        cols = ("cand", "people", "rate", "dup", "rel")
        self.suggest_table = ttk.Treeview(box2, columns=cols, show="headings", height=5, selectmode="browse")
        for col, text, width in zip(cols, ("候補", "数えた人", "割合", "重複", "信頼度"), (65, 80, 60, 55, 90)):
            self.suggest_table.heading(col, text=text)
            self.suggest_table.column(col, width=width, anchor=tk.CENTER)
        self.suggest_table.pack(fill=tk.X, pady=(8, 0))
        self.suggest_table.bind("<<TreeviewSelect>>", self.on_suggest_select)
        self.suggest_table.bind("<Double-Button-1>", lambda e: self.adopt_suggestion())
        self.btn_adopt = ttk.Button(box2, text="選んだ候補をカウント線にする", command=self.adopt_suggestion)
        self.btn_adopt.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(box2, text="順位 = 数えた人 × 信頼度。誤判定らしい線が通る位置は使いません。\n割合 = エリアを通った人のうち数えた割合。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

    # ---------- ④ カウント線 ----------

    def _build_lines(self, f) -> None:
        box = self._section(f, "線の引き方")
        ttk.Label(box, text="・画像を2回クリック（始点 → 終点）で線を追加\n"
                            "・線の端の丸をドラッグで位置を調整\n"
                            "・右クリック / Esc で引きかけの線を取り消し\n"
                            "・矢印の向きに通った人が IN、逆向きが OUT",
                  justify=tk.LEFT).pack(anchor=tk.W)

        box2 = self._section(f, "カウント線の一覧")
        cols = ("name", "in", "out", "total", "dup", "range", "rel")
        self.table = ttk.Treeview(box2, columns=cols, show="headings", height=5, selectmode="browse")
        for col, text, width in zip(
            cols, ("名前", "IN", "OUT", "合計", "重複", "推定", "信頼度"), (78, 38, 44, 44, 46, 62, 54)
        ):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, minwidth=30, stretch=False, anchor=tk.CENTER)
        self.table.pack(fill=tk.X)
        self.table.tag_configure("dup", background=WARN_BG)
        self.table.bind("<<TreeviewSelect>>", self.on_line_select)
        self.table.bind("<Double-Button-1>", lambda e: self.rename_line())
        brow = ttk.Frame(box2)
        brow.pack(fill=tk.X, pady=(6, 0))
        self.btn_rename = ttk.Button(brow, text="名前を変える…", command=self.rename_line)
        self.btn_rename.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_reverse = ttk.Button(brow, text="IN と OUT を入れ替え", command=self.reverse_line)
        self.btn_reverse.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        brow2 = ttk.Frame(box2)
        brow2.pack(fill=tk.X, pady=(4, 0))
        self.btn_delete = ttk.Button(brow2, text="選んだ線を削除", command=self.delete_selected_line)
        self.btn_delete.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_clear = ttk.Button(brow2, text="すべて削除", command=self.clear_lines)
        self.btn_clear.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.line_detail_var = tk.StringVar(value="線を選ぶと、信頼度の内訳と直し方を表示します。")
        ttk.Label(box2, textvariable=self.line_detail_var, justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, pady=(8, 0))

        box3 = self._section(f, "重複カウントの警告")
        rrow = ttk.Frame(box3)
        rrow.pack(fill=tk.X)
        ttk.Label(rrow, text="同じ人が").pack(side=tk.LEFT)
        self.repeat_var = tk.DoubleVar(value=1.0)
        sb = ttk.Spinbox(rrow, from_=0.2, to=10, increment=0.1, textvariable=self.repeat_var, width=5, command=self.on_repeat_change)
        sb.pack(side=tk.LEFT, padx=4)
        sb.bind("<KeyRelease>", lambda e: self.on_repeat_change())
        ttk.Label(rrow, text="秒以内に").pack(side=tk.LEFT)
        ttk.Label(box3, text="線をまたぎ直したら「重複」として警告します。").pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(box3, text="重複がある線は一覧で色が付きます。立ち止まりやすい場所を避けて線を動かすと減ります。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

    # ---------- ⑤ 書き出し ----------

    def _build_export(self, f) -> None:
        box = self._section(f, "カウント線の情報（JSON）")
        ttk.Label(box, text="名前・始点と終点の座標・IN の向き・画像の大きさを書き出します。\n"
                            "書き出した JSON は、別の動画の結果にも読み込めます（画像の大きさが違えば座標を合わせます）。",
                  style="Guide.TLabel", wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))
        self.btn_export = ttk.Button(box, text="JSON に書き出す…", style="Primary.TButton", command=self.export_json)
        self.btn_export.pack(fill=tk.X)
        self.btn_import = ttk.Button(box, text="JSON から読み込む…", command=self.import_json)
        self.btn_import.pack(fill=tk.X, pady=(6, 0))

        box2 = self._section(f, "画像")
        self.btn_save_view = ttk.Button(box2, text="いま表示している画像を保存…", command=self.save_view_image)
        self.btn_save_view.pack(fill=tk.X)

        box3 = self._section(f, "自動で保存されているもの")
        ttk.Label(box3, text="結果フォルダ（動画名_overlay）に、次のファイルが自動で保存されます。\n"
                             "・counts.csv: 線ごとの IN・OUT・合計・重複・信頼度\n"
                             "・crossings.csv: 1回ごとの通過記録\n"
                             "・count_lines.json: カウント線の情報\n"
                             "・count_result.png: カウント線と人数の画像",
                  justify=tk.LEFT, wraplength=350).pack(anchor=tk.W)
        self.btn_folder2 = ttk.Button(box3, text="結果フォルダを開く", command=self.open_result_folder)
        self.btn_folder2.pack(fill=tk.X, pady=(8, 0))

    # ================= 手順の切り替え =================

    def show_step(self, key: str) -> None:
        if key != "analyze" and self.session is None:
            self.status_var.set("先に「① 解析」で動画を解析するか、解析済みの結果を開いてください。")
            key = "analyze"
        # 手順を移るときは、引きかけの線・エリアを取り消す
        self.pending = None
        self.area_draft = None
        self.canvas.delete("pending")
        self.step = key
        for k, frame in self.step_frames.items():
            frame.pack_forget()
        self.step_frames[key].pack(fill=tk.BOTH, expand=True)
        self.scroll.to_top()
        _, _, title, guide = STEPS[STEP_KEYS.index(key)]
        self.step_title.config(text=title)
        self.step_guide.config(text=guide)
        self.legend_var.set(LEGENDS[key])
        self.update_step_buttons()
        self.canvas.config(cursor="crosshair" if key in ("suggest", "lines") else "arrow")
        self.render_dirty = True
        self.schedule_redraw()
        if key == "suggest" and self.session is not None and not self.session.areas:
            self.status_var.set("「自動で見つける」でエリアを作るか、「手で囲む」でエリアを囲んでください。")
        elif key == "lines" and self.session is not None and not self.session.lines:
            self.status_var.set("画像を2回クリック（始点 → 終点）してカウント線を引いてください。")

    def move_step(self, delta: int) -> None:
        i = STEP_KEYS.index(self.step) + delta
        if 0 <= i < len(STEP_KEYS):
            self.show_step(STEP_KEYS[i])

    def update_step_buttons(self) -> None:
        s = self.session
        done = {
            "analyze": s is not None,
            "review": s is not None,
            "suggest": bool(s and s.suggestions),
            "lines": bool(s and s.lines),
            "export": False,
        }
        for key, label, _, _ in STEPS:
            b = self.step_buttons[key]
            # 済んだ手順は緑の太字で示す（記号はフォントによって化けるので使わない）
            style = "StepOn.TButton" if key == self.step else ("StepDone.TButton" if done[key] else "Step.TButton")
            b.config(text=label, style=style)
            b.state(["disabled"] if (self.busy or (key != "analyze" and s is None)) else ["!disabled"])
        i = STEP_KEYS.index(self.step)
        self.btn_back.state(["disabled"] if i == 0 or self.busy else ["!disabled"])
        self.btn_next.state(["disabled"] if i == len(STEPS) - 1 or s is None or self.busy else ["!disabled"])

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
        partial = progress_info(out_dir, video)
        if partial is not None:
            done = partial["frames_done"]
            answer = messagebox.askyesnocancel(
                APP_TITLE,
                f"この動画の解析は前回、途中（{partial['percent']:.0f}%・{done:,}フレーム目）で止まっています。\n\n"
                "「はい」: 続きから再開する\n「いいえ」: 最初から解析し直す",
            )
            if answer is None:
                return
            self.start_analysis(video, out_dir, resume=answer)
            return
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

    def start_analysis(self, video: Path, out_dir: Path, resume: bool = False) -> None:
        self.flush_save()
        self.session = None
        self.reset_selection()
        self.rendered = read_first_frame(video)  # 解析中は最初のフレームを表示しておく
        self.render_dirty = True
        self.refresh_all_tables()
        self.show_step("analyze")
        self.file_var.set(f"解析中: {video.name}")
        self._set_busy(True)
        self.progress["value"] = 0
        self.status_var.set(
            f"{video.name}: " + ("前回の続きから再開しています…" if resume else "検出モデルを準備しています…")
        )

        model = MODELS[self.model_var.get()]
        self.stop_event = threading.Event()
        stop = self.stop_event
        started = time.time()

        first_done: list[int] = []  # 再開したときの開始位置（残り時間を正しく見積もるため）

        def progress(done: int, total: int, ids: int) -> None:
            if not first_done:
                first_done.append(max(0, done - 1))
            self.messages.put(("progress", video.name, done, total, ids, started, first_done[0]))

        def worker() -> None:
            try:
                # 途中経過を結果フォルダに保存しておき、止まっても続きから再開できるようにする
                result = track_video(
                    video,
                    model_name=model,
                    progress=progress,
                    stop_event=stop,
                    checkpoint_dir=out_dir,
                    resume=resume,
                )
                tracks = filter_short_tracks(result.tracks, MIN_POINTS)
                write_result(out_dir, video, result, tracks, smooth=SMOOTH, min_points=MIN_POINTS)
                self.messages.put(("done", out_dir, len(result.tracks), len(tracks), time.time() - started))
            except TrackingCancelled:
                self.messages.put(("cancelled",))
            except Exception as e:  # 画面に理由を出す
                self.messages.put(("error", f"{type(e).__name__}: {e}"))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
        self.after(100, self.poll_messages)

    def poll_messages(self) -> None:
        finished = False
        try:
            while True:
                msg = self.messages.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, name, done, total, ids, started, base = msg
                    percent = done * 100 / total if total else 0
                    self.progress["value"] = percent
                    elapsed = time.time() - started
                    remain = elapsed / (done - base) * (total - done) if done > base else 0
                    self.status_var.set(
                        f"{name}: 解析中 {percent:.0f}%（{done}/{total}フレーム, 追跡ID {ids}件, 残り約{_fmt_sec(remain)}）"
                    )
                elif kind == "done":
                    _, out_dir, n_all, n_drawn, elapsed = msg
                    finished = True
                    self._set_busy(False)
                    self.load_result(out_dir)
                    self.status_var.set(
                        f"解析が終わりました（{_fmt_sec(elapsed)}）。追跡線 {n_drawn}本。「次の手順 →」で②へ進んでください。"
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
                        f"候補線を {len(suggestions)} 本見つけました（{_fmt_sec(elapsed)}）。"
                        "表でダブルクリックするとカウント線になります。"
                    )
                elif kind == "cancelled":
                    finished = True
                    self.status_var.set(
                        "中止しました。途中まで保存したので、同じ動画をもう一度選ぶと続きから再開できます。"
                        if self.session is None
                        else "中止しました。"
                    )
                    if self.session is None:
                        self.file_var.set("（動画をまだ開いていません）")
                elif kind == "error":
                    finished = True
                    self.status_var.set("失敗しました。")
                    messagebox.showerror(APP_TITLE, f"処理に失敗しました。\n\n{msg[1]}")
        except queue.Empty:
            pass
        if finished:
            self._set_busy(False)
            self.progress["value"] = 100 if self.session else 0
        else:
            self.after(100, self.poll_messages)

    def cancel_job(self) -> None:
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
        self.reset_selection()
        self.title(f"{APP_TITLE} - {session.video_name}")
        self.file_var.set(f"{session.video_name}（結果: {session.dir.name}）")
        self._setting = True
        self.width_scale.set(session.line_width)
        self.dim_scale.set(round(session.dim * 100))
        self.ids_var.set(session.show_ids)
        self.grid_var.set(session.grid > 0)
        duration = max(session.duration, 0.1)
        self.start_scale.configure(to=duration)
        self.end_scale.configure(to=duration)
        self.start_scale.set(0)
        self.end_scale.set(duration)
        self.repeat_var.set(round(session.repeat_sec, 1))
        cfg = session.suspect_settings
        for k, var in self.sus_vars.items():
            value = getattr(cfg, k)
            var.set(round(value * 100) if k == "offpath_share" else value)
        self._setting = False
        self._update_view_labels()
        self.update_info()
        self.render_dirty = True
        self.refresh_all_tables()
        self.update_step_buttons()
        self.schedule_redraw()
        self.schedule_save()
        self.status_var.set("結果を開きました。「次の手順 →」で進んでください。")

    def update_info(self) -> None:
        s = self.session
        if s is None:
            self.info_var.set("まだありません。")
            return
        self.info_var.set(
            f"動画: {s.video_name}\n長さ: {_fmt_clock(s.duration)}（{s.width}×{s.height}）\n"
            f"追跡線: {len(s.tracks)}本（うち誤判定らしい線 {len(s.suspects)}本）\n"
            f"カウント線: {len(s.lines)}本"
        )

    def open_batch(self) -> None:
        """一括処理の画面を開く。"""
        from .batch_dialog import BatchDialog

        self.flush_save()
        width = self.session.line_width if self.session is not None else 1
        BatchDialog(self, MODELS, self.model_var.get(), width, open_in_file_manager, self.load_result, self.start_queue)

    # ================= 連続設定 =================

    def start_queue(self, root: Path | None = None) -> None:
        """フォルダの中の解析結果を、1本ずつ順にカウント線設定していく。"""
        from .batch import find_result_dirs

        if root is None:
            chosen = filedialog.askdirectory(title="解析結果の入ったフォルダを選択（一括処理の出力フォルダなど）")
            if not chosen:
                return
            root = Path(chosen)
        dirs = find_result_dirs(root)
        if not dirs:
            messagebox.showinfo(APP_TITLE, "このフォルダには解析結果（動画名_overlay）がありません。先に解析してください。")
            return
        self.flush_save()
        self.queue, self.queue_root = dirs, root
        self.queue_bar.pack(fill=tk.X, pady=(0, 8), before=self.steps_frame)
        self.queue_open(0, carry=None)
        self.status_var.set(f"{len(dirs)}本の結果を順に設定します。線を引いたら「保存して次へ」（PageDown）で進んでください。")

    def queue_open(self, index: int, carry: tuple[list[CountLine], int, int] | None) -> None:
        self.queue_index = index
        self.load_result(self.queue[index])
        s = self.session
        if s is not None and carry is not None and not s.lines and carry[0]:
            from .counting import scale_lines

            lines, w, h = carry
            s.set_lines(scale_lines(lines, w, h, s.width, s.height))
            self.lines_changed()
            self.status_var.set("前の動画の線を引き継ぎました。位置がずれていれば端の丸をドラッグで直してください。")
        self.update_queue_bar()
        self.show_step("lines")

    def queue_go(self, delta: int) -> None:
        if not self.queue or self.busy:
            return
        i = self.queue_index + delta
        if not 0 <= i < len(self.queue):
            if delta > 0:
                self.queue_finish()
            return
        self.pending = None
        carry = None
        if self.session is not None and self.carry_var.get():
            carry = ([CountLine(ln.x1, ln.y1, ln.x2, ln.y2, ln.name) for ln in self.session.lines],
                     self.session.width, self.session.height)
        self.flush_save()
        self.write_queue_summary()
        self.queue_open(i, carry if delta > 0 else None)

    def update_queue_bar(self) -> None:
        if not self.queue:
            return
        # 違うフォルダに同じ名前の動画があっても区別できるよう、結果フォルダの名前を出す
        d = self.queue[self.queue_index]
        name = d.name[: -len("_overlay")] if d.name.endswith("_overlay") else d.name
        self.queue_var.set(f"{self.queue_index + 1} / {len(self.queue)} 本目: {name}")
        self.btn_q_prev.state(["disabled"] if self.queue_index == 0 else ["!disabled"])
        last = self.queue_index == len(self.queue) - 1
        self.btn_q_next.config(text="保存して終了 ✔" if last else "保存して次へ ▶")

    def write_queue_summary(self) -> Path | None:
        """全動画の一覧 CSV（all_counts.csv）を書く。"""
        if not self.queue or self.queue_root is None:
            return None
        from .batch import write_all_counts

        path = self.queue_root / "all_counts.csv"
        try:
            write_all_counts(path, self.queue)
        except OSError as e:
            self.status_var.set(f"一覧 CSV を保存できませんでした: {e}")
            return None
        return path

    def queue_finish(self) -> None:
        if not self.queue:
            return
        self.flush_save()
        path = self.write_queue_summary()
        n = len(self.queue)
        self.queue = None
        self.queue_bar.pack_forget()
        if path is not None:
            self.status_var.set(f"連続設定を終えました。{n}本分の人数を {path} にまとめました。")
            if messagebox.askyesno(APP_TITLE, f"連続設定を終えました。\n{n}本分の人数を一覧にまとめました:\n{path}\n\nフォルダを開きますか？"):
                open_in_file_manager(path.parent)

    # ================= ② 見え方・誤判定 =================

    def on_width_change(self, _value=None) -> None:
        if self.session is None or self._setting:
            return
        width = int(round(float(self.width_scale.get())))
        if width != self.session.line_width:
            self.session.line_width = width
            self.view_changed()
        self._update_view_labels()

    def on_dim_change(self, _value=None) -> None:
        if self.session is None or self._setting:
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
        self.view_changed()

    def _update_view_labels(self) -> None:
        s = self.session
        if s is None:
            return
        self.width_label.config(text=f"追跡線の太さ: {s.line_width}")
        self.dim_label.config(text=f"背景の暗さ: {round(s.dim * 100)}%")
        shown, total = len(s.visible_tracks()), len(s.tracks)
        if s.is_time_filtered():
            text = f"表示する時間帯: {_fmt_clock(s.time_start)}〜{_fmt_clock(s.time_end)}（{shown}/{total}本を表示）"
        else:
            text = f"表示する時間帯: 全時間 0:00〜{_fmt_clock(s.duration)}（{total}本）"
        self.time_label.config(text=text)

    def on_time_change(self, _value=None) -> None:
        if self.session is None or self._setting:
            return
        start = float(self.start_scale.get())
        end = float(self.end_scale.get())
        if start > end:  # 開始が終了を追い越したら、もう一方も動かして揃える
            self._setting = True
            if abs(start - self.session.time_start) > 1e-9:
                self.end_scale.set(start)
                end = start
            else:
                self.start_scale.set(end)
                start = end
            self._setting = False
        if end >= self.session.duration - 0.05:  # 動画の端まで寄せたときは「全時間」と同じ扱い
            end = self.session.duration
        if (start, end) != (self.session.time_start, self.session.time_end):
            self.session.time_start, self.session.time_end = start, end
            self.render_dirty = True
            self.schedule_redraw()  # 表示だけの変更なので保存はしない
        self._update_view_labels()

    def reset_time_range(self) -> None:
        if self.session is None:
            return
        self._setting = True
        self.start_scale.set(0)
        self.end_scale.set(self.session.duration)
        self._setting = False
        self.on_time_change()

    def refresh_suspect_table(self) -> None:
        self.suspect_table.delete(*self.suspect_table.get_children())
        s = self.session
        if s is None:
            self.suspect_count_var.set("")
            return
        found = s.suspects_found
        for tid in sorted(found):
            ok = tid in s.suspect_ok
            self.suspect_table.insert(
                "", tk.END, iid=str(tid), values=(tid, reason_text(found[tid]), "誤判定ではない" if ok else "誤判定らしい"),
                tags=("ok",) if ok else (),
            )
        if self.selected_suspect is not None and str(self.selected_suspect) in self.suspect_table.get_children():
            self.suspect_table.selection_set(str(self.selected_suspect))
        else:
            self.selected_suspect = None
        self.suspect_count_var.set(f"誤判定らしい線: {len(s.suspects)}本 / 全{len(s.tracks)}本")
        self._update_toggle_label()

    def _update_toggle_label(self) -> None:
        s = self.session
        if s is not None and self.selected_suspect in s.suspect_ok:
            self.btn_toggle_suspect.config(text="選んだ線を「誤判定らしい」に戻す")
        else:
            self.btn_toggle_suspect.config(text="選んだ線を「誤判定ではない」にする")

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
        self.suspects_changed()

    def schedule_suspect_update(self) -> None:
        """判定基準の入力が止まってから検知し直す（数字を打っている途中で何度も計算しないように）。"""
        if self._setting or self.session is None:
            return
        if self.suspect_job is not None:
            self.after_cancel(self.suspect_job)
        self.suspect_job = self.after(400, self.apply_suspect_settings)

    def apply_suspect_settings(self) -> None:
        self.suspect_job = None
        if self.session is None:
            return
        v = self.sus_vars
        try:
            cfg = SuspectSettings(
                jump_on=bool(v["jump_on"].get()),
                jump_factor=float(v["jump_factor"].get()),
                zigzag_on=bool(v["zigzag_on"].get()),
                zigzag_straightness=float(v["zigzag_straightness"].get()),
                offpath_on=bool(v["offpath_on"].get()),
                offpath_share=float(v["offpath_share"].get()) / 100,
                offpath_min_tracks=int(v["offpath_min_tracks"].get()),
            )
        except (tk.TclError, ValueError):
            return  # 入力の途中（空欄など）は無視する
        self.session.set_suspect_settings(cfg)
        self.suspects_changed()
        self.status_var.set(f"判定基準を変えました。誤判定らしい線は {len(self.session.suspects)}本です。")

    def reset_suspect_settings(self) -> None:
        if self.session is None:
            return
        cfg = SuspectSettings()
        self._setting = True
        for k, var in self.sus_vars.items():
            value = getattr(cfg, k)
            var.set(round(value * 100) if k == "offpath_share" else value)
        self._setting = False
        self.apply_suspect_settings()

    def suspects_changed(self) -> None:
        self.refresh_suspect_table()
        self.refresh_table()
        self.update_info()
        self.render_dirty = True
        self.schedule_redraw()
        self.schedule_save()
        if self.session is not None and self.session.suggestions:
            self.status_var.set("誤判定の扱いを変えました。候補線に反映するには、③でもう一度「候補を探す」を押してください。")

    # ================= 画像の上のマウス操作 =================

    def to_image(self, x: float, y: float) -> tuple[float, float] | None:
        """画面上の座標 → 元画像の座標。画像の外なら None。"""
        if self.session is None:
            return None
        ix = (x - self.view_offset[0]) / self.view_scale
        iy = (y - self.view_offset[1]) / self.view_scale
        if 0 <= ix < self.session.width and 0 <= iy < self.session.height:
            return ix, iy
        return None

    def to_image_clamped(self, x: float, y: float) -> tuple[float, float]:
        ix = (x - self.view_offset[0]) / self.view_scale
        iy = (y - self.view_offset[1]) / self.view_scale
        return min(max(ix, 0), self.session.width - 1), min(max(iy, 0), self.session.height - 1)

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.view_scale + self.view_offset[0], iy * self.view_scale + self.view_offset[1]

    def on_press(self, event) -> None:
        if self.session is None or self.busy:
            return
        if self.step == "suggest":
            self.on_area_press(event)
        elif self.step == "lines":
            self.on_line_press(event)

    def on_drag(self, event) -> None:
        if self.session is None:
            return
        if self.vertex_drag is not None:
            ai, vi = self.vertex_drag
            self.session.areas[ai].polygon[vi] = list(self.to_image_clamped(event.x, event.y))
            self.render_dirty = True
            self.schedule_redraw()
        elif self.line_drag is not None:
            li, which = self.line_drag
            x, y = self.to_image_clamped(event.x, event.y)
            self.session.move_endpoint(li, which, x, y, recount=False)  # ドラッグ中は描き直しだけ（軽くする）
            self.render_dirty = True
            self.schedule_redraw()

    def on_release(self, _event=None) -> None:
        if self.session is None:
            return
        if self.vertex_drag is not None:
            ai, _ = self.vertex_drag
            self.vertex_drag = None
            had = any(sg.area_index == ai for sg in self.session.suggestions)
            self.session.area_changed(ai)
            self.areas_changed()
            if had:
                self.status_var.set(f"E{ai + 1} の形を変えたので、このエリアの候補線を消しました。もう一度「候補を探す」を押してください。")
        elif self.line_drag is not None:
            li, _ = self.line_drag
            self.line_drag = None
            self.session.recount()
            self.lines_changed()
            c = self.session.counts[li]
            self.status_var.set(f"{self.session.lines[li].label(li)} を動かしました: IN {c.in_count}人 / OUT {c.out_count}人 / 合計 {c.total}人")

    def on_double_click(self, _event=None) -> None:
        if self.step == "suggest" and self.area_draft is not None:
            self.finish_area_draft()

    def on_motion(self, event) -> None:
        self.cursor = (event.x, event.y)
        if self.pending is not None:
            self.draw_pending(event.x, event.y)
        elif self.area_draft is not None:
            self.draw_area_draft()
        elif self.session is not None and self.step in ("lines", "suggest"):
            # つかめる丸の上ではカーソルを手の形にする
            grab = (self.pick_endpoint(event.x, event.y) if self.step == "lines" else self.pick_vertex(event.x, event.y))
            self.canvas.config(cursor="hand2" if grab is not None else "crosshair")

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

    # ---------- ④ カウント線の操作 ----------

    def pick_endpoint(self, sx: float, sy: float) -> tuple[int, int] | None:
        """画面座標の近くにあるカウント線の端 (線番号, 0=始点/1=終点)。"""
        if self.session is None:
            return None
        best, best_d = None, PICK_PX
        for li, ln in enumerate(self.session.lines):
            for which, (x, y) in enumerate(((ln.x1, ln.y1), (ln.x2, ln.y2))):
                px, py = self.to_screen(x, y)
                d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = (li, which), d
        return best

    def on_line_press(self, event) -> None:
        if self.pending is None:
            picked = self.pick_endpoint(event.x, event.y)
            if picked is not None:
                self.line_drag = picked
                self.select_line(picked[0])
                return
        point = self.to_image(event.x, event.y)
        if point is None:
            return
        if self.pending is None:
            self.pending = point
            self.draw_pending(event.x, event.y)
            self.status_var.set("終点をクリックしてください（右クリック / Esc で取り消し）。")
            return
        x1, y1 = self.pending
        if self.session.add_line(CountLine(round(x1), round(y1), round(point[0]), round(point[1]))):
            self.pending = None
            self.canvas.delete("pending")
            self.selected_line = len(self.session.lines) - 1
            self.lines_changed()
            c = self.session.counts[-1]
            name = self.session.lines[-1].label(len(self.session.lines) - 1)
            self.status_var.set(
                f"{name} を追加しました: IN {c.in_count}人 / OUT {c.out_count}人 / 合計 {c.total}人。"
                "向きが逆なら「IN と OUT を入れ替え」を押してください。"
            )

    def draw_pending(self, x: float, y: float) -> None:
        """引きかけの線（始点からマウス位置まで）を画面上に仮表示する。"""
        self.canvas.delete("pending")
        if self.pending is None or self.session is None:
            return
        sx, sy = self.to_screen(*self.pending)
        b, g, r = line_color(len(self.session.lines))
        color = f"#{r:02x}{g:02x}{b:02x}"
        self.canvas.create_line(sx, sy, x, y, fill="black", width=6, tags="pending")
        self.canvas.create_line(sx, sy, x, y, fill=color, width=4, tags="pending", dash=(8, 4))
        self.canvas.create_oval(sx - 6, sy - 6, sx + 6, sy + 6, fill=color, outline="black", tags="pending")

    def select_line(self, index: int | None) -> None:
        self.selected_line = index
        items = self.table.get_children()
        if index is not None and index < len(items):
            self.table.selection_set(items[index])
            self.table.see(items[index])
        self.update_line_detail()
        self.render_dirty = True
        self.schedule_redraw()

    def on_line_select(self, _event=None) -> None:
        sel = self.table.selection()
        if not sel or self.session is None:
            return
        index = self.table.index(sel[0])
        if index != self.selected_line:
            self.select_line(index)

    def _need_line(self) -> int | None:
        if self.session is None or self.selected_line is None or self.selected_line >= len(self.session.lines):
            messagebox.showinfo(APP_TITLE, "一覧から線を選んでください。")
            return None
        return self.selected_line

    def rename_line(self) -> None:
        i = self._need_line()
        if i is None:
            return
        ln = self.session.lines[i]
        name = simpledialog.askstring(APP_TITLE, "カウント線の名前（例: 改札口、1階エスカレーター）:",
                                      initialvalue=ln.label(i), parent=self)
        if name is None:
            return
        self.session.rename_line(i, name)
        self.lines_changed()

    def reverse_line(self) -> None:
        i = self._need_line()
        if i is None:
            return
        self.session.reverse_line(i)
        self.lines_changed()
        c = self.session.counts[i]
        self.status_var.set(f"{self.session.lines[i].label(i)} の IN と OUT を入れ替えました: IN {c.in_count}人 / OUT {c.out_count}人")

    def delete_selected_line(self) -> None:
        i = self._need_line()
        if i is None:
            return
        self.session.remove_line(i)
        self.selected_line = None
        self.lines_changed()

    def clear_lines(self) -> None:
        if self.session is None or not self.session.lines:
            return
        if messagebox.askyesno(APP_TITLE, "カウント線をすべて削除しますか？"):
            self.session.clear()
            self.selected_line = None
            self.lines_changed()

    def on_repeat_change(self) -> None:
        if self.session is None or self._setting:
            return
        try:
            value = float(self.repeat_var.get())
        except (tk.TclError, ValueError):
            return
        if value > 0 and abs(value - self.session.repeat_sec) > 1e-9:
            self.session.set_repeat_sec(value)
            self.lines_changed()

    def lines_changed(self) -> None:
        self.render_dirty = True
        self.refresh_table()
        self.update_info()
        self.update_step_buttons()
        self.schedule_redraw()
        self.schedule_save()

    def refresh_table(self) -> None:
        self.table.delete(*self.table.get_children())
        s = self.session
        if s is None:
            self.update_line_detail()
            return
        for i, c in enumerate(s.counts):
            ev = s.evaluations[i] if i < len(s.evaluations) else None
            dup = ev.duplicates if ev else 0
            values = (
                s.lines[i].label(i), c.in_count, c.out_count, c.total,
                f"⚠ {dup}" if dup else "0",
                f"{ev.range_text}" if ev else "",
                f"{ev.reliability:.0f} {ev.grade}" if ev else "",
            )
            self.table.insert("", tk.END, values=values, tags=("dup",) if dup else ())
        if self.selected_line is not None and self.selected_line < len(s.lines):
            self.table.selection_set(self.table.get_children()[self.selected_line])
        else:
            self.selected_line = None
        self.update_line_detail()

    def update_line_detail(self) -> None:
        """選んだカウント線の内訳（重複・信頼度が低い理由と直し方）を表示する。"""
        s = self.session
        if s is None or self.selected_line is None or self.selected_line >= len(s.evaluations):
            self.line_detail_var.set("線を選ぶと、信頼度の内訳と直し方を表示します。")
            return
        i = self.selected_line
        ev, c = s.evaluations[i], s.counts[i]
        text = (f"{s.lines[i].label(i)}: IN {c.in_count}人 / OUT {c.out_count}人 / 合計 {c.total}人\n"
                f"推定 {ev.range_text}人・信頼度 {ev.reliability:.0f}点（{ev.grade}）")
        if ev.duplicates:
            text += f"\n⚠ 同じ人を {s.repeat_sec:g}秒以内に重ねて数えた回数: {ev.duplicates}回"
        if ev.reasons:
            for reason, hint in zip(ev.reasons, ev.hints):
                text += f"\n・{reason}\n　→ {hint}"
        else:
            text += "\n数え間違いの痕跡はほとんどありません。"
        self.line_detail_var.set(text)

    # ---------- ③ エリア・候補 ----------

    def reset_selection(self) -> None:
        self.pending = None
        self.selected_area = None
        self.selected_suggestion = None
        self.selected_line = None
        self.selected_suspect = None
        self.area_draft = None
        self.vertex_drag = None
        self.line_drag = None

    def areas_changed(self) -> None:
        self.refresh_area_tables()
        self.refresh_table()  # カウント線の信頼度はエリアの形でも変わる
        self.update_step_buttons()
        self.render_dirty = True
        self.schedule_redraw()
        self.schedule_save()

    def refresh_area_tables(self) -> None:
        self.area_table.delete(*self.area_table.get_children())
        self.suggest_table.delete(*self.suggest_table.get_children())
        s = self.session
        if s is None:
            return
        for i, area in enumerate(s.areas):
            self.area_table.insert("", tk.END, values=(f"E{i + 1}", area.kind_label, f"{s.area_people(i)}人"))
        for sg in s.suggestions:
            self.suggest_table.insert(
                "", tk.END,
                values=(f"E{sg.area_index + 1}-{sg.rank}", f"{sg.people}人", f"{sg.coverage:.0%}", sg.duplicates,
                        f"{sg.reliability:.0f} {sg.grade}" if sg.grade else ""),
            )
        if self.selected_area is not None and self.selected_area < len(s.areas):
            self.area_table.selection_set(self.area_table.get_children()[self.selected_area])
            self.kind_var.set(s.areas[self.selected_area].kind_label)
        else:
            self.selected_area = None
            self.kind_var.set("")

    def refresh_all_tables(self) -> None:
        self.refresh_table()
        self.refresh_area_tables()
        self.refresh_suspect_table()

    def detect_areas(self) -> None:
        if self.session is None:
            return
        if self.session.areas and not messagebox.askyesno(APP_TITLE, "今のエリアと候補線を消して、エリアを見つけ直しますか？"):
            return
        self.session.auto_detect_areas()
        self.selected_area = None
        self.selected_suggestion = None
        self.areas_changed()
        n = len(self.session.areas)
        if n:
            self.status_var.set(f"エリアを {n} 件見つけました。種類を選び、形がずれていれば角の丸をドラッグで直してください。")
        else:
            self.status_var.set("人がよく通る場所が見つかりませんでした。「手で囲む」でエリアを囲んでください。")

    def start_area_draft(self) -> None:
        if self.session is None:
            return
        self.area_draft = []
        self.status_var.set("エリアの角を順にクリックしてください。Enter（またはダブルクリック）で確定、右クリックで1つ戻す。")

    def finish_area_draft(self) -> None:
        if self.area_draft is None or self.session is None:
            return
        pts: list[tuple[float, float]] = []
        for p in self.area_draft:  # ダブルクリックで同じ点が重なって入るので取り除く
            if not pts or abs(p[0] - pts[-1][0]) + abs(p[1] - pts[-1][1]) > 3:
                pts.append(p)
        if len(pts) < 3:
            self.status_var.set("エリアには角が3つ以上必要です。続けてクリックしてください。")
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
        if len(pts) >= 2:
            self.canvas.create_line(*[v for p in pts for v in p], fill="#50c8ff", width=3, dash=(6, 3), tags="pending")
        for x, y in pts[: len(self.area_draft)]:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#50c8ff", outline="black", tags="pending")

    def pick_vertex(self, sx: float, sy: float) -> tuple[int, int] | None:
        """画面座標の近くにあるエリアの角 (エリア番号, 角の番号)。"""
        if self.session is None:
            return None
        best, best_d = None, PICK_PX
        for ai, area in enumerate(self.session.areas):
            for vi, (x, y) in enumerate(area.polygon):
                px, py = self.to_screen(x, y)
                d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = (ai, vi), d
        return best

    def on_area_press(self, event) -> None:
        if self.area_draft is not None:
            point = self.to_image(event.x, event.y)
            if point is not None:
                self.area_draft.append(point)
                self.draw_area_draft()
            return
        picked = self.pick_vertex(event.x, event.y)
        if picked is not None:
            self.vertex_drag = picked
            self.select_area(picked[0])
            return
        point = self.to_image(event.x, event.y)
        if point is not None:
            for i, area in enumerate(self.session.areas):
                if area.contains(*point):
                    self.select_area(i)
                    return

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
            messagebox.showinfo(APP_TITLE, "一覧から削除するエリアを選んでください。")
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
                messagebox.showinfo(APP_TITLE, "エリアが見つかりませんでした。「手で囲む」でエリアを囲んでください。")
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
            messagebox.showinfo(APP_TITLE, "一覧から候補を選んでください。")
            return
        sg = self.session.suggestions[self.selected_suggestion]
        if self.session.adopt_suggestion(self.selected_suggestion):
            self.selected_line = len(self.session.lines) - 1
            self.lines_changed()
            c = self.session.counts[-1]
            name = self.session.lines[-1].label(len(self.session.lines) - 1)
            self.status_var.set(
                f"候補 E{sg.area_index + 1}-{sg.rank} を {name} にしました: IN {c.in_count}人 / OUT {c.out_count}人。"
                "④で名前と向きを決められます。"
            )

    # ================= ⑤ 書き出し =================

    def export_json(self) -> None:
        s = self.session
        if s is None:
            return
        if not s.lines:
            messagebox.showinfo(APP_TITLE, "カウント線がまだありません。④で線を引いてください。")
            return
        path = filedialog.asksaveasfilename(
            title="カウント線を JSON に書き出す",
            initialdir=str(s.dir),
            initialfile=f"{Path(s.video_name).stem}_count_lines.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            s.export_lines(Path(path))
            self.status_var.set(f"書き出しました: {path}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"書き出しに失敗しました。\n\n{e}")

    def import_json(self) -> None:
        s = self.session
        if s is None:
            return
        path = filedialog.askopenfilename(title="カウント線の JSON を読み込む", initialdir=str(s.dir),
                                          filetypes=[("JSON", "*.json"), ("すべてのファイル", "*.*")])
        if not path:
            return
        replace = True
        if s.lines:
            answer = messagebox.askyesnocancel(
                APP_TITLE, "いまのカウント線をどうしますか？\n\n「はい」: 置き換える\n「いいえ」: 残したまま追加する")
            if answer is None:
                return
            replace = bool(answer)
        try:
            note = s.import_lines(Path(path), replace=replace)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"読み込めませんでした。\n\n{e}")
            return
        self.selected_line = None
        self.lines_changed()
        self.status_var.set(f"{len(s.lines)}本のカウント線を読み込みました。{note}")
        if note:
            messagebox.showinfo(APP_TITLE, note)

    def save_view_image(self) -> None:
        """いま表示している画像を保存する。"""
        s = self.session
        if s is None:
            return
        name = f"{Path(s.video_name).stem}_view"
        if s.is_time_filtered():
            name += f"_{_fmt_clock(s.time_start).replace(':', 'm')}s-{_fmt_clock(s.time_end).replace(':', 'm')}s"
        path = filedialog.asksaveasfilename(
            title="いま表示している画像を保存",
            initialdir=str(s.dir),
            initialfile=name + ".png",
            defaultextension=".png",
            filetypes=[("PNG 画像", "*.png"), ("JPEG 画像", "*.jpg")],
        )
        if not path:
            return
        try:
            save_image(Path(path), self.render_current())
            self.status_var.set(f"保存しました: {path}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"保存に失敗しました。\n\n{e}")

    # ================= 描画・保存 =================

    def render_current(self):
        """いまの手順に合わせて、画像の上に描くものを選んで描く（手順ごとに必要なものだけ）。"""
        s = self.session
        step = self.step
        s.show_areas = step == "suggest"
        return s.render(
            self.selected_area if step == "suggest" else None,
            self.selected_suggestion if step == "suggest" else None,
            selected_line=self.selected_line if step == "lines" else None,
            highlight_track=self.selected_suspect if step == "review" else None,
            show_lines=step in ("suggest", "lines", "export"),
        )

    def view_changed(self) -> None:
        self.render_dirty = True
        self.schedule_redraw()
        self.schedule_save()

    def schedule_redraw(self) -> None:
        if self.redraw_job is None:  # 描き直しが溜まらないよう、まとめて1回にする
            self.redraw_job = self.after(15, self.redraw)

    def redraw(self) -> None:
        self.redraw_job = None
        self.canvas.delete("image")
        self.canvas.delete("hint")
        if self.session is None and self.rendered is None:
            # まだ何も開いていないときは、画像の場所に次にすることを書いておく
            cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
            self.canvas.create_text(
                cw / 2, ch / 2, fill="#c8c8c8", font=self.font_big, tags="hint", justify=tk.CENTER,
                text="右の「動画を選んで解析を始める…」で動画を選ぶと\nここに足元の通り道（追跡線）が表示されます",
            )
            return
        if self.session is not None and (self.render_dirty or self.rendered is None):
            self.rendered = self.render_current()
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
        if self.save_job is not None:  # 操作が止まってから保存する（操作のたびに書き出すと重い）
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
        else:
            messagebox.showinfo(APP_TITLE, "まだ結果を開いていません。")

    def is_busy(self) -> bool:
        return self.busy

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = ["disabled"] if busy else ["!disabled"]
        for w in (
            self.btn_video, self.btn_result, self.btn_batch, self.btn_folder, self.btn_folder2,
            self.btn_toggle_suspect, self.btn_sus_reset, self.btn_all_time,
            self.btn_detect, self.btn_add_area, self.btn_del_area, self.btn_suggest, self.btn_adopt,
            self.btn_rename, self.btn_reverse, self.btn_delete, self.btn_clear,
            self.btn_export, self.btn_import, self.btn_save_view, self.btn_queue,
            self.btn_q_prev, self.btn_q_next, self.btn_q_end,
        ):
            w.state(state)
        self.model_box.state(["disabled"] if busy else ["!disabled", "readonly"])
        self.btn_cancel.state(["!disabled"] if busy else ["disabled"])
        if busy:
            self.btn_cancel.config(text="候補探しを中止" if self.session is not None else "解析を中止")
        else:
            self.btn_cancel.config(text="中止")
        self.update_step_buttons()
        if not busy:
            self.update_queue_bar()

    def on_close(self) -> None:
        if self.stop_event is not None and not self.stop_event.is_set() and self.busy:
            if not messagebox.askyesno(APP_TITLE, "処理中です。中止して終了しますか？"):
                return
            self.stop_event.set()
            # 解析中なら、途中経過を保存し終えるまで少し待つ（次回続きから再開できるように）
            if self.worker is not None and self.worker.is_alive():
                self.status_var.set("途中経過を保存しています…")
                self.update_idletasks()
                self.worker.join(timeout=15)
        self.flush_save()
        if self.queue:
            self.write_queue_summary()
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
