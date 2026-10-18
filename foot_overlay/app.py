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
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .counting import CountLine
from .drawing import filter_short_tracks
from .session import Session, line_color, write_result
from .tracking import TrackingCancelled, track_video

APP_TITLE = "足元軌跡カウントツール"
VIDEO_TYPES = [("動画ファイル", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"), ("すべてのファイル", "*.*")]
MODELS = {
    "標準（速い）": "yolo11n.pt",
    "高精度（約3倍遅い）": "yolo11s.pt",
}
MIN_POINTS = 10  # この回数未満しか検出されなかったIDはノイズとして除く
SMOOTH = 5
MAX_LINE_WIDTH = 15


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
        self.geometry("1280x800")
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
        ttk.Label(toolbar, text="検出モデル:").pack(side=tk.LEFT, padx=(16, 4))
        self.model_var = tk.StringVar(value=next(iter(MODELS)))
        self.model_box = ttk.Combobox(toolbar, textvariable=self.model_var, values=list(MODELS), state="readonly", width=18)
        self.model_box.pack(side=tk.LEFT)
        self.btn_cancel = ttk.Button(toolbar, text="解析を中止", command=self.cancel_analysis)
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
        self.bind("<Escape>", lambda e: self.cancel_or_undo())
        self.bind("<Control-z>", lambda e: self.undo_line())

        side = ttk.Frame(body, padding=8, width=320)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        self._build_view_panel(side)
        self._build_count_panel(side)

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

        self.ids_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="追跡ID番号を表示", variable=self.ids_var, command=self.on_view_toggle).pack(anchor=tk.W)
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="座標の格子を表示（100px）", variable=self.grid_var, command=self.on_view_toggle).pack(
            anchor=tk.W
        )

    def _build_count_panel(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="カウント線", padding=8)
        box.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(
            box,
            text="画像を2回クリックで線を追加\n右クリック / Esc: 引きかけの線を取り消し\n"
            "Ctrl+Z: 最後の線を削除\nA→B は線の中央の矢印の向きの通過人数",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        columns = ("line", "ab", "ba", "total")
        self.table = ttk.Treeview(box, columns=columns, show="headings", height=8, selectmode="browse")
        for col, text, width in zip(columns, ("線", "A→B", "B→A", "合計"), (50, 70, 70, 70)):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(box)
        buttons.pack(fill=tk.X, pady=(6, 0))
        self.btn_delete = ttk.Button(buttons, text="選んだ線を削除", command=self.delete_selected_line)
        self.btn_delete.pack(side=tk.LEFT)
        self.btn_clear = ttk.Button(buttons, text="すべて削除", command=self.clear_lines)
        self.btn_clear.pack(side=tk.RIGHT)

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
        self.render_dirty = True
        self.refresh_table()
        self.canvas.delete("all")
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
                elif kind == "cancelled":
                    finished = True
                    self.status_var.set("解析を中止しました。")
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
        self.title(f"{APP_TITLE} - {session.video_name}")
        self.width_scale.set(session.line_width)
        self.dim_scale.set(round(session.dim * 100))
        self.ids_var.set(session.show_ids)
        self.grid_var.set(session.grid > 0)
        self._update_view_labels()
        self.render_dirty = True
        self.refresh_table()
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
        self.view_changed()

    def _update_view_labels(self) -> None:
        if self.session is None:
            return
        self.width_label.config(text=f"追跡線の太さ: {self.session.line_width}")
        self.dim_label.config(text=f"背景の暗さ: {round(self.session.dim * 100)}%")

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

    def on_left_click(self, event) -> None:
        if self.session is None:
            return
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
        if self.pending is not None:
            self.pending = None
            self.canvas.delete("pending")
            self.status_var.set("線の追加を取り消しました。")

    def undo_line(self) -> None:
        if self.session is not None and self.session.lines:
            self.session.undo()
            self.lines_changed()

    def on_motion(self, event) -> None:
        if self.pending is not None:
            self.draw_pending(event.x, event.y)

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
            self.table.insert("", tk.END, values=(f"L{i}", c.a_to_b, c.b_to_a, c.total))

    # ================= 描画・保存 =================

    def schedule_redraw(self) -> None:
        # スライダーを動かしている間に描き直しが溜まらないよう、まとめて1回にする
        if self.redraw_job is None:
            self.redraw_job = self.after(15, self.redraw)

    def redraw(self) -> None:
        self.redraw_job = None
        self.canvas.delete("image")
        if self.session is None:
            return
        if self.render_dirty or self.rendered is None:
            self.rendered = self.session.render()
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
        for w in (self.btn_video, self.btn_result, self.btn_delete, self.btn_clear, self.btn_folder):
            w.config(state=state)
        self.model_box.config(state=tk.DISABLED if busy else "readonly")
        self.btn_cancel.config(state=tk.NORMAL if busy else tk.DISABLED)

    def on_close(self) -> None:
        if self.stop_event is not None and not self.stop_event.is_set() and str(self.btn_cancel["state"]) == tk.NORMAL:
            if not messagebox.askyesno(APP_TITLE, "解析中です。中止して終了しますか？"):
                return
            self.stop_event.set()
        self.flush_save()
        self.destroy()


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
