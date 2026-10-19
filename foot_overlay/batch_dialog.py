"""一括処理の画面: たくさんの動画をまとめて解析し、重畳画像を1つのフォルダに出力する。"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .batch import BatchItem, BatchResult, collect_items, run_batch

VIDEO_TYPES = [("動画ファイル", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV"), ("すべてのファイル", "*.*")]


def _fmt_sec(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}秒"
    if sec < 3600:
        return f"{sec // 60}分{sec % 60:02d}秒"
    return f"{sec // 3600}時間{sec % 3600 // 60:02d}分"


class BatchDialog(tk.Toplevel):
    def __init__(self, master, models: dict[str, str], default_model: str, line_width: int, open_folder, open_result):
        super().__init__(master)
        self.title("一括処理 - 重畳画像をまとめて作成")
        self.geometry("820x600")
        self.minsize(700, 480)
        self.transient(master)
        self.models = models
        self.open_folder = open_folder
        self.open_result = open_result  # 結果フォルダをメイン画面で開く関数

        self.inputs: list[Path] = []
        self.excluded: set[Path] = set()
        self.items: list[BatchItem] = []
        self.status: dict[Path, str] = {}
        self.people: dict[Path, int] = {}
        self.messages: queue.Queue = queue.Queue()
        self.stop_event: threading.Event | None = None
        self.running = False

        self._build(default_model, line_width)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.grab_set()  # 一括処理中にメイン画面で別の解析を始めないようにする

    # ---------- 画面 ----------

    def _build(self, default_model: str, line_width: int) -> None:
        pad = {"padx": 10, "pady": 4}
        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)
        ttk.Label(top, text="処理する動画:").pack(side=tk.LEFT)
        self.btn_add_folder = ttk.Button(top, text="フォルダを追加…", command=self.add_folder)
        self.btn_add_folder.pack(side=tk.LEFT, padx=4)
        self.btn_add_files = ttk.Button(top, text="ファイルを追加…", command=self.add_files)
        self.btn_add_files.pack(side=tk.LEFT)
        self.btn_remove = ttk.Button(top, text="選んだ動画を外す", command=self.remove_selected)
        self.btn_remove.pack(side=tk.LEFT, padx=4)
        self.btn_clear = ttk.Button(top, text="全部外す", command=self.clear_all)
        self.btn_clear.pack(side=tk.LEFT)
        self.count_var = tk.StringVar(value="0本")
        ttk.Label(top, textvariable=self.count_var).pack(side=tk.RIGHT)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill=tk.BOTH, expand=True, **pad)
        cols = ("name", "status", "people", "path")
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        for col, text, width, anchor in zip(
            cols,
            ("出力名", "状態", "人数", "動画の場所"),
            (180, 130, 60, 400),
            (tk.W, tk.CENTER, tk.CENTER, tk.W),
        ):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor=anchor)
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.bind("<Double-Button-1>", self.on_double_click)

        out = ttk.Frame(self)
        out.pack(fill=tk.X, **pad)
        ttk.Label(out, text="出力フォルダ:").pack(side=tk.LEFT)
        self.out_var = tk.StringVar()
        ttk.Entry(out, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.btn_out = ttk.Button(out, text="参照…", command=self.choose_out)
        self.btn_out.pack(side=tk.LEFT)

        opts = ttk.Frame(self)
        opts.pack(fill=tk.X, **pad)
        ttk.Label(opts, text="検出モデル:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=default_model)
        self.model_box = ttk.Combobox(opts, textvariable=self.model_var, values=list(self.models), state="readonly", width=18)
        self.model_box.pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(opts, text="追跡線の太さ:").pack(side=tk.LEFT)
        self.width_var = tk.IntVar(value=line_width)
        ttk.Spinbox(opts, from_=1, to=15, textvariable=self.width_var, width=4).pack(side=tk.LEFT, padx=(4, 16))
        self.skip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="解析済みの動画は飛ばす（途中から再開できる）", variable=self.skip_var).pack(side=tk.LEFT)

        run = ttk.Frame(self)
        run.pack(fill=tk.X, **pad)
        self.btn_start = ttk.Button(run, text="一括処理を開始", command=self.start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(run, text="中止", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_open_out = ttk.Button(run, text="出力フォルダを開く", command=self.open_out)
        self.btn_open_out.pack(side=tk.RIGHT)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_var = tk.StringVar(value="フォルダかファイルを追加して「一括処理を開始」を押してください。")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.TOP, anchor=tk.W)
        self.progress = ttk.Progressbar(bottom, maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 0))

    # ---------- 動画の追加・削除 ----------

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="動画の入ったフォルダを選択（サブフォルダも対象）")
        if folder:
            self._add_inputs([Path(folder)])
            if not self.out_var.get():
                self.out_var.set(str(Path(folder) / "重畳画像"))

    def add_files(self) -> None:
        files = filedialog.askopenfilenames(parent=self, title="動画を選択（複数可）", filetypes=VIDEO_TYPES)
        if files:
            self._add_inputs([Path(f) for f in files])
            if not self.out_var.get():
                self.out_var.set(str(Path(files[0]).parent / "重畳画像"))

    def _add_inputs(self, paths: list[Path]) -> None:
        for p in paths:
            if p not in self.inputs:
                self.inputs.append(p)
            # 外していた動画も、あらためて追加されたら対象に戻す
            self.excluded.difference_update([v for v in self.excluded if v == p or p in v.parents])
        self.refresh_items()

    def remove_selected(self) -> None:
        for iid in self.table.selection():
            self.excluded.add(self.items[int(iid)].video)
        self.refresh_items()

    def clear_all(self) -> None:
        self.inputs.clear()
        self.excluded.clear()
        self.refresh_items()

    def refresh_items(self) -> None:
        out = self._out_dir()
        self.items = [i for i in collect_items(self.inputs) if i.video not in self.excluded and not _inside(i.video, out)]
        self.table.delete(*self.table.get_children())
        for k, item in enumerate(self.items):
            status = self.status.get(item.video, "")
            people = self.people.get(item.video, "")
            self.table.insert("", tk.END, iid=str(k), values=(item.name, status, people, str(item.video)))
        self.count_var.set(f"{len(self.items)}本")

    def choose_out(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="重畳画像の出力フォルダを選択")
        if folder:
            self.out_var.set(folder)
            self.refresh_items()

    def _out_dir(self) -> Path | None:
        text = self.out_var.get().strip()
        return Path(text) if text else None

    def open_out(self) -> None:
        out = self._out_dir()
        if out is not None and out.is_dir():
            self.open_folder(out)
        else:
            messagebox.showinfo("一括処理", "出力フォルダはまだありません。", parent=self)

    def on_double_click(self, _event=None) -> None:
        """処理済みの動画をダブルクリックすると、メイン画面で結果を開く（カウント線を引ける）。"""
        sel = self.table.selection()
        out = self._out_dir()
        if not sel or out is None or self.running:
            return
        item = self.items[int(sel[0])]
        result_dir = out / "results" / f"{item.name}_overlay"
        if (result_dir / "tracks.csv").is_file():
            self.grab_release()
            self.open_result(result_dir)
            self.destroy()

    # ---------- 実行 ----------

    def start(self) -> None:
        if not self.items:
            messagebox.showinfo("一括処理", "処理する動画を追加してください。", parent=self)
            return
        out = self._out_dir()
        if out is None:
            messagebox.showinfo("一括処理", "出力フォルダを指定してください。", parent=self)
            return
        try:
            width = min(max(int(self.width_var.get()), 1), 15)
        except (tk.TclError, ValueError):
            width = 1
        items = list(self.items)
        model = self.models[self.model_var.get()]
        skip = self.skip_var.get()
        self.stop_event = threading.Event()
        stop = self.stop_event
        for item in items:
            self.status[item.video] = "待ち"
        self.refresh_items()
        self._set_running(True)
        started = time.time()

        def progress(i, n, item, done, total):
            self.messages.put(("progress", i, n, item, done, total, started))

        def on_result(res: BatchResult):
            self.messages.put(("result", res))

        def worker():
            try:
                results = run_batch(
                    items,
                    out,
                    model_name=model,
                    line_width=width,
                    skip_done=skip,
                    progress=progress,
                    on_result=on_result,
                    stop_event=stop,
                )
                self.messages.put(("done", results, time.time() - started))
            except Exception as e:  # 出力フォルダが作れない等
                self.messages.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self.poll)

    def poll(self) -> None:
        finished = False
        try:
            while True:
                msg = self.messages.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, n, item, done, total, started = msg
                    frac = (i + (done / total if total else 0)) / n
                    self.progress["value"] = frac * 100
                    elapsed = time.time() - started
                    remain = elapsed / frac * (1 - frac) if frac > 0.002 else 0
                    pct = done * 100 // total if total else 0
                    self.status_var.set(
                        f"[{i + 1}/{n}] {item.name}: 解析中 {pct}%　全体 {frac:.0%}・残り約{_fmt_sec(remain)}"
                    )
                    if self.status.get(item.video) != "解析中":
                        self.status[item.video] = "解析中"
                        self._update_row(item)
                elif kind == "result":
                    res: BatchResult = msg[1]
                    label = res.status if res.status != "失敗" else f"失敗: {res.message[:40]}"
                    self.status[res.item.video] = label
                    self.people[res.item.video] = res.people
                    self._update_row(res.item)
                elif kind == "done":
                    finished = True
                    _, results, elapsed = msg
                    ok = sum(r.status == "完了" for r in results)
                    skipped = sum(r.status.startswith("スキップ") for r in results)
                    failed = sum(r.status == "失敗" for r in results)
                    stopped = sum(r.status == "中止" for r in results)
                    text = f"完了 {ok}本・スキップ {skipped}本・失敗 {failed}本" + (f"・中止 {stopped}本" if stopped else "")
                    self.status_var.set(f"{text}（{_fmt_sec(elapsed)}）。一覧は batch_summary.csv にあります。")
                    self.progress["value"] = 100
                    messagebox.showinfo(
                        "一括処理",
                        f"{text}\n\n重畳画像は出力フォルダに「動画名.png」で保存しました。\n"
                        "一覧の行をダブルクリックすると、その動画の結果を開いてカウント線を引けます。",
                        parent=self,
                    )
                elif kind == "error":
                    finished = True
                    self.status_var.set("一括処理に失敗しました。")
                    messagebox.showerror("一括処理", msg[1], parent=self)
        except queue.Empty:
            pass
        if finished:
            self._set_running(False)
        else:
            self.after(150, self.poll)

    def _update_row(self, item: BatchItem) -> None:
        for k, it in enumerate(self.items):
            if it.video == item.video:
                self.table.set(str(k), "status", self.status.get(item.video, ""))
                self.table.set(str(k), "people", self.people.get(item.video, ""))
                self.table.see(str(k))
                break

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("中止しています…（いま解析中の動画を止めます）")

    def _set_running(self, running: bool) -> None:
        self.running = running
        state = tk.DISABLED if running else tk.NORMAL
        for w in (self.btn_add_folder, self.btn_add_files, self.btn_remove, self.btn_clear, self.btn_out, self.btn_start):
            w.config(state=state)
        self.model_box.config(state=tk.DISABLED if running else "readonly")
        self.btn_stop.config(state=tk.NORMAL if running else tk.DISABLED)

    def on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("一括処理", "処理中です。中止して閉じますか？", parent=self):
                return
            self.stop()
        self.grab_release()
        self.destroy()


def _inside(path: Path, folder: Path | None) -> bool:
    """出力フォルダの中の動画は処理対象にしない（出力先を入力フォルダの中に作った場合のため）。"""
    if folder is None:
        return False
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False
