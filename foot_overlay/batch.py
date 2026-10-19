"""大量の動画をまとめて解析し、重畳画像を1つのフォルダに出力する。

  python -m foot_overlay.batch 動画やフォルダ... -o 出力フォルダ

出力フォルダの中身:
  動画名.png             足元軌跡の重畳画像（動画ごと）
  results/動画名_overlay  解析結果（アプリの「解析結果を開く」で開くとカウント線を引ける）
  batch_summary.csv      動画ごとの処理結果の一覧
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .drawing import filter_short_tracks
from .session import Session, write_result
from .tracking import TrackingCancelled, track_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}
MIN_POINTS = 10
SMOOTH = 5


@dataclass
class BatchItem:
    video: Path
    name: str  # 出力ファイル名（拡張子なし）。フォルダ指定時はサブフォルダ名も含めて重複を避ける


@dataclass
class BatchResult:
    item: BatchItem
    status: str  # "完了" / "スキップ（解析済み）" / "失敗" / "中止"
    people: int = 0
    frames: int = 0
    seconds: float = 0.0
    message: str = ""


def find_videos(folder: Path) -> list[Path]:
    """フォルダ内（サブフォルダも含む）の動画を名前順に返す。"""
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def collect_items(inputs: list[Path]) -> list[BatchItem]:
    """入力（動画ファイル・フォルダ）から処理する動画の一覧を作る。出力名が重ならないようにする。"""
    items: list[BatchItem] = []
    used: set[str] = set()
    seen: set[Path] = set()
    for path in inputs:
        if path.is_dir():
            pairs = [(v, "_".join(v.relative_to(path).with_suffix("").parts)) for v in find_videos(path)]
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            pairs = [(path, path.stem)]
        else:
            continue
        for video, name in pairs:
            key = video.resolve()
            if key in seen:
                continue
            seen.add(key)
            base, n = name, 2
            while name.lower() in used:
                name = f"{base}({n})"
                n += 1
            used.add(name.lower())
            items.append(BatchItem(video, name))
    return items


ItemProgress = Callable[[int, int, BatchItem, int, int], None]
"""(何本目, 全本数, 動画, 処理済みフレーム, 総フレーム)"""


def run_batch(
    items: list[BatchItem],
    out_dir: Path,
    model_name: str = "yolo11n.pt",
    line_width: int | None = None,
    skip_done: bool = True,
    progress: ItemProgress | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    stop_event: threading.Event | None = None,
) -> list[BatchResult]:
    """動画を1本ずつ解析し、重畳画像を out_dir に保存する。1本失敗しても残りは続ける。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results: list[BatchResult] = []

    for index, item in enumerate(items):
        if stop_event is not None and stop_event.is_set():
            results.append(BatchResult(item, "中止"))
            continue
        result_dir = results_dir / f"{item.name}_overlay"
        image_path = out_dir / f"{item.name}.png"
        started = time.time()
        try:
            if skip_done and (result_dir / "tracks.csv").is_file() and image_path.is_file():
                session = Session(result_dir)
                res = BatchResult(item, "スキップ（解析済み）", people=len(session.tracks))
            else:

                def frame_progress(done: int, total: int, ids: int, index=index, item=item) -> None:
                    if progress is not None:
                        progress(index, len(items), item, done, total)

                tracked = track_video(
                    item.video, model_name=model_name, progress=frame_progress, stop_event=stop_event
                )
                tracks = filter_short_tracks(tracked.tracks, MIN_POINTS)
                view = {"line_width": line_width} if line_width else None
                write_result(result_dir, item.video, tracked, tracks, smooth=SMOOTH, min_points=MIN_POINTS, view=view)
                session = Session(result_dir)
                session.save()
                shutil.copyfile(result_dir / "overlay.png", image_path)
                res = BatchResult(
                    item,
                    "完了",
                    people=len(session.tracks),
                    frames=tracked.total_frames,
                    seconds=time.time() - started,
                )
        except TrackingCancelled:
            res = BatchResult(item, "中止", seconds=time.time() - started)
        except Exception as e:  # 壊れた動画などがあっても残りは処理する
            res = BatchResult(item, "失敗", seconds=time.time() - started, message=f"{type(e).__name__}: {e}")
        results.append(res)
        if on_result is not None:
            on_result(res)
        write_summary(out_dir / "batch_summary.csv", results)
    return results


def write_summary(path: Path, results: list[BatchResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["動画", "出力画像", "状態", "人数(追跡線)", "フレーム数", "処理秒数", "メッセージ"])
        for r in results:
            image = f"{r.item.name}.png" if r.status in ("完了", "スキップ（解析済み）") else ""
            w.writerow([str(r.item.video), image, r.status, r.people, r.frames, f"{r.seconds:.0f}", r.message])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="foot_overlay.batch", description="動画をまとめて解析し、重畳画像を1つのフォルダに出力します。")
    p.add_argument("inputs", nargs="+", type=Path, help="動画ファイル、または動画の入ったフォルダ（サブフォルダも対象）")
    p.add_argument("-o", "--out-dir", type=Path, required=True, help="出力フォルダ")
    p.add_argument("--model", default="yolo11n.pt", help="YOLOモデル（既定: yolo11n.pt）")
    p.add_argument("--line-width", type=int, help="追跡線の太さpx（省略時は自動）")
    p.add_argument("--redo", action="store_true", help="解析済みの動画も解析し直す")
    args = p.parse_args(argv)

    items = collect_items(args.inputs)
    if not items:
        print("処理する動画が見つかりませんでした。", file=sys.stderr)
        return 1
    print(f"{len(items)}本の動画を処理します → {args.out_dir}")

    def progress(i, n, item, done, total):
        percent = done * 100 // total if total else 0
        print(f"\r  [{i + 1}/{n}] {item.name}: {percent:3d}%", end="", flush=True)

    def on_result(r: BatchResult):
        print(f"\r  [{r.status}] {r.item.name}  人数 {r.people}  {r.seconds:.0f}秒 {r.message}".ljust(60))

    results = run_batch(
        items,
        args.out_dir,
        model_name=args.model,
        line_width=args.line_width,
        skip_done=not args.redo,
        progress=progress,
        on_result=on_result,
    )
    failed = sum(r.status == "失敗" for r in results)
    print(f"完了: {len(results) - failed}本 / 失敗: {failed}本  一覧: {args.out_dir / 'batch_summary.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
