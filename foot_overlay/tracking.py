"""動画から人物を検出・追跡し、足元座標の軌跡を集める。"""

from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PERSON_CLASS_ID = 0  # COCO の "person"


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float


@dataclass
class TrackingResult:
    first_frame: np.ndarray
    fps: float
    total_frames: int
    tracks: dict[int, list[TrackPoint]] = field(default_factory=dict)


def open_video(path: Path) -> tuple[cv2.VideoCapture, Path | None]:
    """動画を開く。

    Windows の OpenCV は日本語を含むパスを開けないことがあるため、
    開けなかった場合は英数字だけの一時パスにコピーして開き直す。
    戻り値の2つ目は後で削除すべき一時ファイル（なければ None）。
    """
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None
    cap.release()

    tmp_dir = Path(tempfile.mkdtemp(prefix="foot_overlay_"))
    tmp_path = tmp_dir / ("input" + path.suffix.lower())
    shutil.copyfile(path, tmp_path)
    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"動画を開けませんでした: {path}")
    return cap, tmp_path


def foot_point(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """バウンディングボックスの下辺中央を足元とみなす。"""
    return (x1 + x2) / 2.0, y2


ProgressCallback = Callable[[int, int, int], None]
"""進み具合の通知: (処理済みフレーム数, 総フレーム数, これまでの追跡ID数)"""


class TrackingCancelled(Exception):
    """利用者が解析を中止した。"""


def print_progress(done: int, total: int, ids: int) -> None:
    percent = int(done * 100 / total) if total else 0
    end = "\n" if total and done >= total else ""
    print(f"\r  追跡中... {percent:3d}%  (人数ID: {ids})", end=end, flush=True)


def track_video(
    video_path: Path,
    model_name: str = "yolo11n.pt",
    conf: float = 0.3,
    imgsz: int = 640,
    tracker: str = "bytetrack.yaml",
    device: str = "cpu",
    max_frames: int | None = None,
    progress: ProgressCallback | None = print_progress,
    stop_event: threading.Event | None = None,
) -> TrackingResult:
    """動画の全フレームを解析し、追跡IDごとの足元座標リストを返す。

    フレームの間引きはしない（追跡IDの取り違えや、線をまたぐ瞬間の見逃しを減らすため）。
    """
    # ultralytics の読み込みは重いので、必要になってから import する
    from ultralytics import YOLO

    from .runtime import disable_ultralytics_sync, resolve_model

    disable_ultralytics_sync()
    model = YOLO(resolve_model(model_name))
    cap, tmp_path = open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames is not None:
            total = min(total, max_frames) if total else max_frames

        ok, first = cap.read()
        if not ok:
            raise RuntimeError("最初のフレームを読み込めませんでした")

        result = TrackingResult(first_frame=first.copy(), fps=fps, total_frames=total)
        frame_idx = 0
        frame = first
        reported = -1

        while True:
            if stop_event is not None and stop_event.is_set():
                raise TrackingCancelled()
            _track_one_frame(model, frame, frame_idx, result, conf, imgsz, tracker, device)
            frame_idx += 1

            # 通知は 1% 刻み程度に抑える（画面更新やコンソール出力が処理を遅くしないように）
            step = max(1, total // 100) if total else 30
            if progress is not None and (frame_idx // step != reported):
                reported = frame_idx // step
                progress(frame_idx, max(total, frame_idx), len(result.tracks))

            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

        result.total_frames = frame_idx
        if progress is not None:
            progress(frame_idx, frame_idx, len(result.tracks))
        return result
    finally:
        cap.release()
        if tmp_path is not None:
            shutil.rmtree(tmp_path.parent, ignore_errors=True)


def _track_one_frame(model, frame, frame_idx, result, conf, imgsz, tracker, device) -> None:
    outputs = model.track(
        frame,
        persist=True,  # フレーム間でIDを引き継ぐ
        classes=[PERSON_CLASS_ID],
        conf=conf,
        imgsz=imgsz,
        tracker=tracker,
        device=device,
        verbose=False,
    )
    boxes = outputs[0].boxes
    if boxes is None or boxes.id is None:
        return
    ids = boxes.id.int().tolist()
    for track_id, (x1, y1, x2, y2) in zip(ids, boxes.xyxy.tolist()):
        fx, fy = foot_point(x1, y1, x2, y2)
        result.tracks.setdefault(track_id, []).append(TrackPoint(frame_idx, fx, fy))
