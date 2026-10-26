"""動画から人物を検出・追跡し、足元座標の軌跡を集める。"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
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


SPLIT_ID_BASE = 100000  # 画面外から戻ってきた人に解析中だけ振る仮IDの始まり（最後に通し番号へ振り直す）


@dataclass
class TrackingResult:
    first_frame: np.ndarray
    fps: float
    total_frames: int
    tracks: dict[int, list[TrackPoint]] = field(default_factory=dict)
    # 追跡器のID → いま記録中の線のID（画面外に出て戻ってきたら新しい線にする）
    _current_id: dict[int, int] = field(default_factory=dict)
    _left_frame: set[int] = field(default_factory=set)
    _next_split_id: int = SPLIT_ID_BASE
    _id_offset: int = 0  # 再開したとき、新しい追跡器のIDに足す数
    resumed_from: int = 0  # 途中から再開したときの再開フレーム（最初からなら 0）


def edge_margin(width: int, height: int) -> float:
    """足元が画面の縁からこの距離以内なら「画面外」とみなす。"""
    return max(2.0, 0.005 * max(width, height))


def foot_outside_frame(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> bool:
    """検出枠が画面の下端・左右の端で切れていて、足元が画面外にあるか。

    枠が縁で切れていると「枠の下辺の中央」は本当の足元ではなく画面の縁になるため、
    そのまま記録すると追跡線が画面の辺に沿って描かれてしまう。
    """
    m = edge_margin(width, height)
    return y2 >= height - 1 - m or x1 <= m or x2 >= width - 1 - m


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


RESUME_OVERLAP_SEC = 2.0  # 再開するとき、止まった位置のこれだけ手前から追跡し直して人物をつなぐ
CHECKPOINT_SEC = 30.0  # 途中経過を保存する間隔（秒）。電源が落ちても失うのはこれ以内


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
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    checkpoint_sec: float = CHECKPOINT_SEC,
) -> TrackingResult:
    """動画の全フレームを解析し、追跡IDごとの足元座標リストを返す。

    フレームの間引きはしない（追跡IDの取り違えや、線をまたぐ瞬間の見逃しを減らすため）。
    checkpoint_dir を渡すと、そこに途中経過を保存し（resume.py）、前回止まった位置から再開する。
    """
    # ultralytics の読み込みは重いので、必要になってから import する
    from ultralytics import YOLO

    from .resume import Checkpoint
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
        h, w = first.shape[:2]
        result = TrackingResult(first_frame=first.copy(), fps=fps, total_frames=total)

        ckpt = None
        saved = None
        if checkpoint_dir is not None:
            settings = {"model": Path(model_name).name, "conf": conf, "imgsz": imgsz, "tracker": tracker,
                        "max_frames": max_frames}
            ckpt = Checkpoint(checkpoint_dir, video_path, settings)
            saved = ckpt.load() if resume else None
            if saved is None:
                ckpt.start_fresh()

        resume_from = 0  # このフレームから記録する（それより前は保存済み）
        frame_idx = 0
        frame = first
        if saved is not None and saved.frames_done > 0:
            result.tracks = {
                tid: [TrackPoint(f, x, y) for f, x, y in pts] for tid, pts in saved.tracks.items()
            }
            result._next_split_id = max(SPLIT_ID_BASE, saved.next_split_id)
            # 新しい追跡器のIDは 1 から振り直されるので、保存済みのIDと重ならないようずらす
            result._id_offset = max((t for t in result.tracks if t < SPLIT_ID_BASE), default=0)
            result.resumed_from = resume_from = saved.frames_done
            frame_idx = max(0, resume_from - int(round(RESUME_OVERLAP_SEC * fps)))
            # 再開位置の手前まで読み飛ばす（デコードだけなので解析よりずっと速い）
            for _ in range(frame_idx - 1):
                if not cap.grab():
                    break
            if frame_idx > 0:
                ok, frame = cap.read()
                if not ok:
                    frame = None

        overlap: dict[int, list] = {}  # 追跡し直した区間の検出結果（保存済みの線とつなぐのに使う）
        reported = -1
        last_save = time.monotonic()
        try:
            while frame is not None:
                if stop_event is not None and stop_event.is_set():
                    raise TrackingCancelled()
                obs = _detect(model, frame, conf, imgsz, tracker, device)
                if frame_idx < resume_from:
                    overlap[frame_idx] = obs
                else:
                    if overlap:
                        _stitch(result, overlap, w, h)
                        overlap = {}
                    _record(result, frame_idx, obs, w, h)
                frame_idx += 1

                # 通知は 1% 刻み程度に抑える（画面更新やコンソール出力が処理を遅くしないように）
                step = max(1, total // 100) if total else 30
                if progress is not None and (frame_idx // step != reported):
                    reported = frame_idx // step
                    progress(frame_idx, max(total, frame_idx), len(result.tracks))

                if ckpt is not None and frame_idx > resume_from and time.monotonic() - last_save >= checkpoint_sec:
                    ckpt.commit(frame_idx, total, result.tracks, result._next_split_id)
                    last_save = time.monotonic()

                if max_frames is not None and frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
        except BaseException:
            # 中止・Ctrl+C・読み込みエラーのどれでも、ここまでの結果を残して次回続きから再開できるようにする
            if ckpt is not None and frame_idx > resume_from:
                try:
                    ckpt.commit(frame_idx, total, result.tracks, result._next_split_id)
                except OSError:
                    pass
            raise

        result.total_frames = max(frame_idx, resume_from)
        result.tracks = _renumber_split_ids(result.tracks)
        if progress is not None:
            progress(result.total_frames, result.total_frames, len(result.tracks))
        return result
    finally:
        cap.release()
        if tmp_path is not None:
            shutil.rmtree(tmp_path.parent, ignore_errors=True)


Observation = tuple[int, float, float, bool]
"""1人分の検出: (追跡器のID, 足元x, 足元y, 足元が画面外か)"""


def _detect(model, frame, conf, imgsz, tracker, device) -> list[Observation]:
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
        return []
    h, w = frame.shape[:2]
    out = []
    for track_id, (x1, y1, x2, y2) in zip(boxes.id.int().tolist(), boxes.xyxy.tolist()):
        fx, fy = foot_point(x1, y1, x2, y2)
        out.append((track_id, fx, fy, foot_outside_frame(x1, y1, x2, y2, w, h)))
    return out


def _record(result: TrackingResult, frame_idx: int, obs: list[Observation], w: int, h: int) -> None:
    for track_id, fx, fy, outside in obs:
        if outside:
            # 足元が画面外に出たら、この人の追跡線はここで終わり
            if track_id in result._current_id:
                result._left_frame.add(track_id)
            continue
        if track_id in result._left_frame:
            # 画面外から戻ってきた → 別の線として記録する
            result._left_frame.discard(track_id)
            result._current_id[track_id] = result._next_split_id
            result._next_split_id += 1
        out_id = result._current_id.setdefault(track_id, track_id + result._id_offset)
        result.tracks.setdefault(out_id, []).append(TrackPoint(frame_idx, fx, fy))


def _stitch(result: TrackingResult, overlap: dict[int, list[Observation]], w: int, h: int) -> None:
    """再開時に追跡し直した区間で、新しい追跡器のIDと保存済みの線を足元の位置で対応づける。

    止まった瞬間に歩いていた人が、再開後に別人（2本の線）にならないようにするため。
    """
    near = 0.02 * max(w, h)
    old_at: dict[int, list[tuple[int, float, float]]] = {}
    for tid, pts in result.tracks.items():
        for p in pts:
            if p.frame in overlap:
                old_at.setdefault(p.frame, []).append((tid, p.x, p.y))
    votes: dict[tuple[int, int], int] = {}
    seen: dict[int, int] = {}
    last_outside: dict[int, bool] = {}
    for f in sorted(overlap):
        for tid, fx, fy, outside in overlap[f]:
            last_outside[tid] = outside
            if outside:
                continue
            seen[tid] = seen.get(tid, 0) + 1
            best, best_d = None, near
            for oid, ox, oy in old_at.get(f, []):
                d = ((ox - fx) ** 2 + (oy - fy) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = oid, d
            if best is not None:
                votes[(tid, best)] = votes.get((tid, best), 0) + 1
    used_new: set[int] = set()
    used_old: set[int] = set()
    for (tid, oid), n in sorted(votes.items(), key=lambda kv: -kv[1]):
        if tid in used_new or oid in used_old or n < max(2, seen[tid] // 2):
            continue
        used_new.add(tid)
        used_old.add(oid)
        result._current_id[tid] = oid
        if last_outside.get(tid):
            result._left_frame.add(tid)


def _renumber_split_ids(tracks: dict[int, list[TrackPoint]]) -> dict[int, list[TrackPoint]]:
    """画面外から戻ってきた人の仮ID（100000〜）を、通常のIDの続きの番号に振り直す。"""
    normal = [t for t in tracks if t < SPLIT_ID_BASE]
    next_id = max(normal, default=0) + 1
    out = {t: tracks[t] for t in normal}
    for t in sorted(t for t in tracks if t >= SPLIT_ID_BASE):
        out[next_id] = tracks[t]
        next_id += 1
    return out


def split_at_frame_edges(tracks: dict[int, list[TrackPoint]], width: int, height: int) -> dict[int, list[TrackPoint]]:
    """保存済みの足元座標から、画面の縁に張り付いた点を除き、そこで線を分ける。

    この修正より前に解析した結果を開いたときにも、画面外の足元が描かれないようにするため。
    """
    m = edge_margin(width, height)
    out: dict[int, list[TrackPoint]] = {}
    next_id = max(tracks, default=0) + 1
    for tid in sorted(tracks):
        current: list[TrackPoint] = []
        segments: list[list[TrackPoint]] = []
        for p in tracks[tid]:
            if p.y >= height - 1 - m or p.x <= m or p.x >= width - 1 - m:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append(p)
        if current:
            segments.append(current)
        for k, seg in enumerate(segments):
            if k == 0:
                out[tid] = seg
            else:
                out[next_id] = seg
                next_id += 1
    return out
