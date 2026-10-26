"""解析の途中経過を結果フォルダに保存し、止まったところから再開できるようにする。

中止ボタン・アプリの終了・PCの電源断のどれで止まっても、最後に保存した位置から続けられる。

  analysis_points.csv    これまでに追跡した足元の座標（追記していく。毎回全部を書き直さないので長い動画でも速い）
  analysis_progress.json どこまで解析したか（何フレーム目までか、points のどこまでが有効か）

progress.json は「一時ファイルに書いてから置き換える」ので、書いている途中で電源が落ちても壊れない。
points.csv は progress.json に記録した長さまでを有効とし、それより後ろ（保存しきれなかった分）は捨てる。
解析が最後まで終わって結果を書き出したら、この2つは消す（session.write_result が消す）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROGRESS_NAME = "analysis_progress.json"
POINTS_NAME = "analysis_points.csv"
FORMAT = "foot-trajectory-overlay/analysis-progress"


@dataclass
class SavedProgress:
    frames_done: int  # 0〜frames_done-1 フレーム目までは解析して保存済み
    total_frames: int
    tracks: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)  # 出力ID → [(frame, x, y)]
    next_split_id: int = 0

    @property
    def percent(self) -> float:
        return self.frames_done * 100 / self.total_frames if self.total_frames else 0.0


def video_fingerprint(video: Path) -> dict:
    """同じ動画かどうかの目印（名前と大きさ。コピーしても変わらない）。"""
    try:
        size = video.stat().st_size
    except OSError:
        size = -1
    return {"name": video.name, "size": size}


class Checkpoint:
    def __init__(self, folder: Path, video: Path, settings: dict):
        self.folder = folder
        self.progress_path = folder / PROGRESS_NAME
        self.points_path = folder / POINTS_NAME
        self.key = {"video": video_fingerprint(video), "settings": settings}
        self._written: dict[int, int] = {}  # 出力ID → points.csv に書いた点の数
        self._bytes = 0

    # ---- 読み込み ----

    def load(self) -> SavedProgress | None:
        """同じ動画・同じ設定の途中経過があれば読む。なければ（または別物なら）None。"""
        state = read_state(self.folder)
        if state is None or state.get("key") != self.key:
            return None
        valid = int(state.get("points_bytes", 0))
        try:
            data = self.points_path.read_bytes()[:valid] if valid else b""
        except OSError:
            return None
        if len(data) < valid:
            return None  # 保存したはずの点が欠けている
        tracks: dict[int, list[tuple[int, float, float]]] = {}
        for row in data.decode("utf-8").splitlines():
            parts = row.split(",")
            if len(parts) != 4:
                continue
            tracks.setdefault(int(parts[0]), []).append((int(parts[1]), float(parts[2]), float(parts[3])))
        # 有効な長さより後ろは捨て、そこから追記を続ける
        with self.points_path.open("r+b") as f:
            f.truncate(valid)
        self._bytes = valid
        self._written = {tid: len(pts) for tid, pts in tracks.items()}
        return SavedProgress(
            frames_done=int(state["frames_done"]),
            total_frames=int(state.get("total_frames", 0)),
            tracks=tracks,
            next_split_id=int(state.get("next_split_id", 0)),
        )

    # ---- 書き込み ----

    def start_fresh(self) -> None:
        """前の途中経過を捨てて、最初から記録を始める。"""
        self.folder.mkdir(parents=True, exist_ok=True)
        discard(self.folder)
        self.points_path.write_bytes(b"")
        self._written = {}
        self._bytes = 0

    def commit(self, frames_done: int, total_frames: int, tracks: dict, next_split_id: int) -> None:
        """frames_done フレーム目の手前までの結果を保存する。前回から増えた点だけを追記する。"""
        rows = []
        for tid, pts in tracks.items():
            done = self._written.get(tid, 0)
            for p in pts[done:]:
                if p.frame < frames_done:
                    rows.append(f"{tid},{p.frame},{p.x:.2f},{p.y:.2f}\n")
                    done += 1
            self._written[tid] = done
        if rows:
            data = "".join(rows).encode("utf-8")
            with self.points_path.open("ab") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            self._bytes += len(data)
        state = {
            "format": FORMAT,
            "version": 1,
            "key": self.key,
            "frames_done": frames_done,
            "total_frames": total_frames,
            "points_bytes": self._bytes,
            "next_split_id": next_split_id,
        }
        tmp = self.progress_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.progress_path)


def read_state(folder: Path) -> dict | None:
    try:
        state = json.loads((folder / PROGRESS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or state.get("format") != FORMAT or "frames_done" not in state:
        return None
    return state


def progress_info(folder: Path, video: Path | None = None) -> dict | None:
    """画面に「途中まで解析済み」と出すための情報。video を渡すと、その動画の途中経過のときだけ返す。

    戻り値: {"frames_done", "total_frames", "percent"}
    """
    state = read_state(folder)
    if state is None:
        return None
    if video is not None and state.get("key", {}).get("video") != video_fingerprint(video):
        return None
    done, total = int(state["frames_done"]), int(state.get("total_frames", 0))
    return {"frames_done": done, "total_frames": total, "percent": done * 100 / total if total else 0.0}


def discard(folder: Path) -> None:
    for name in (PROGRESS_NAME, POINTS_NAME, PROGRESS_NAME.replace(".json", ".tmp")):
        (folder / name).unlink(missing_ok=True)
