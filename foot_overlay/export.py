"""結果ファイルの書き出し。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .tracking import TrackPoint


def save_image(path: Path, img: np.ndarray) -> None:
    """日本語を含むパスでも保存できるよう、imencode 経由で書き出す。

    Windows の cv2.imwrite は日本語パスに書き込めないため。
    """
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"画像のエンコードに失敗しました: {path}")
    path.write_bytes(buf.tobytes())


def save_tracks_csv(path: Path, tracks: dict[int, list[TrackPoint]], fps: float) -> None:
    """足元座標の生データを CSV に保存する（後でカウント処理に使う想定）。"""
    # utf-8-sig にすると Excel でそのまま開いても文字化けしない
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "frame", "time_sec", "foot_x", "foot_y"])
        for tid in sorted(tracks):
            for p in tracks[tid]:
                writer.writerow([tid, p.frame, f"{p.frame / fps:.3f}", f"{p.x:.1f}", f"{p.y:.1f}"])
