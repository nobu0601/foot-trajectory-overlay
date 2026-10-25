"""画像に日本語の文字を描く（OpenCV の文字描画は日本語を描けないため、Pillow を使う）。"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Windows に標準で入っている日本語フォント（上から順に探す）。Linux・Mac 用も少し入れておく
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    windir = os.environ.get("WINDIR")
    candidates = list(FONT_CANDIDATES)
    if windir and sys.platform.startswith("win"):
        candidates = [str(Path(windir) / "Fonts" / Path(c).name) for c in candidates[:3]] + candidates
    return next((c for c in candidates if Path(c).is_file()), None)


@lru_cache(maxsize=16)
def _font(size: int):
    path = _font_path()
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def text_size(text: str, size: int) -> tuple[int, int]:
    font = _font(size)
    if font is None:
        return len(text) * size // 2, size
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def put_texts(img: np.ndarray, items: list[tuple[str, tuple[float, float], int, tuple[int, int, int]]], outline: bool = True) -> None:
    """img（BGR）に文字をまとめて描く。items = [(文字, 左上の位置, 大きさpx, 色BGR), ...]

    日本語フォントが見つからない場合は、英数字以外を描けないので ? になる（OpenCV で代わりに描く）。
    """
    if not items:
        return
    font_ok = _font(12) is not None
    if not font_ok:
        import cv2

        for text, (x, y), size, color in items:
            safe = text.encode("ascii", "replace").decode()
            cv2.putText(img, safe, (int(x), int(y + size)), cv2.FONT_HERSHEY_SIMPLEX, size / 28, color, 1, cv2.LINE_AA)
        return
    # 文字の周りだけを切り出して描くと速いが、件数が少ないので画像全体を一度だけ変換する
    pil = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    for text, (x, y), size, color in items:
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        stroke = max(1, size // 8) if outline else 0
        draw.text((x, y), text, font=_font(size), fill=rgb, stroke_width=stroke, stroke_fill=(0, 0, 0))
    img[:, :, :] = np.asarray(pil)[:, :, ::-1]
