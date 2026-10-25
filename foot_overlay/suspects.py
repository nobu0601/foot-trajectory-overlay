"""追跡線の様子から、誤判定らしい線（人ではない・別人と混ざった等）を見つける。

見つけた線はカウントから消すのではなく、
  - 画面では灰色の点線で表示する
  - カウント線の候補は、これらの線が通らない位置に引く
  - 手で引いたカウント線を通った場合は、信頼度の理由に表示する
ために使う。利用者が「誤判定ではない」と戻した線は対象外にする。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

REASONS = {
    "jump": "瞬間移動する",
    "zigzag": "じぐざぐで前に進まない",
    "offpath": "人が歩かない場所を通る",
}

JUMP_FACTOR = 6.0  # 1フレームの移動が、歩く人の普通の移動のこの倍以上なら瞬間移動
JUMP_MIN_RATIO = 0.03  # ただし画像の長辺のこの割合以上動いたときだけ（止まっている人の揺れと区別する）
ZIGZAG_STRAIGHTNESS = 0.3  # 1秒ごとの「進んだ距離 ÷ 歩いた道のり」の中央値がこれ未満ならじぐざぐ
ZIGZAG_MIN_STEP_RATIO = 0.01  # 1秒の道のりが画像の長辺のこの割合以上ある区間だけを見る（立ち止まりと区別する）
OFFPATH_CELL_RATIO = 0.05  # 「人が歩く場所」を調べるマスの大きさ
OFFPATH_SHARE = 0.7  # 線の点のうち、この割合以上が「他に誰も通らないマス」にあれば対象
OFFPATH_MIN_TRACKS = 10  # 追跡線がこれより少ない動画では判定しない（比べる相手が少ない）


@dataclass
class SuspectSettings:
    """誤判定の判定基準（「誤判定の疑い」の画面で変えられる。meta.json に保存）。"""

    jump_on: bool = True
    jump_factor: float = JUMP_FACTOR  # 普通の動きの何倍以上で瞬間移動とみなすか
    zigzag_on: bool = True
    zigzag_straightness: float = ZIGZAG_STRAIGHTNESS  # 1秒ごとの前進率がこれ未満ならじぐざぐ
    offpath_on: bool = True
    offpath_share: float = OFFPATH_SHARE  # 線の何割以上が誰も通らない場所ならば対象か
    offpath_min_tracks: int = OFFPATH_MIN_TRACKS  # 追跡線が何本以上の動画で判定するか

    @classmethod
    def from_dict(cls, d: dict | None) -> "SuspectSettings":
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base

    def to_dict(self) -> dict:
        return asdict(self)


def find_suspects(
    tracks_xy: dict[int, np.ndarray],
    width: int,
    height: int,
    fps: float = 30.0,
    settings: SuspectSettings | None = None,
) -> dict[int, list[str]]:
    """誤判定らしい追跡ID → 理由のキー（REASONS のキー）の一覧。"""
    cfg = settings or SuspectSettings()
    size = max(width, height)
    out: dict[int, list[str]] = {}

    # 歩く人の普通の1フレームの移動量（全員の移動量の中央値）
    steps = [np.hypot(*np.diff(xy, axis=0).T) for xy in tracks_xy.values() if len(xy) >= 2]
    moving = np.concatenate(steps) if steps else np.zeros(0)
    moving = moving[moving > 0.2]
    typical = float(np.median(moving)) if len(moving) else 1.0
    jump_limit = max(cfg.jump_factor * typical, JUMP_MIN_RATIO * size)

    for tid, xy in tracks_xy.items():
        if len(xy) < 2:
            continue
        step = np.hypot(*np.diff(xy, axis=0).T)
        if cfg.jump_on and step.max() >= jump_limit:
            out.setdefault(tid, []).append("jump")
        # じぐざぐ: 線全体ではなく1秒ごとに見る（行って戻った人は、各区間ではまっすぐ歩いている）
        w = max(2, round(fps))
        if cfg.zigzag_on and len(xy) > w:
            path = np.convolve(step, np.ones(w), mode="valid")  # 各1秒区間の道のり
            net = np.hypot(*(xy[w:] - xy[:-w]).T)[: len(path)]
            moving = path >= ZIGZAG_MIN_STEP_RATIO * size
            if moving.sum() >= 3 and float(np.median(net[moving] / path[moving])) < cfg.zigzag_straightness:
                out.setdefault(tid, []).append("zigzag")

    if cfg.offpath_on and len(tracks_xy) >= cfg.offpath_min_tracks:
        cell = max(4.0, OFFPATH_CELL_RATIO * size)
        gw, gh = int(math.ceil(width / cell)), int(math.ceil(height / cell))
        visits = np.zeros((gh, gw), np.int32)
        cells_of: dict[int, np.ndarray] = {}
        for tid, xy in tracks_xy.items():
            gx = np.clip((xy[:, 0] / cell).astype(int), 0, gw - 1)
            gy = np.clip((xy[:, 1] / cell).astype(int), 0, gh - 1)
            flat = gy * gw + gx
            cells_of[tid] = flat
            np.add.at(visits.reshape(-1), np.unique(flat), 1)  # 1人1マス1回だけ数える
        for tid, flat in cells_of.items():
            others = visits.reshape(-1)[flat] - 1  # 自分を除いた、そのマスを通った人数
            if np.mean(others == 0) >= cfg.offpath_share:
                out.setdefault(tid, []).append("offpath")
    return out


def reason_text(keys: list[str]) -> str:
    return "・".join(REASONS.get(k, k) for k in keys)
