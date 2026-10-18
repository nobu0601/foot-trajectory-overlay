"""実行環境まわり（exe化したときの同梱ファイルの場所、オフライン動作）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """PyInstaller で exe 化されて動いているか。"""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """exe（またはリポジトリ）のあるフォルダ。"""
    return Path(sys.executable).resolve().parent if is_frozen() else PROJECT_ROOT


def bundle_dir() -> Path:
    """exe に同梱したファイルの展開先。"""
    return Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))


def resolve_model(name: str) -> str:
    """モデル名を、同梱・手元にあるファイルのパスに直す。

    見つかればそのパスを返すので、ultralytics がネットからダウンロードしに行かない。
    見つからなければ名前をそのまま返す（ネットにつながっていれば自動ダウンロードされる）。
    """
    path = Path(name)
    if path.is_file():
        return str(path)
    for folder in (app_dir() / "models", bundle_dir() / "models", PROJECT_ROOT / "models", Path.cwd()):
        candidate = folder / path.name
        if candidate.is_file():
            return str(candidate)
    if is_frozen():
        raise FileNotFoundError(f"検出モデル {path.name} が見つかりません。exe と同じ場所の models フォルダに置いてください。")
    return name


def configure_offline() -> None:
    """exe 版はオフラインPCで使う前提なので、ultralytics のネット接続を止める。

    ultralytics は起動時の接続確認・利用状況の送信・不足ライブラリの自動インストールを
    行おうとするため、ここで無効にしておく（オフラインで待たされたり失敗したりしないように）。
    """
    if not is_frozen():
        return
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    # 設定ファイルを exe の隣ではなく利用者ごとの場所に置く（書き込み権限の問題を避ける）
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "FootTrajectoryCounter"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(base / "ultralytics"))


def disable_ultralytics_sync() -> None:
    """ultralytics の利用状況送信を止める（ultralytics を import した後に呼ぶ）。"""
    try:
        from ultralytics.utils import SETTINGS

        if SETTINGS.get("sync", False):
            SETTINGS.update({"sync": False})
    except Exception:
        pass
