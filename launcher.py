"""exe の入口（PyInstaller 用）。

  FootTrajectoryCounter.exe                 アプリ画面を開く
  FootTrajectoryCounter.exe 動画.mp4         その動画の解析から始める
  FootTrajectoryCounter.exe --cli ...       画面なしのコマンド版（python -m foot_overlay と同じ引数）
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def _redirect_output_if_windowed() -> None:
    """画面だけの exe では標準出力が無いので、ログファイルに書く（エラー調査用）。"""
    if sys.stdout is not None and sys.stderr is not None:
        return
    import os
    import tempfile

    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "FootTrajectoryCounter"
    try:
        base.mkdir(parents=True, exist_ok=True)
        log = open(base / "log.txt", "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    except OSError:
        log = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def main() -> int:
    multiprocessing.freeze_support()
    _redirect_output_if_windowed()

    from foot_overlay.runtime import configure_offline

    configure_offline()  # ultralytics を読み込む前に設定する必要がある

    args = sys.argv[1:]
    if args[:1] == ["--cli"]:
        from foot_overlay.__main__ import main as cli_main

        return cli_main(args[1:])
    from foot_overlay.app import main as app_main

    return app_main(args)


if __name__ == "__main__":
    sys.exit(main())
