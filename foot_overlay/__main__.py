"""コマンドライン入口。

  python -m foot_overlay                       アプリ画面を開く
  python -m foot_overlay 動画.mp4               解析して重畳画像を保存（画面なし）
  python -m foot_overlay 動画名_overlay --line x1,y1,x2,y2
                                               保存済みの結果でカウント線の通過人数を数える
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .counting import parse_line_arg
from .drawing import filter_short_tracks
from .session import Session, write_result
from .tracking import track_video


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="foot_overlay",
        description=(
            "mp4動画の全フレームで人物を追跡し、足元の軌跡を最初のフレームに重ねた画像を出力します。"
            "引数なしで起動するとアプリ画面が開きます。"
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="動画ファイル、または出力済みの 動画名_overlay フォルダ（複数可）")
    p.add_argument("-o", "--out-dir", type=Path, help="出力先フォルダ（省略時は 動画名_overlay/ を動画の隣に作成）")
    p.add_argument("--model", default="yolo11n.pt", help="YOLOモデル（既定: yolo11n.pt。精度重視なら yolo11s.pt）")
    p.add_argument("--conf", type=float, default=0.3, help="人物と判定する確信度のしきい値（既定: 0.3）")
    p.add_argument("--imgsz", type=int, default=640, help="推論時の画像サイズ（既定: 640。遠くの人が小さいなら 960 など）")
    p.add_argument("--min-points", type=int, default=10, help="この回数未満しか検出されなかったIDは描かない（既定: 10）")
    p.add_argument("--smooth", type=int, default=5, help="軌跡をならす移動平均の幅（既定: 5。1で無効）")
    p.add_argument("--line-width", type=int, help="追跡線の太さpx（省略時は画像サイズに合わせて自動）")
    p.add_argument("--dim", type=float, default=0.35, help="背景を暗くする割合 0〜1（既定: 0.35）")
    p.add_argument("--grid", type=int, default=100, help="座標グリッドの間隔px（既定: 100。0で非表示）")
    p.add_argument("--no-ids", action="store_true", help="軌跡の始点にID番号を書かない")
    p.add_argument("--tracker", default="bytetrack.yaml", help="追跡アルゴリズム設定（既定: bytetrack.yaml）")
    p.add_argument("--device", default="cpu", help="推論デバイス（既定: cpu。NVIDIA GPU なら 0）")
    p.add_argument("--max-frames", type=int, help="先頭から指定フレーム数だけ処理する（お試し用）")
    p.add_argument(
        "--line",
        action="append",
        metavar="x1,y1,x2,y2",
        help="カウント線を座標で指定する（複数回指定可）。線をまたいだ人数を数えて保存する",
    )
    p.add_argument(
        "--suggest",
        nargs="?",
        type=int,
        const=20000,
        metavar="本数",
        help="エリアを自動検出し（areas.json があればそれを使う）、エリアごとにカウント線の候補を3本提案する。"
        "本数はエリアごとに評価する候補線の数（既定: 20000）",
    )
    return p.parse_args(argv)


def process_video(video: Path, args: argparse.Namespace) -> Path:
    out_dir = args.out_dir or video.with_name(f"{video.stem}_overlay")
    if args.out_dir and len(args.inputs) > 1:
        out_dir = args.out_dir / video.stem

    print(f"\n▶ {video.name}")
    started = time.time()
    result = track_video(
        video,
        model_name=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker=args.tracker,
        device=args.device,
        max_frames=args.max_frames,
    )
    tracks = filter_short_tracks(result.tracks, args.min_points)
    view = {"line_width": args.line_width, "show_ids": not args.no_ids, "grid": args.grid, "dim": args.dim}
    write_result(out_dir, video, result, tracks, smooth=args.smooth, min_points=args.min_points, view=view)
    Session(out_dir).save()  # overlay.png を描く

    h, w = result.first_frame.shape[:2]
    print(f"  動画: {w}x{h}, {result.fps:.1f}fps, {result.total_frames}フレーム（全フレーム解析）")
    print(f"  追跡ID: {len(result.tracks)}件 → 描画: {len(tracks)}件（{args.min_points}回未満の短いIDを除外）")
    print(f"  処理時間: {time.time() - started:.0f}秒")
    print(f"  出力先: {out_dir}")
    return out_dir


def count_with_lines(result_dir: Path, line_args: list[str], args: argparse.Namespace) -> None:
    session = Session(result_dir)
    if args.line_width:
        session.line_width = args.line_width
    session.set_lines([parse_line_arg(t) for t in line_args])
    session.save()
    print(f"\n▶ カウント結果（{result_dir.name}）")
    for text in session.summary_lines():
        print(f"  {text}")


def suggest(result_dir: Path, n_candidates: int) -> None:
    session = Session(result_dir)
    if not session.areas:
        session.auto_detect_areas()
    print(f"\n▶ カウント線の候補（{result_dir.name}・エリア{len(session.areas)}件・各{n_candidates}本を評価）")
    session.suggestions = session.compute_suggestions(n_candidates=n_candidates)
    session.save()
    for i, area in enumerate(session.areas):
        print(f"  E{i + 1}（{area.kind_label}）: エリアを通った人 {session.area_people(i)}人")
        for sg in [s for s in session.suggestions if s.area_index == i]:
            ln = sg.line
            print(
                f"    候補{sg.rank}: ({ln.x1:.0f},{ln.y1:.0f})-({ln.x2:.0f},{ln.y2:.0f})"
                f"  人数 {sg.people}人（{sg.coverage:.0%}） 通過 {sg.passes}回 二重カウント {sg.duplicates}回"
            )
    print(f"  保存先: {result_dir / 'suggestions.png'}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from .app import main as app_main

        return app_main([])

    args = parse_args(argv)
    failed = 0
    for path in args.inputs:
        try:
            if path.is_file():
                path = process_video(path, args)
            elif not path.is_dir():
                print(f"ファイルが見つかりません: {path}", file=sys.stderr)
                failed += 1
                continue
            if args.suggest:
                suggest(path, args.suggest)
            if args.line:
                count_with_lines(path, args.line, args)
        except Exception as e:  # 1本失敗しても残りは処理する
            print(f"\n✖ {path.name} の処理に失敗しました: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
