"""コマンドライン入口: python -m foot_overlay 動画.mp4"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .drawing import draw_overlay, filter_short_tracks
from .export import save_image, save_tracks_csv
from .tracking import track_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="foot_overlay",
        description="mp4動画の人物を追跡し、足元の軌跡を最初のフレームに重ねた画像を出力します。",
    )
    p.add_argument("videos", nargs="+", type=Path, help="入力する動画ファイル（複数可）")
    p.add_argument("-o", "--out-dir", type=Path, help="出力先フォルダ（省略時は 動画名_overlay/ を動画の隣に作成）")
    p.add_argument("--model", default="yolo11n.pt", help="YOLOモデル（既定: yolo11n.pt。精度重視なら yolo11s.pt）")
    p.add_argument("--conf", type=float, default=0.3, help="人物と判定する確信度のしきい値（既定: 0.3）")
    p.add_argument("--imgsz", type=int, default=640, help="推論時の画像サイズ（既定: 640。遠くの人が小さいなら 960 など）")
    p.add_argument("--step", type=int, default=1, help="何フレームごとに処理するか（既定: 1。2にすると約2倍速）")
    p.add_argument("--min-points", type=int, default=10, help="この回数未満しか検出されなかったIDは描かない（既定: 10）")
    p.add_argument("--smooth", type=int, default=5, help="軌跡をならす移動平均の幅（既定: 5。1で無効）")
    p.add_argument("--dim", type=float, default=0.35, help="背景を暗くする割合 0〜1（既定: 0.35）")
    p.add_argument("--grid", type=int, default=100, help="座標グリッドの間隔px（既定: 100。0で非表示）")
    p.add_argument("--no-ids", action="store_true", help="軌跡の始点にID番号を書かない")
    p.add_argument("--tracker", default="bytetrack.yaml", help="追跡アルゴリズム設定（既定: bytetrack.yaml）")
    p.add_argument("--device", default="cpu", help="推論デバイス（既定: cpu。NVIDIA GPU なら 0）")
    p.add_argument("--max-frames", type=int, help="先頭から指定フレーム数だけ処理する（お試し用）")
    return p.parse_args(argv)


def process(video: Path, args: argparse.Namespace) -> Path:
    out_dir = args.out_dir or video.with_name(f"{video.stem}_overlay")
    if args.out_dir and len(args.videos) > 1:
        out_dir = args.out_dir / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▶ {video.name}")
    started = time.time()
    result = track_video(
        video,
        model_name=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        frame_step=max(1, args.step),
        tracker=args.tracker,
        device=args.device,
        max_frames=args.max_frames,
    )
    # 間引き処理したときは、必要な検出回数もその分減らす
    min_points = max(2, args.min_points // max(1, args.step))
    tracks = filter_short_tracks(result.tracks, min_points)

    overlay = draw_overlay(
        result.first_frame,
        tracks,
        smooth_window=args.smooth,
        dim=args.dim,
        show_ids=not args.no_ids,
        grid=args.grid,
    )
    save_image(out_dir / "overlay.png", overlay)
    save_image(out_dir / "first_frame.png", result.first_frame)
    save_tracks_csv(out_dir / "tracks.csv", tracks, result.fps)

    h, w = result.first_frame.shape[:2]
    print(f"  動画: {w}x{h}, {result.fps:.1f}fps, {result.total_frames}フレーム")
    print(f"  追跡ID: {len(result.tracks)}件 → 描画: {len(tracks)}件（{min_points}回未満の短いIDを除外）")
    print(f"  処理時間: {time.time() - started:.0f}秒")
    print(f"  出力先: {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failed = 0
    for video in args.videos:
        if not video.is_file():
            print(f"ファイルが見つかりません: {video}", file=sys.stderr)
            failed += 1
            continue
        try:
            process(video, args)
        except Exception as e:  # 1本失敗しても残りの動画は処理する
            print(f"\n✖ {video.name} の処理に失敗しました: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
