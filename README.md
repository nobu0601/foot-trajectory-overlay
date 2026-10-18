# foot-trajectory-overlay

mp4 動画に映る人物を画像解析で追跡し、**足元の通り道（軌跡）を動画の最初のフレームにすべて重ねた画像**を出力するツールです。
人の流れが一目でわかるので、人数カウント用の「カウント線」をどこに引くか決める参考にできます。

- 人物検出: YOLO11（ultralytics）
- 追跡: ByteTrack
- 足元の位置: 人物の枠の「下辺の中央」
- GPU なしの Windows PC で動きます

## 出力されるもの

動画と同じフォルダに `動画名_overlay/` が作られ、次の3つが入ります。

| ファイル | 内容 |
|---|---|
| `overlay.png` | 最初のフレームに全員の足元軌跡を重ねた画像 |
| `first_frame.png` | 何も描いていない最初のフレーム（カウント線を引く下絵用） |
| `tracks.csv` | 足元座標の生データ（`track_id, frame, time_sec, foot_x, foot_y`） |

`overlay.png` の見方:

- **線**: 1人分の足元の通り道（人ごとに色分け）
- **丸と番号**: 追跡を開始した位置と追跡ID
- **矢印**: 追跡が終わった位置と、そのときの進行方向
- **白い格子と数字**: 画像のピクセル座標（100px ごと）。カウント線の座標を読み取るのに使います

## 使い方（Windows）

1. [Python 3.11 または 3.12](https://www.python.org/downloads/) をインストールします（インストール時に「Add python.exe to PATH」にチェック）。
2. このリポジトリをダウンロードして展開します。
3. mp4 ファイルを `run.bat` に**ドラッグ＆ドロップ**します（複数まとめても可）。

初回だけ必要なライブラリと学習済みモデル（yolo11n.pt、約6MB）を自動でダウンロードするため、数分かかります。

処理時間の目安（CPU 4コア）: 768×576・10fps の動画 80 秒分で約 40 秒。

## コマンドで使う場合

```bat
.venv\Scripts\python.exe -m foot_overlay 動画.mp4 [オプション]
```

よく使うオプション:

| オプション | 説明 |
|---|---|
| `--step 2` | 2フレームに1回だけ処理して約2倍速にする（長い動画向け） |
| `--model yolo11s.pt` | 少し重いが精度の高いモデルを使う |
| `--imgsz 960` | 遠くの人が小さく映っていて見逃すときに大きくする |
| `--conf 0.4` | 人以外を誤検出するときに上げる（見逃すときは下げる） |
| `--min-points 20` | 短い（ノイズっぽい）軌跡をもっと除外する |
| `--grid 0` | 座標の格子を消す |
| `--no-ids` | 追跡ID番号を描かない |
| `--dim 0` | 背景を暗くしない |
| `--max-frames 300` | 先頭300フレームだけでお試し実行 |
| `-o 出力フォルダ` | 出力先を指定する |

すべてのオプションは `--help` で確認できます。

## 開発者向け

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # Windows は .venv\Scripts\pip
.venv/bin/python -m pytest
```

構成:

```txt
foot_overlay/
  tracking.py   動画の読み込みと YOLO + ByteTrack による追跡、足元座標の収集
  drawing.py    軌跡の整形（短いIDの除外・移動平均）と重畳描画
  export.py     画像・CSV の保存（日本語パス対応）
  __main__.py   コマンドライン
tests/          描画・保存まわりのテスト（モデル不要）
```

## ライセンスについて

人物検出に使っている [ultralytics](https://github.com/ultralytics/ultralytics) は AGPL-3.0 ライセンスです。
社内で使うだけなら問題になりにくいですが、このツールを組み込んだ製品・サービスを外部に提供する場合は、AGPL の条件を満たすか ultralytics の商用ライセンスが必要です。

## 今後の予定

- 重畳画像の上でカウント線を引き、座標を保存する機能
- `tracks.csv` を使った、カウント線の通過人数（方向別）の集計
