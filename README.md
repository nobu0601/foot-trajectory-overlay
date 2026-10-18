# foot-trajectory-overlay（足元軌跡カウントツール）

mp4 動画に映る人物を画像解析で追跡し、**足元の通り道（軌跡）を動画の最初のフレームにすべて重ねた画像**を作ります。
その画像を見ながらマウスで**カウント線**を引くと、**線をまたいだ人数（向き別・合計）**がその場で表示されます。

- 人物検出: YOLO11（ultralytics）／追跡: ByteTrack
- 解析は**全フレーム**で行います（間引きなし）
- 足元の位置: 人物の枠の「下辺の中央」
- GPU なしの Windows PC で動きます

## 使い方（Windows）

### exe 版（おすすめ・オフラインPCで使える）

Python のインストールは不要です。インターネットにつながっていない PC でも動きます。

1. リポジトリの **Releases** から `FootTrajectoryCounter-windows.zip` をダウンロードします（USBメモリ等でオフラインPCへ移して構いません）。
2. zip を展開し、`FootTrajectoryCounter` フォルダ内の **`FootTrajectoryCounter.exe`** をダブルクリックします。
   （mp4 ファイルを exe にドラッグ＆ドロップすると、その動画の解析からすぐ始まります）

- フォルダごと置いて使ってください（exe だけを取り出すと動きません）。
- 検出モデル（標準・高精度の2種類）は同梱済みで、ネットへの接続は一切行いません。
- 初回起動は 10〜20 秒ほどかかることがあります。
- うまく動かないときは `%LOCALAPPDATA%\FootTrajectoryCounter\log.txt` にエラー内容が残ります。

### Python 版

1. [Python 3.12](https://www.python.org/downloads/) をインストールします（インストール時に「Add python.exe to PATH」にチェック）。
2. このリポジトリをダウンロードして展開します（緑の「Code」ボタン →「Download ZIP」）。
3. `run.bat` を**ダブルクリック**するとアプリが開きます（初回だけライブラリと検出モデルを自動でダウンロードするため、数分かかります）。

### 操作

1. アプリを開きます（上のどちらかの方法）。
2. **「動画を開いて解析…」** で mp4 を選びます。下の欄に進み具合と残り時間が出ます。途中でやめるときは「解析を中止」。
3. 解析が終わると、足元軌跡を重ねた画像が表示されます。
   - **線**: 1人分の足元の通り道（人ごとに色分け）
   - **丸と番号**: 追跡を開始した位置と追跡ID
   - **矢印**: 追跡が終わった位置と、そのときの進行方向
   - **白い格子と数字**: 画像のピクセル座標（100px ごと）
4. 右側の **「追跡線の太さ」スライダー** で、軌跡の線の太さをいつでも変えられます。背景の暗さ・ID番号・格子の表示も切り替えられます。
5. 画像を**2回クリック**（始点 → 終点）すると**カウント線**が引かれ、右の表に人数が出ます。何本でも引けます。
   - **A→B / B→A**: 線の中央の矢印の向き（A側からB側）に通った人数と、その逆向きの人数
   - **合計**: 両方向の合計
   - 線の上の**赤い点**が、実際に数えた通過位置です
   - 右クリック / Esc: 引きかけの線を取り消し
   - Ctrl+Z: 最後の線を削除。表で選んで「選んだ線を削除」もできます

結果は動画と同じフォルダの `動画名_overlay/` に**自動保存**されます（「保存フォルダを開く」ボタンで開けます）。
次回は「解析結果を開く…」でこのフォルダを選ぶと、解析し直さずに続きから線を引けます。

### 保存されるファイル

| ファイル | 内容 |
|---|---|
| `overlay.png` | 最初のフレームに全員の足元軌跡を重ねた画像 |
| `count_result.png` | 上の画像＋カウント線＋線ごとの人数 |
| `counts.csv` | 線ごとの人数（線の座標, A→B, B→A, 合計） |
| `crossings.csv` | 1回ごとの通過記録（線, 追跡ID, フレーム, 秒, 向き, 位置） |
| `first_frame.png` | 何も描いていない最初のフレーム |
| `tracks.csv` | 足元座標の生データ（`track_id, frame, time_sec, foot_x, foot_y`） |
| `lines.json` / `meta.json` | カウント線と表示設定（次回開いたときに復元） |

CSV は Excel でそのまま開けます。

## 数え方のルール

- 重畳画像と同じく、足元の座標を少しならしてから判定します（画像で見た線と結果が一致します）。
- 足元が線の上で小刻みに揺れても何度も数えないよう、線から少し離れて初めて「渡った」とみなします。
- 線の端より外側を回り込んだ人は数えません。
- 同じ人が線を行って戻った場合は、A→B と B→A に1回ずつ数えます。
- 人物が物陰に隠れて追跡IDが途切れると、同じ人が別IDになることがあります。人の重なりが多い場所より、見通しのよい場所に線を引くと正確になります。

## 処理時間の目安

CPU 4コアで、768×432・12fps・50秒の動画が約25秒。
フルHD・30fps の動画では、動画の長さの数倍かかることがあります（解析は全フレームで行うため）。
精度を上げたいときはツールバーの「検出モデル」を「高精度」にします（約3倍遅くなります）。

## コマンドで使う場合

画面を使わずに処理することもできます。

```bat
rem 解析して重畳画像を保存
.venv\Scripts\python.exe -m foot_overlay 動画.mp4 --line-width 4

rem 保存済みの結果に、座標を指定したカウント線を引いて数える（複数指定可）
.venv\Scripts\python.exe -m foot_overlay 動画_overlay --line 100,300,600,300 --line 400,100,400,500
```

exe 版では `FootTrajectoryCounter.exe --cli 動画.mp4 ...` で同じことができます（結果はファイルに出力されます）。

主なオプション: `--model yolo11s.pt`（高精度）、`--imgsz 960`（遠くの小さい人を見逃すとき）、
`--conf 0.4`（誤検出が多いとき）、`--min-points 20`（短いノイズ軌跡を除く）、`--max-frames 300`（お試し）。
すべてのオプションは `--help` で確認できます。

## exe の作り方（開発者向け）

GitHub Actions（`.github/workflows/build-windows.yml`）が Windows 上で自動的に作ります。

- `v0.2.0` のような `v` で始まるタグを push すると、ビルド・動作確認のあと Releases に zip が公開されます。
- Actions 画面の「Build Windows exe」→「Run workflow」で手動実行もできます（zip は実行結果の Artifacts から取得）。

手元の Windows で作る場合は、`models/` に `yolo11n.pt` と `yolo11s.pt` を置いてから次を実行します。

```bat
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm FootTrajectoryCounter.spec
```

`dist\FootTrajectoryCounter\` に exe 一式ができます。

## 開発者向け

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # Windows は .venv\Scripts\pip
.venv/bin/python -m pytest
```

```txt
foot_overlay/
  app.py        アプリ画面（tkinter）
  tracking.py   動画の読み込みと YOLO + ByteTrack による全フレーム追跡
  drawing.py    軌跡の整形（短いIDの除外・移動平均）と重畳描画
  counting.py   カウント線の通過判定（向き別）と CSV/JSON の読み書き
  session.py    結果フォルダの読み書き、表示設定、カウント線の編集状態
  export.py     画像・CSV の保存（日本語パス対応）
  runtime.py    exe 化したときの同梱モデルの場所・オフライン設定
  __main__.py   コマンドライン
launcher.py     exe の入口（--cli でコマンド版）
FootTrajectoryCounter.spec  exe 化の設定（PyInstaller）
tests/          判定・描画・保存のテスト（モデル不要）
```

## ライセンスについて

人物検出に使っている [ultralytics](https://github.com/ultralytics/ultralytics) は AGPL-3.0 ライセンスです。
社内で使うだけなら問題になりにくいですが、このツールを組み込んだ製品・サービスを外部に提供する場合は、AGPL の条件を満たすか ultralytics の商用ライセンスが必要です。
