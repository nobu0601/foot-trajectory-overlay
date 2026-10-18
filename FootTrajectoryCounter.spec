# PyInstaller 設定: pyinstaller --noconfirm FootTrajectoryCounter.spec
# 事前に models/yolo11n.pt と models/yolo11s.pt を置いておくこと（exe に同梱される）
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

models = [(str(p), "models") for p in Path("models").glob("*.pt")]
if not models:
    raise SystemExit("models/ に検出モデル(.pt)がありません。オフラインで動かすには同梱が必要です。")

datas = models
datas += collect_data_files("ultralytics")  # bytetrack.yaml などの設定ファイル
for pkg in ("ultralytics", "lap", "torch", "torchvision", "opencv-python", "numpy", "pillow"):
    datas += copy_metadata(pkg)  # ライブラリのバージョン確認がオフラインでも通るように

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    datas=datas,
    hiddenimports=collect_submodules("ultralytics") + ["lap", "PIL._tkinter_finder"],
    excludes=["matplotlib", "pandas", "IPython", "jupyter", "notebook", "tensorboard"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FootTrajectoryCounter",
    console=False,  # 黒い画面を出さない
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="FootTrajectoryCounter", upx=False)
