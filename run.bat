@echo off
setlocal
cd /d "%~dp0"

rem Double-click to open the app, or drag and drop an mp4 file onto this file.

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating Python environment. This runs only the first time...
  py -3 -m venv .venv 2>/dev/null || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Python was not found. Install Python 3.11 or 3.12 from https://www.python.org/
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [setup] Failed to install packages.
    rmdir /s /q .venv
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" -m foot_overlay.app %*
if errorlevel 1 pause
