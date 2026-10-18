@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Drag and drop mp4 file^(s^) onto run.bat.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating Python environment. This runs only the first time...
  py -3 -m venv .venv 2>nul || python -m venv .venv
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

".venv\Scripts\python.exe" -m foot_overlay %*
echo.
pause
