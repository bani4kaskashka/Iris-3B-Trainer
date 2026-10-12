@echo off
setlocal
cd /d "%~dp0"
title Iris-3B Trainer

rem Python for the dashboard and its jobs: IRIS_PYTHON if set, else .venv (made on the first run)
set "PY=%IRIS_PYTHON%"
if not defined PY (
  if not exist ".venv\iris-ready" call :setup
  if not exist ".venv\iris-ready" goto :failed
  set "PY=%~dp0.venv\Scripts\python.exe"
)

rem upstream Iris-3B code (model, loss, text encoder), unless settings.json points somewhere else
"%PY%" -c "import settings, sys; sys.exit(0 if (settings.iris_repo() / 'src' / 'iris3b').is_dir() else 1)" 2>nul
if errorlevel 1 call :clone_iris

echo.
echo Starting the dashboard. Keep this window open; close it to stop the dashboard.
echo Jobs keep running in the background either way.
"%PY%" dashboard.py --open %*
if errorlevel 1 goto :failed
exit /b 0


:setup
echo First run: setting up Python in .venv (a few GB of downloads)
set "BASE="
py -3.12 -c "" >nul 2>nul && set "BASE=py -3.12"
if not defined BASE py -3.13 -c "" >nul 2>nul && set "BASE=py -3.13"
if not defined BASE python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "BASE=python"
if not defined BASE (
  echo Python 3.12 or 3.13 was not found. Get it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^), then run start.bat again.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" %BASE% -m venv .venv
if not exist ".venv\Scripts\python.exe" exit /b 1
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
echo.
echo Which GPU is this for?
echo   1  NVIDIA
echo   2  AMD Radeon RX 9000 series (RDNA 4)
echo   3  AMD Radeon RX 7000 series (RDNA 3)
echo   4  Skip, I'll install PyTorch into .venv myself
choice /c 1234 /n /m "Pick 1-4: "
if errorlevel 4 goto :deps
if errorlevel 3 (set "TORCH_INDEX=https://rocm.nightlies.amd.com/v2/gfx110X-all/" & goto :torch)
if errorlevel 2 (set "TORCH_INDEX=https://rocm.nightlies.amd.com/v2/gfx120X-all/" & goto :torch)
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
:torch
"%VPY%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 exit /b 1
:deps
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo ok> ".venv\iris-ready"
echo.
echo Setup done. Next: open the Models card in the dashboard and press Start to download Iris-3B,
echo or point models_dir in the Settings tab at a ComfyUI models folder that already has it.
exit /b 0


:clone_iris
where git >nul 2>nul
if errorlevel 1 (
  echo Git was not found, so the Iris-3B code can't be fetched. Install Git from https://git-scm.com
  echo or download https://github.com/speridlabs/iris-3b as a zip into an "iris-3b" folder here.
  exit /b 0
)
echo Fetching the Iris-3B code from github.com/speridlabs/iris-3b
git clone --depth 1 https://github.com/speridlabs/iris-3b iris-3b
exit /b 0


:failed
echo.
echo Something went wrong, see the messages above.
pause
exit /b 1
