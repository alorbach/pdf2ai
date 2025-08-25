@echo off
setlocal

pushd "%~dp0"

if not exist .venv (
  echo [setup] Creating virtual environment at .venv
  py -m venv .venv
)

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [error] Failed to activate virtualenv (.venv\Scripts\activate.bat not found) 1>&2
  exit /b 1
)

python -m pip install --upgrade pip >nul
pip install -r requirements.txt

python gui.py %*

popd
endlocal

