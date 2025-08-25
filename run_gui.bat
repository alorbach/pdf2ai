@echo off
setlocal

pushd "%~dp0"

REM Use current active venv if present
if defined VIRTUAL_ENV (
  echo [info] Using active venv: %VIRTUAL_ENV%
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt
  python gui.py %*
  goto :END
)

if not exist .venv (
  echo [setup] Creating virtual environment at .venv
  call :MAKE_VENV
)

if not exist ".venv\Scripts\activate.bat" (
  echo [error] Failed to activate virtualenv (.venv\Scripts\activate.bat not found) 1>&2
  echo        Ensure Python 3.9+ is installed and 'py' or 'python' is on PATH. 1>&2
  goto :END
)

call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip >nul
pip install -r requirements.txt

python gui.py %*

goto :END

:MAKE_VENV
where py >nul 2>nul
if %errorlevel%==0 (
  py -m venv .venv
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python -m venv .venv
  ) else (
    echo [error] Python not found. Install Python 3 and add to PATH. 1>&2
    exit /b 1
  )
)
exit /b 0

:END
popd
endlocal

