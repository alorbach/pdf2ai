@echo off
setlocal enabledelayedexpansion

pushd %~dp0

if not exist .venv (
  echo [setup] Creating virtual environment at .venv
  py -m venv .venv
)

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
) else (
  echo [error] Failed to activate virtualenv (.venv\Scripts\activate.bat not found) 1>&2
  exit /b 1
)

python -m pip install --upgrade pip >nul
pip install -r requirements.txt

where tesseract >nul 2>nul
if errorlevel 1 (
  echo [hint] Tesseract not found. OCR will be disabled unless installed. 1>&2
  echo        Windows: Install from https://github.com/tesseract-ocr/tesseract 1>&2
  echo        Or via Chocolatey: choco install tesseract 1>&2
)

REM Poppler optional
where pdftoppm >nul 2>nul
if errorlevel 1 (
  echo [hint] Poppler utilities not found (optional). 1>&2
  echo        Windows: Install from https://github.com/oschwartz10612/poppler-windows 1>&2
)

python pdf2mm.py %*

popd
endlocal

