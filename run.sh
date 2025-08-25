#!/usr/bin/env bash
#
# pdf2ai runner script
# Author: Andre Lorbach <alorbach@adiscon.com>
# Copyright (c) 2025 Andre Lorbach
# License: MIT
#
# This code was created using AI Agent Models and human code oversight.
#
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

if [ ! -d .venv ]; then
  echo "[setup] Creating virtual environment at .venv"
  python3 -m venv .venv
fi

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[error] Failed to activate virtualenv (.venv/bin/activate not found)" >&2
  exit 1
fi

python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# Hints for external tools
if ! command -v tesseract >/dev/null 2>&1; then
  echo "[hint] Tesseract not found. OCR will be disabled unless installed." >&2
  echo "       Debian/Ubuntu: sudo apt-get install tesseract-ocr" >&2
  echo "       macOS (brew):  brew install tesseract" >&2
  echo "       Windows:       choco install tesseract" >&2
fi

if ! command -v pdftoppm >/dev/null 2>&1 && ! command -v pdftotext >/dev/null 2>&1; then
  echo "[hint] Poppler utilities not found (optional)." >&2
  echo "       Debian/Ubuntu: sudo apt-get install poppler-utils" >&2
  echo "       macOS (brew):  brew install poppler" >&2
fi

python pdf2ai.py "$@"

