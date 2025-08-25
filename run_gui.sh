#!/usr/bin/env bash
#
# pdf2ai GUI runner script
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

python gui.py "$@"

