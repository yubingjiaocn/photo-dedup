#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
RUFF="${RUFF:-.venv/bin/ruff}"
"$RUFF" check src tests scripts
"$PY" -m compileall -q src tests scripts
"$PY" -m pytest -q

git diff --check
