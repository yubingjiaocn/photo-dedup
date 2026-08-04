#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
RUFF="${RUFF:-.venv/bin/ruff}"
"$RUFF" check src tests
"$PY" -m compileall -q src tests
"$PY" -m pytest -q

git diff --check
