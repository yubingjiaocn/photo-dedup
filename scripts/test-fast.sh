#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
RUFF="${RUFF:-.venv/bin/ruff}"
"$RUFF" check src tests
"$PY" -m compileall -q src tests
"$PY" -m pytest -q \
  tests/test_db.py \
  tests/test_motion_photo.py \
  tests/test_cluster.py \
  tests/test_p0_policy.py \
  tests/test_safety_e2e.py \
  tests/test_root_scope.py \
  tests/test_stage1_telemetry.py \
  tests/test_stage1_model_failures.py \
  tests/test_stage1_oversize.py \
  tests/test_thumbnails.py \
  tests/test_stage3_performance.py
