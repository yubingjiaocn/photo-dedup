#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
RUFF="${RUFF:-.venv/bin/ruff}"
"$RUFF" check src tests scripts
"$PY" -m compileall -q src tests scripts
"$PY" -m pytest -q \
  tests/test_db.py \
  tests/test_motion_photo.py \
  tests/test_cluster.py \
  tests/test_dataset_contract.py \
  tests/test_phase_selection.py \
  tests/test_offline_evaluation.py \
  tests/test_phase_ab.py \
  tests/test_event_policy.py \
  tests/test_p0_policy.py \
  tests/test_safety_e2e.py \
  tests/test_root_scope.py \
  tests/test_root_scope_gates.py \
  tests/test_root_scope_leaks.py \
  tests/test_stage1_telemetry.py \
  tests/test_stage1_telemetry_run.py \
  tests/test_stage1_model_failures.py \
  tests/test_stage1_oversize.py \
  tests/test_stage1_batched_iqa.py \
  tests/test_stage1_prefetch.py \
  tests/test_stage1_throughput.py \
  tests/test_stage1_resume.py \
  tests/test_thumbnails.py \
  tests/test_review_workbench_ui.py \
  tests/test_stage3_performance.py
