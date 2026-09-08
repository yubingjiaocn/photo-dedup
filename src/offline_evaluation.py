"""Offline annotation schema and baseline/candidate comparison metrics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dataset_contract import DatasetManifest, Purpose, authorize

ANNOTATION_SCHEMA_VERSION = 1
_REQUIRED = {"schema_version", "event_id", "group_id", "phase_ids", "acceptable_keepers"}


def validate_annotation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUIRED:
        raise ValueError("annotation has invalid keys")
    if value["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("annotation schema_version must be 1")
    phases = value["phase_ids"]
    keepers = value["acceptable_keepers"]
    if not isinstance(phases, Mapping) or not phases:
        raise ValueError("phase_ids must be a non-empty mapping")
    if not isinstance(keepers, Mapping) or set(keepers) != set(phases):
        raise ValueError("acceptable_keepers must cover exactly the annotated phases")
    if any(not isinstance(items, list) or not items for items in keepers.values()):
        raise ValueError("each phase needs at least one acceptable keeper")
    return dict(value)


def evaluate_predictions(annotations: Sequence[Mapping[str, Any]], predictions: Mapping[str, Sequence[str]]) -> dict[str, float]:
    total_phases = 0
    covered_phases = 0
    selected_total = 0
    acceptable_selected = 0
    for raw in annotations:
        annotation = validate_annotation(raw)
        group_id = str(annotation["group_id"])
        selected = set(predictions.get(group_id, ()))
        selected_total += len(selected)
        for acceptable in annotation["acceptable_keepers"].values():
            total_phases += 1
            acceptable_set = set(map(str, acceptable))
            overlap = selected & acceptable_set
            covered_phases += bool(overlap)
            acceptable_selected += len(overlap)
    return {
        "phase_recall": covered_phases / total_phases if total_phases else 0.0,
        "keeper_precision": acceptable_selected / selected_total if selected_total else 0.0,
        "selected_per_phase": selected_total / total_phases if total_phases else 0.0,
        "phase_count": float(total_phases),
    }


def compare(
    manifest: DatasetManifest, annotations: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    contract = authorize(manifest, Purpose.FINAL_EVALUATION, labels_requested=True)
    base = evaluate_predictions(annotations, baseline)
    cand = evaluate_predictions(annotations, candidate)
    return {
        "schema_version": 1,
        "contract": contract,
        "baseline": base,
        "candidate": cand,
        "delta": {key: cand[key] - base[key] for key in ("phase_recall", "keeper_precision", "selected_per_phase")},
        "writeback": "forbidden",
    }
