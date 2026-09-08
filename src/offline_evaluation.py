"""Strict offline annotation validation and auditable evaluation metrics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dataset_contract import DatasetManifest, Purpose, authorize

ANNOTATION_SCHEMA_VERSION = 1
_REQUIRED = {
    "schema_version", "event_id", "group_id", "member_ids", "phase_ids",
    "acceptable_keepers", "group_impure", "phase_undersegmented", "risk_member_ids",
}
_PREDICTION_REQUIRED = {"keepers", "phases", "review_member_ids"}


def _id(value: Any, label: str) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty string or integer")
    return str(value)


def _id_list(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list")
    values = [_id(item, label) for item in value]
    if not allow_empty and not values:
        raise ValueError(f"{label} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate member IDs")
    return values


def validate_annotation(
    value: Mapping[str, Any], *, expected_events: set[str] | None = None,
) -> dict[str, Any]:
    """Validate IDs, exact phase partition, keeper bounds, and event scope."""
    if not isinstance(value, Mapping) or set(value) != _REQUIRED:
        raise ValueError("annotation has invalid keys")
    if value["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("annotation schema_version must be 1")
    event_id = _id(value["event_id"], "event_id")
    group_id = _id(value["group_id"], "group_id")
    if expected_events is not None and event_id not in expected_events:
        raise ValueError(f"annotation event_id {event_id!r} is outside the manifest")
    members = _id_list(value["member_ids"], "member_ids")
    member_set = set(members)
    phases = value["phase_ids"]
    keepers = value["acceptable_keepers"]
    if not isinstance(phases, Mapping) or not phases:
        raise ValueError("phase_ids must be a non-empty mapping")
    if not isinstance(keepers, Mapping) or set(map(str, keepers)) != set(map(str, phases)):
        raise ValueError("acceptable_keepers must cover exactly the annotated phases")
    normal_phases: dict[str, list[str]] = {}
    normal_keepers: dict[str, list[str]] = {}
    seen: set[str] = set()
    for raw_phase, raw_members in phases.items():
        phase_id = _id(raw_phase, "phase_id")
        phase_members = _id_list(raw_members, f"phase_ids[{phase_id}]")
        outside = set(phase_members) - member_set
        overlap = set(phase_members) & seen
        if outside:
            raise ValueError(f"phase {phase_id} references members outside group: {sorted(outside)}")
        if overlap:
            raise ValueError(f"phase partitions overlap: {sorted(overlap)}")
        seen.update(phase_members)
        acceptable = _id_list(keepers[raw_phase], f"acceptable_keepers[{phase_id}]")
        if not set(acceptable) <= set(phase_members):
            raise ValueError(f"acceptable keepers for {phase_id} fall outside that phase")
        normal_phases[phase_id] = phase_members
        normal_keepers[phase_id] = acceptable
    if seen != member_set:
        raise ValueError(f"phase partition must cover every member exactly once; missing={sorted(member_set-seen)}")
    if not isinstance(value["group_impure"], bool) or not isinstance(value["phase_undersegmented"], bool):
        raise ValueError("group_impure and phase_undersegmented must be booleans")
    risk = _id_list(value["risk_member_ids"], "risk_member_ids", allow_empty=True)
    if not set(risk) <= member_set:
        raise ValueError("risk_member_ids must stay within member_ids")
    return {
        "schema_version": 1, "event_id": event_id, "group_id": group_id,
        "member_ids": members, "phase_ids": normal_phases,
        "acceptable_keepers": normal_keepers,
        "group_impure": value["group_impure"],
        "phase_undersegmented": value["phase_undersegmented"],
        "risk_member_ids": risk,
    }


def _validate_prediction(raw: Any, annotation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _PREDICTION_REQUIRED:
        raise ValueError(f"prediction for group {annotation['group_id']} has invalid keys")
    member_set = set(annotation["member_ids"])
    keepers = _id_list(raw["keepers"], "prediction keepers")
    if not set(keepers) <= member_set:
        raise ValueError("prediction keeper falls outside group")
    phases = raw["phases"]
    if not isinstance(phases, Mapping) or not phases:
        raise ValueError("prediction phases must be a non-empty mapping")
    seen: set[str] = set()
    normal_phases: dict[str, list[str]] = {}
    for raw_id, raw_members in phases.items():
        phase_id = _id(raw_id, "prediction phase id")
        items = _id_list(raw_members, f"prediction phase {phase_id}")
        if not set(items) <= member_set:
            raise ValueError("prediction phase member falls outside group")
        if seen & set(items):
            raise ValueError("prediction phases overlap")
        seen.update(items)
        normal_phases[phase_id] = items
    if seen != member_set:
        raise ValueError("prediction phases must cover every group member exactly once")
    review = _id_list(raw["review_member_ids"], "review_member_ids", allow_empty=True)
    if not set(review) <= member_set:
        raise ValueError("review_member_ids must stay within the group")
    return {"keepers": keepers, "phases": normal_phases, "review_member_ids": review}


def _undersegmented(annotation: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    annotated_phase = {
        member: phase for phase, members in annotation["phase_ids"].items() for member in members
    }
    return any(
        len({annotated_phase[member] for member in members}) > 1
        for members in prediction["phases"].values()
    )


def evaluate_predictions(
    annotations: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]],
    *, expected_events: set[str] | None = None,
) -> dict[str, float]:
    total_phases = covered_phases = selected_total = acceptable_selected = 0
    member_total = review_count = risk_total = risk_reviewed = 0
    impure_total = impure_reviewed = undersegmented_total = 0
    seen_groups: set[str] = set()
    for raw in annotations:
        annotation = validate_annotation(raw, expected_events=expected_events)
        group_id = annotation["group_id"]
        if group_id in seen_groups:
            raise ValueError(f"duplicate annotation group_id: {group_id}")
        seen_groups.add(group_id)
        if group_id not in predictions:
            raise ValueError(f"missing prediction for annotated group {group_id}")
        prediction = _validate_prediction(predictions[group_id], annotation)
        selected = set(prediction["keepers"])
        review = set(prediction["review_member_ids"])
        selected_total += len(selected)
        member_total += len(annotation["member_ids"])
        review_count += len(review)
        risks = set(annotation["risk_member_ids"])
        risk_total += len(risks)
        risk_reviewed += len(risks & review)
        if annotation["group_impure"]:
            impure_total += 1
            impure_reviewed += bool(review)
        predicted_undersegmented = _undersegmented(annotation, prediction)
        undersegmented_total += predicted_undersegmented
        for acceptable in annotation["acceptable_keepers"].values():
            total_phases += 1
            overlap = selected & set(acceptable)
            covered_phases += bool(overlap)
            acceptable_selected += len(overlap)
    extra = set(map(str, predictions)) - seen_groups
    if extra:
        raise ValueError(f"predictions contain unannotated groups: {sorted(extra)}")
    return {
        "phase_recall": covered_phases / total_phases if total_phases else 0.0,
        "keeper_precision": acceptable_selected / selected_total if selected_total else 0.0,
        "retention": selected_total / member_total if member_total else 0.0,
        "risk_review_count": float(review_count),
        "risk_member_recall": risk_reviewed / risk_total if risk_total else 0.0,
        "group_impurity_count": float(impure_total),
        "group_impurity_review_recall": impure_reviewed / impure_total if impure_total else 0.0,
        "phase_undersegmentation_count": float(undersegmented_total),
        "annotated_phase_undersegmentation_count": float(sum(
            bool(validate_annotation(item, expected_events=expected_events)["phase_undersegmented"])
            for item in annotations
        )),
        "selected_count": float(selected_total),
        "member_count": float(member_total),
        "phase_count": float(total_phases),
    }


def compare(
    manifest: DatasetManifest, annotations: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = authorize(manifest, Purpose.FINAL_EVALUATION, labels_requested=True)
    events = set(manifest.events)
    base = evaluate_predictions(annotations, baseline, expected_events=events)
    cand = evaluate_predictions(annotations, candidate, expected_events=events)
    delta_keys = (
        "phase_recall", "keeper_precision", "retention", "risk_review_count",
        "risk_member_recall", "group_impurity_review_recall", "phase_undersegmentation_count",
    )
    return {
        "schema_version": 1, "contract": contract, "baseline": base, "candidate": cand,
        "delta": {key: cand[key] - base[key] for key in delta_keys},
        "writeback": "forbidden",
    }
