"""Strict, dependency-free JSON contract for shadow scene-routing metadata.

This module only validates and normalizes routing evidence.  It intentionally
contains no decision or action vocabulary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = 1
_FORBIDDEN_MODEL_KEYS = frozenset({"decision", "action", "auto_remove", "keep", "delete", "manifest"})


class State(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


class Applicability(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    WEAK_ONLY = "WEAK_ONLY"
    SCENE_SPECIFIC = "SCENE_SPECIFIC"
    UNKNOWN = "UNKNOWN"


class ReasonCode(StrEnum):
    ROUTER_DISABLED = "ROUTER_DISABLED"
    MOTION_PHOTO_BOUND_ASSET = "MOTION_PHOTO_BOUND_ASSET"
    DECODE_FAILED = "DECODE_FAILED"
    DECODE_PARTIAL = "DECODE_PARTIAL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    FEATURE_MISSING = "FEATURE_MISSING"
    LOW_ABSOLUTE_SCORE = "LOW_ABSOLUTE_SCORE"
    LOW_EXCLUSIVE_MARGIN = "LOW_EXCLUSIVE_MARGIN"
    HIGH_EXCLUSIVE_ENTROPY = "HIGH_EXCLUSIVE_ENTROPY"
    PROMPT_INCONSISTENT = "PROMPT_INCONSISTENT"
    ROUTER_CONFLICT = "ROUTER_CONFLICT"
    OUT_OF_CALIBRATION_DOMAIN = "OUT_OF_CALIBRATION_DOMAIN"
    SUBJECT_NOT_OWNED = "SUBJECT_NOT_OWNED"
    BACKGROUND_FACE_ONLY = "BACKGROUND_FACE_ONLY"
    NONHUMAN_FACE_RISK = "NONHUMAN_FACE_RISK"
    PRINTED_OR_SCREEN_FACE_RISK = "PRINTED_OR_SCREEN_FACE_RISK"
    MULTIPLE_SUBJECTS = "MULTIPLE_SUBJECTS"
    SUBJECT_ASSIGNMENT_UNKNOWN = "SUBJECT_ASSIGNMENT_UNKNOWN"
    DETECTOR_NOT_APPLICABLE = "DETECTOR_NOT_APPLICABLE"
    FACE_COUNT_MISMATCH = "FACE_COUNT_MISMATCH"
    FACE_BOX_MISMATCH = "FACE_BOX_MISMATCH"
    CROP_CONTEXT_CONFLICT = "CROP_CONTEXT_CONFLICT"
    MAX_FACES_REACHED = "MAX_FACES_REACHED"
    SMALL_FACE = "SMALL_FACE"
    PROFILE_OR_OCCLUDED = "PROFILE_OR_OCCLUDED"
    EYE_STATE_UNKNOWN = "EYE_STATE_UNKNOWN"
    BYTE_IDENTICAL = "BYTE_IDENTICAL"
    GROUP_IMPURE = "GROUP_IMPURE"
    LOW_MARGIN = "LOW_MARGIN"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    RELATIVE_ONLY = "RELATIVE_ONLY"
    SUBJECTIVE_ONLY = "SUBJECTIVE_ONLY"
    CROSS_ROUTING_GROUP = "CROSS_ROUTING_GROUP"
    SCENE_CALIBRATION_MISSING = "SCENE_CALIBRATION_MISSING"


SCENE_TAGS = frozenset({"INDOOR", "OUTDOOR", "NIGHT_LOW_LIGHT", "FIREWORKS", "LANDSCAPE_CITYSCAPE"})
SUBJECT_TAGS = frozenset({
    "REAL_PERSON_SUBJECT", "COSTUME_MASKED_PERSON", "NONHUMAN_CHARACTER_DOLL_STATUE",
    "FOOD_STILL_LIFE", "DOCUMENT_SCREENSHOT", "NO_DOMINANT_SUBJECT",
})
STATES = frozenset(state.value for state in State)
REASON_CODES = frozenset(reason.value for reason in ReasonCode)
EYE_QUALITY_VALUES = frozenset({Applicability.APPLICABLE.value, Applicability.NOT_APPLICABLE.value, Applicability.UNKNOWN.value})
FACE_QUALITY_VALUES = EYE_QUALITY_VALUES
GLOBAL_SHARPNESS_VALUES = frozenset({Applicability.APPLICABLE.value, Applicability.WEAK_ONLY.value, Applicability.UNKNOWN.value})
EXPOSURE_VALUES = frozenset({Applicability.APPLICABLE.value, Applicability.SCENE_SPECIFIC.value, Applicability.UNKNOWN.value})
QUALITY_SCOPE = "WITHIN_TRUSTED_GROUP_ONLY"

def safe_default_record(*, reasons: Sequence[str | ReasonCode] = (), model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a fail-closed record for disabled or unavailable routing models."""
    reason_list = _reasons(reasons, "default reasons")
    return {
        "schema_version": SCHEMA_VERSION,
        "model": dict(model or {}),
        "scene_context": {"tags": [], "state": State.UNKNOWN.value, "reasons": list(reason_list)},
        "subject_protection": {"tags": [], "state": State.UNKNOWN.value, "reasons": list(reason_list)},
        "applicability": {
            "eye_quality": Applicability.UNKNOWN.value,
            "face_quality": Applicability.UNKNOWN.value,
            "global_sharpness": Applicability.WEAK_ONLY.value,
            "exposure": Applicability.UNKNOWN.value,
            "reasons": list(reason_list),
        },
        "quality_evidence": {
            "scope": QUALITY_SCOPE,
            "signals": [],
            "missing": [],
            "conflicts": [],
            "reasons": list(reason_list),
        },
    }

def validate_routing_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and JSON-normalize a routing record, raising clear ``ValueError``s."""
    _mapping(record, "routing record")
    _keys(record, {"schema_version", "model", "scene_context", "subject_protection", "applicability", "quality_evidence"}, "routing record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    model = _model(record["model"])
    scene = _namespace(record["scene_context"], "scene_context", SCENE_TAGS, conflict=True)
    subject = _namespace(record["subject_protection"], "subject_protection", SUBJECT_TAGS, conflict=False)
    applicability = _applicability(record["applicability"])
    quality = _quality_evidence(record["quality_evidence"])
    return {"schema_version": SCHEMA_VERSION, "model": model, "scene_context": scene,
            "subject_protection": subject, "applicability": applicability, "quality_evidence": quality}


normalize_routing_record = validate_routing_record

def _namespace(value: Any, name: str, allowed_tags: frozenset[str], *, conflict: bool) -> dict[str, Any]:
    data = _mapping(value, name)
    expected = {"tags", "state", "reasons"}
    _keys(data, expected, name)
    state = _state(data["state"], f"{name}.state")
    tags = [_tag(tag, name, allowed_tags, conflict) for tag in _list(data["tags"], f"{name}.tags")]
    if state == State.KNOWN and not tags:
        raise ValueError(f"{name}.state KNOWN requires at least one tag")
    if state == State.UNKNOWN and tags:
        raise ValueError(f"{name}.state UNKNOWN requires zero tags")
    codes = [tag["code"] for tag in tags]
    if len(codes) != len(set(codes)):
        raise ValueError(f"{name}.tags must not repeat tag codes")
    return {"tags": tags, "state": state, "reasons": _reasons(data["reasons"], f"{name}.reasons")}

def _tag(value: Any, namespace: str, allowed: frozenset[str], conflict: bool) -> dict[str, Any]:
    data = _mapping(value, f"{namespace} tag")
    expected = {"code", "score", "support"} | ({"conflict"} if conflict else set())
    _keys(data, expected, f"{namespace} tag")
    code = data["code"]
    if code == State.UNKNOWN.value:
        raise ValueError(f"{namespace} tag code UNKNOWN is invalid: UNKNOWN is a state, not a content tag")
    if code not in allowed:
        raise ValueError(f"unknown or cross-namespace tag {code!r} in {namespace}")
    score = data["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f"{namespace} tag score must be finite")
    output = {"code": code, "score": float(score), "support": _strings(data["support"], f"{namespace} tag support")}
    if conflict:
        output["conflict"] = _strings(data["conflict"], f"{namespace} tag conflict")
    return output


def _applicability(value: Any) -> dict[str, Any]:
    data = _mapping(value, "applicability")
    _keys(data, {"eye_quality", "face_quality", "global_sharpness", "exposure", "reasons"}, "applicability")
    allowed = {"eye_quality": EYE_QUALITY_VALUES, "face_quality": FACE_QUALITY_VALUES,
               "global_sharpness": GLOBAL_SHARPNESS_VALUES, "exposure": EXPOSURE_VALUES}
    output = {key: _choice(data[key], values, f"applicability.{key}") for key, values in allowed.items()}
    output["reasons"] = _reasons(data["reasons"], "applicability.reasons")
    return output


def _quality_evidence(value: Any) -> dict[str, Any]:
    data = _mapping(value, "quality_evidence")
    _keys(data, {"scope", "signals", "missing", "conflicts", "reasons"}, "quality_evidence")
    if data["scope"] != QUALITY_SCOPE:
        raise ValueError(f"quality_evidence.scope must be {QUALITY_SCOPE}")
    return {"scope": QUALITY_SCOPE, "signals": _strings(data["signals"], "quality_evidence.signals"),
            "missing": _strings(data["missing"], "quality_evidence.missing"),
            "conflicts": _strings(data["conflicts"], "quality_evidence.conflicts"),
            "reasons": _reasons(data["reasons"], "quality_evidence.reasons")}


def _state(value: Any, label: str) -> str:
    return _choice(value, STATES, label)


def _reasons(value: Any, label: str) -> list[str]:
    reasons = _strings(value, label)
    illegal = set(reasons) - REASON_CODES
    if illegal:
        raise ValueError(f"{label} contains illegal reason code(s): {sorted(illegal)!r}")
    return reasons


def _choice(value: Any, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"{label} has illegal value {value!r}")
    return str(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _model(value: Any) -> dict[str, Any]:
    """Normalize model audit data to the small, finite JSON subset we persist."""
    if not isinstance(value, Mapping):
        raise ValueError("model must be an object")
    return _json_value(value, "model")


def _json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} object keys must be strings")
            if key.casefold() in _FORBIDDEN_MODEL_KEYS:
                raise ValueError(f"{label} contains forbidden model key {key!r}")
            output[key] = _json_value(item, f"{label}.{key}")
        return output
    if isinstance(value, list):
        return [_json_value(item, f"{label}[]") for item in value]
    raise ValueError(f"{label} must contain only JSON object/list/string/bool/null/finite number values")


def _keys(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(data)
    if actual != expected:
        raise ValueError(f"{label} has invalid keys; missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}")


def _list(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _strings(value: Any, label: str) -> list[str]:
    items = _list(value, label)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{label} must contain only strings")
    return items
