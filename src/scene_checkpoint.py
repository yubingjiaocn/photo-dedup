"""Strict checkpoint validation for the offline scene calibration report."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

CHECKPOINT_SCHEMA_VERSION = 2
NAMESPACES = ("scene_context", "subject_protection")
TOP_KEYS = {
    "checkpoint_schema_version", "input_identity", "config_sha256", "next_index",
    "seen", "valid", "invalid", "unknown", "reasons", "unknown_reasons",
    "conflict_records", "missing_records", "tag_present", "stats", "identities",
    "model_revisions", "identity_errors", "bad_records", "prefix_digest",
}


def new_state(identity: Mapping[str, Any], config_hash: str | None) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "input_identity": dict(identity), "config_sha256": config_hash,
        "next_index": 0, "seen": 0, "valid": 0, "invalid": 0,
        "unknown": {name: 0 for name in NAMESPACES}, "reasons": Counter(),
        "unknown_reasons": {name: Counter() for name in NAMESPACES},
        "conflict_records": 0, "missing_records": 0, "tag_present": Counter(),
        "stats": {}, "identities": Counter(), "model_revisions": Counter(),
        "identity_errors": 0, "bad_records": [], "prefix_digest": "0" * 64,
    }


def restore_checkpoint(raw: Any, identity: Mapping[str, Any], config_hash: str | None) -> dict[str, Any]:
    """Accept only a complete, internally consistent checkpoint state."""
    if not isinstance(raw, dict) or set(raw) != TOP_KEYS:
        raise ValueError("checkpoint keys/schema are invalid")
    if raw["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if raw["input_identity"] != dict(identity) or raw["config_sha256"] != config_hash:
        raise ValueError("checkpoint input/config mismatch")
    state = dict(raw)
    for key in ("next_index", "seen", "valid", "invalid", "conflict_records", "missing_records", "identity_errors"):
        state[key] = _count(raw[key], key)
    if state["next_index"] != state["seen"] or state["seen"] != state["valid"] + state["invalid"]:
        raise ValueError("checkpoint index/count invariants are invalid")
    if state["conflict_records"] > state["valid"] or state["missing_records"] > state["valid"]:
        raise ValueError("checkpoint quality counters exceed valid records")
    if state["identity_errors"] > state["invalid"]:
        raise ValueError("checkpoint identity errors exceed invalid records")
    state["unknown"] = _namespace_counts(raw["unknown"], state["valid"], "unknown")
    state["reasons"] = _counter(raw["reasons"], "reasons")
    state["tag_present"] = _counter(raw["tag_present"], "tag_present")
    state["identities"] = _counter(raw["identities"], "identities")
    state["model_revisions"] = _counter(raw["model_revisions"], "model_revisions")
    if sum(state["identities"].values()) != state["valid"]:
        raise ValueError("checkpoint identity counts do not equal valid records")
    if sum(state["model_revisions"].values()) != state["valid"]:
        raise ValueError("checkpoint revision counts do not equal valid records")
    unknown_reasons = raw["unknown_reasons"]
    if not isinstance(unknown_reasons, dict) or set(unknown_reasons) != set(NAMESPACES):
        raise ValueError("checkpoint unknown_reasons keys are invalid")
    state["unknown_reasons"] = {name: _counter(unknown_reasons[name], f"unknown_reasons.{name}") for name in NAMESPACES}
    state["stats"] = _stats(raw["stats"], state["valid"])
    state["bad_records"] = _bad_records(raw["bad_records"], state["invalid"])
    digest = raw["prefix_digest"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("checkpoint prefix digest is invalid")
    return state


def json_safe(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json_safe(left) == json_safe(right)


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint {label} must be a nonnegative integer")
    return value


def _counter(value: Any, label: str) -> Counter[str]:
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
        raise ValueError(f"checkpoint {label} must be a string-keyed object")
    return Counter({key: _count(item, f"{label}.{key}") for key, item in value.items()})


def _namespace_counts(value: Any, maximum: int, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(NAMESPACES):
        raise ValueError(f"checkpoint {label} keys are invalid")
    result = {key: _count(item, f"{label}.{key}") for key, item in value.items()}
    if any(item > maximum for item in result.values()):
        raise ValueError(f"checkpoint {label} exceeds valid records")
    return result


def _stats(value: Any, valid: int) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("checkpoint stats must be an object")
    output = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, dict) or set(item) != {"count", "sum", "min", "max"}:
            raise ValueError("checkpoint stat shape is invalid")
        count = _count(item["count"], f"stats.{key}.count")
        if count <= 0 or count != valid:
            raise ValueError("checkpoint stat count is out of range")
        total, minimum, maximum = (_number(item[name], f"stats.{key}.{name}") for name in ("sum", "min", "max"))
        if minimum > maximum or total < count * minimum or total > count * maximum:
            raise ValueError("checkpoint stat sum/min/max invariants are invalid")
        output[key] = {"count": count, "sum": total, "min": minimum, "max": maximum}
    return output


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"checkpoint {label} must be finite")
    return float(value)


def _bad_records(value: Any, invalid: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > min(20, invalid):
        raise ValueError("checkpoint bad_records count is invalid")
    output = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"record_id", "error"}:
            raise ValueError("checkpoint bad record shape is invalid")
        if any(not isinstance(item[key], str) for key in item):
            raise ValueError("checkpoint bad record fields must be strings")
        output.append(dict(item))
    return output
