"""Tests for the strict, decision-free shadow routing schema."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.routing_schema import (
    QUALITY_SCOPE,
    ReasonCode,
    State,
    safe_default_record,
    validate_routing_record,
)


def _known_record():
    record = safe_default_record(reasons=[ReasonCode.MODEL_UNAVAILABLE])
    record["scene_context"] = {
        "tags": [{"code": "OUTDOOR", "score": 0.81, "support": ["p1"], "conflict": []}],
        "state": State.KNOWN, "reasons": [],
    }
    record["subject_protection"] = {
        "tags": [{"code": "REAL_PERSON_SUBJECT", "score": 0.77, "support": ["p2"]}],
        "state": State.KNOWN, "reasons": [],
    }
    return record


def test_safe_default_is_fail_closed_and_preserves_reasons():
    record = safe_default_record(reasons=[ReasonCode.MODEL_UNAVAILABLE])
    assert record["scene_context"]["state"] == "UNKNOWN"
    assert record["subject_protection"]["state"] == "UNKNOWN"
    assert record["applicability"]["global_sharpness"] == "WEAK_ONLY"
    assert record["quality_evidence"]["scope"] == QUALITY_SCOPE
    assert all("MODEL_UNAVAILABLE" in section["reasons"] for section in (
        record["scene_context"], record["subject_protection"], record["applicability"], record["quality_evidence"],
    ))
    assert validate_routing_record(record) == record
    record["scene_context"]["reasons"].append("FEATURE_MISSING")
    assert record["subject_protection"]["reasons"] == ["MODEL_UNAVAILABLE"]


def test_legal_multilabel_is_retained():
    record = _known_record()
    record["scene_context"]["tags"].extend([
        {"code": "NIGHT_LOW_LIGHT", "score": 0.65, "support": [], "conflict": []},
        {"code": "FIREWORKS", "score": 0.93, "support": [], "conflict": []},
    ])
    record["subject_protection"]["tags"].append(
        {"code": "COSTUME_MASKED_PERSON", "score": 0.55, "support": []}
    )
    normalized = validate_routing_record(record)
    assert [tag["code"] for tag in normalized["scene_context"]["tags"]] == [
        "OUTDOOR", "NIGHT_LOW_LIGHT", "FIREWORKS"
    ]


@pytest.mark.parametrize("namespace", ["scene_context", "subject_protection"])
def test_unknown_cannot_be_a_content_tag(namespace):
    record = _known_record()
    tag = {"code": "UNKNOWN", "score": 0.1, "support": []}
    if namespace == "scene_context":
        tag["conflict"] = []
    record[namespace]["tags"] = [tag]
    with pytest.raises(ValueError, match="UNKNOWN is a state"):
        validate_routing_record(record)


def test_cross_namespace_tag_is_rejected():
    record = _known_record()
    record["scene_context"]["tags"][0]["code"] = "REAL_PERSON_SUBJECT"
    with pytest.raises(ValueError, match="cross-namespace"):
        validate_routing_record(record)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_is_rejected(score):
    record = _known_record()
    record["scene_context"]["tags"][0]["score"] = score
    with pytest.raises(ValueError, match="finite"):
        validate_routing_record(record)


@pytest.mark.parametrize("field,value", [("state", "MAYBE"), ("reason", "UNLISTED_REASON")])
def test_illegal_state_and_reason_are_rejected(field, value):
    record = _known_record()
    if field == "state":
        record["subject_protection"]["state"] = value
    else:
        record["quality_evidence"]["reasons"] = [value]
    with pytest.raises(ValueError, match="illegal"):
        validate_routing_record(record)


def test_json_roundtrip_normalizes_to_json_safe_data():
    record = _known_record()
    normalized = validate_routing_record(json.loads(json.dumps(record)))
    assert json.loads(json.dumps(normalized)) == normalized


def test_schema_has_no_decision_or_action_fields():
    record = safe_default_record()
    rendered = json.dumps(record).lower()
    assert "decision" not in rendered
    assert "action" not in rendered
    assert not {"decision", "action", "auto_remove", "keep"} & set(record)


@pytest.mark.parametrize("model", [
    {"nested": {"score": float("nan")}},
    {"nested": [float("inf")]},
    {"decision": "anything"},
    {"nested": {"MANIFEST": "anything"}},
    {"object": datetime.now()},
    {1: "not a string key"},
])
def test_model_rejects_non_json_values_and_hidden_permission_keys(model):
    record = safe_default_record(model=model)
    with pytest.raises(ValueError):
        validate_routing_record(record)


def test_model_must_be_a_mapping():
    record = safe_default_record()
    record["model"] = []
    with pytest.raises(ValueError, match="model must be an object"):
        validate_routing_record(record)


@pytest.mark.parametrize("namespace", ["scene_context", "subject_protection"])
def test_namespace_state_and_tag_cardinality_are_enforced(namespace):
    known_empty = _known_record()
    known_empty[namespace]["tags"] = []
    with pytest.raises(ValueError, match="KNOWN requires"):
        validate_routing_record(known_empty)

    unknown_tagged = _known_record()
    unknown_tagged[namespace]["state"] = "UNKNOWN"
    with pytest.raises(ValueError, match="UNKNOWN requires"):
        validate_routing_record(unknown_tagged)


def test_duplicate_tag_codes_are_rejected_within_namespace():
    record = _known_record()
    record["scene_context"]["tags"].append(dict(record["scene_context"]["tags"][0]))
    with pytest.raises(ValueError, match="must not repeat"):
        validate_routing_record(record)
