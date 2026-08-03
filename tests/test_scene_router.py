"""Tests for the fail-closed, decision-free shadow router skeleton."""

from __future__ import annotations

import json

import pytest

from src.routing_schema import ReasonCode, safe_default_record
from src.scene_router import DecodeState, RoutingInput, SceneRouter, route_scene
from src.config import load_config


SHADOW = {"scene_routing": {"enabled": True, "mode": "shadow"}}
DISABLED = {"scene_routing": {"enabled": False, "mode": "shadow"}}


def _reasons(record):
    return record["scene_context"]["reasons"]


def _known_record():
    record = safe_default_record(model={"name": "test"})
    record["scene_context"] = {
        "tags": [
            {"code": "OUTDOOR", "score": 0.8, "support": ["p1"], "conflict": []},
            {"code": "FIREWORKS", "score": 0.9, "support": ["p2"], "conflict": []},
        ],
        "state": "KNOWN", "reasons": [],
    }
    record["subject_protection"] = {
        "tags": [{"code": "REAL_PERSON_SUBJECT", "score": 0.7, "support": ["p3"]}],
        "state": "KNOWN", "reasons": [],
    }
    return record


def test_disabled_returns_distinct_safe_default_without_calling_provider():
    def provider(_input):
        raise AssertionError("disabled router must not invoke provider")

    record = route_scene(DISABLED, {}, provider=provider)
    assert _reasons(record) == [ReasonCode.ROUTER_DISABLED]
    assert record["model"]["router_status"] == "disabled"


def test_default_config_disables_shadow_router():
    record = route_scene(load_config(), {})
    assert _reasons(record) == [ReasonCode.ROUTER_DISABLED]


def test_motion_photo_metadata_precedes_disabled_and_keeps_visual_states_unknown():
    record = route_scene(DISABLED, {"file_kind": "jpg_motion"})
    assert _reasons(record) == [ReasonCode.MOTION_PHOTO_BOUND_ASSET]
    assert record["scene_context"]["state"] == "UNKNOWN"
    assert record["subject_protection"]["state"] == "UNKNOWN"


@pytest.mark.parametrize(("state", "reason"), [
    (DecodeState.FAILED, ReasonCode.DECODE_FAILED),
    (DecodeState.PARTIAL, ReasonCode.DECODE_PARTIAL),
    ("unrecognized", ReasonCode.DECODE_FAILED),
])
def test_decode_failures_fail_closed(state, reason):
    record = route_scene(SHADOW, {}, decode_state=state)
    assert _reasons(record) == [reason]


def test_missing_exceptional_and_invalid_providers_are_contained():
    missing = route_scene(SHADOW, {})
    assert _reasons(missing) == [ReasonCode.MODEL_UNAVAILABLE]

    def broken(_input):
        raise RuntimeError("boom")

    failed = SceneRouter(SHADOW, provider=broken).route(RoutingInput(metadata={}, image=object()))
    assert _reasons(failed) == [ReasonCode.MODEL_UNAVAILABLE]
    assert failed["model"]["router_status"] == "provider_error"

    invalid = SceneRouter(SHADOW, provider=lambda _input: {"bad": "record"}).route(
        RoutingInput(metadata={}, image=object())
    )
    assert _reasons(invalid) == [ReasonCode.ROUTER_CONFLICT]
    assert invalid["model"]["router_status"] == "invalid_provider_record"

    non_json = SceneRouter(SHADOW, provider=lambda _input: safe_default_record(model={"bad": object()})).route(
        RoutingInput(metadata={}, image=object())
    )
    assert _reasons(non_json) == [ReasonCode.ROUTER_CONFLICT]

    non_finite = SceneRouter(SHADOW, provider=lambda _input: safe_default_record(model={"bad": float("nan")})).route(
        RoutingInput(metadata={}, image=object())
    )
    assert _reasons(non_finite) == [ReasonCode.ROUTER_CONFLICT]

    hidden_permission = SceneRouter(SHADOW, provider=lambda _input: safe_default_record(model={"nested": {"delete": True}})).route(
        RoutingInput(metadata={}, image=object())
    )
    assert _reasons(hidden_permission) == [ReasonCode.ROUTER_CONFLICT]


def test_legal_multilabel_provider_output_is_normalized():
    router = SceneRouter(SHADOW, provider=lambda _input: _known_record())
    record = router.route(RoutingInput(metadata={}, decode_state="ok", image=object()))
    assert [tag["code"] for tag in record["scene_context"]["tags"]] == ["OUTDOOR", "FIREWORKS"]
    assert record["subject_protection"]["tags"][0]["code"] == "REAL_PERSON_SUBJECT"


@pytest.mark.parametrize("mode", ["active", "decision", "AUTO_REMOVE"])
def test_non_shadow_modes_are_loudly_rejected(mode):
    with pytest.raises(ValueError, match="mode"):
        SceneRouter({"scene_routing": {"enabled": True, "mode": mode}})


@pytest.mark.parametrize("settings", [
    {"scene_routing": {"enabled": "false", "mode": "shadow"}},
    {"scene_routing": None},
    {"scene_routing": []},
])
def test_invalid_scene_routing_settings_are_loudly_rejected(settings):
    with pytest.raises(ValueError):
        SceneRouter(settings)


def test_enabled_provider_without_image_abstains_without_calling_provider():
    called = False

    def provider(_input):
        nonlocal called
        called = True
        return _known_record()

    record = SceneRouter(SHADOW, provider=provider).route(RoutingInput(metadata={}))
    assert _reasons(record) == [ReasonCode.FEATURE_MISSING]
    assert record["model"]["router_status"] == "image_missing"
    assert not called


def test_router_record_never_contains_decision_or_action():
    record = SceneRouter(SHADOW, provider=lambda _input: _known_record()).route(RoutingInput(metadata={}, image=object()))
    rendered = json.dumps(record).lower()
    assert "decision" not in rendered
    assert "action" not in rendered
    assert not {"decision", "action", "auto_remove", "keep"} & set(record)
