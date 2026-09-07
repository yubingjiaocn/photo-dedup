"""Constrained semantic event policy with monotonic-safe fallbacks.

The model-facing contract contains intent and protection hints only. A
deterministic compiler maps those hints to a finite set of conservative
presets; it cannot authorize deletion or arbitrary numeric thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    GENERAL = "general"
    STAGE = "stage"
    CONVENTION = "convention"
    PET = "pet"
    DESIGN_REFERENCE = "design_reference"


class Capability(StrEnum):
    FULL = "full"
    NO_VLM = "no_vlm"
    NO_LLM = "no_llm"
    OFFLINE = "offline"


_ALLOWED_KEYS = {"schema_version", "event_kind", "confidence", "valuable_states", "mixed_theme", "identity_sensitive"}
_ALLOWED_STATES = {"action_phase", "different_subject", "multi_angle", "detail", "singleton", "rare_state", "live_photo_moment"}
_PRESETS: dict[EventKind, dict[str, Any]] = {
    EventKind.GENERAL: {"max_visual_group": 8, "keepers_per_phase": 1, "require_identity": True},
    EventKind.STAGE: {"max_visual_group": 5, "keepers_per_phase": 2, "require_identity": False},
    EventKind.CONVENTION: {"max_visual_group": 8, "keepers_per_phase": 1, "require_identity": True},
    EventKind.PET: {"max_visual_group": 6, "keepers_per_phase": 2, "require_identity": False},
    EventKind.DESIGN_REFERENCE: {"max_visual_group": 4, "keepers_per_phase": 2, "require_identity": False},
}


def validate_event_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ALLOWED_KEYS:
        raise ValueError("event policy has invalid keys")
    if value["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    try:
        kind = EventKind(value["event_kind"])
    except (ValueError, TypeError) as exc:
        raise ValueError("unknown event_kind") from exc
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    states = value["valuable_states"]
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise ValueError("valuable_states must be a list")
    if not all(isinstance(state, str) and state in _ALLOWED_STATES for state in states):
        raise ValueError("valuable_states contains an unknown state")
    if not isinstance(value["mixed_theme"], bool) or not isinstance(value["identity_sensitive"], bool):
        raise ValueError("policy flags must be bools")
    return {"schema_version": 1, "event_kind": kind.value, "confidence": float(confidence),
            "valuable_states": list(dict.fromkeys(states)), "mixed_theme": value["mixed_theme"],
            "identity_sensitive": value["identity_sensitive"]}


def compile_event_policy(value: Mapping[str, Any] | None, capability: Capability | str = Capability.FULL) -> dict[str, Any]:
    """Compile policy to bounded knobs; weaker evidence only gets safer."""
    try:
        capability = Capability(capability)
    except (ValueError, TypeError) as exc:
        raise ValueError("unknown capability") from exc
    try:
        policy = validate_event_policy(value or {})
    except ValueError:
        policy = {"event_kind": EventKind.GENERAL.value, "confidence": 0.0, "valuable_states": [],
                  "mixed_theme": True, "identity_sensitive": True}
    kind = EventKind(policy["event_kind"])
    preset = dict(_PRESETS[kind])
    weak = capability != Capability.FULL or policy["confidence"] < 0.65 or policy["mixed_theme"]
    if weak:
        preset["max_visual_group"] = min(preset["max_visual_group"], 4)
        preset["keepers_per_phase"] = max(preset["keepers_per_phase"], 2)
        preset["require_identity"] = True
    if policy["identity_sensitive"]:
        preset["require_identity"] = True
    return {"preset": kind.value, "capability": capability.value, **preset,
            "protect_singletons": True, "protect_rare_states": True,
            "large_group_action": "split_or_multi_keeper", "auto_remove_scope": "byte_identical_only",
            "review_required": True}
