"""Fail-closed, shadow-only scene-routing state machine.

This module deliberately has no image decoding, model loading, decision, or
action logic.  Stage 1 can later provide one decoded RGB image to a provider;
until then this boundary safely records metadata and provider outcomes only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .routing_schema import ReasonCode, normalize_routing_record, safe_default_record


class DecodeState(StrEnum):
    """The sole decode outcome accepted by the routing boundary."""

    OK = "ok"
    FAILED = "failed"
    PARTIAL = "partial"


RoutingProvider = Callable[["RoutingInput"], Mapping[str, Any]]


@dataclass(frozen=True)
class RoutingInput:
    """Inputs to one routing attempt, including an already-decoded RGB image.

    ``image`` is deliberately opaque here: this boundary must not open or
    decode files itself.  A future provider may consume the in-memory image
    supplied by Stage 1, preserving the decode-once fan-out invariant.
    """

    metadata: Mapping[str, Any]
    decode_state: DecodeState | str = DecodeState.OK
    image: Any = None


class SceneRouter:
    """Run metadata gates and optionally normalize a shadow provider record."""

    def __init__(self, settings: Mapping[str, Any] | Any, provider: RoutingProvider | None = None) -> None:
        scene_settings = _scene_settings(settings)
        enabled = scene_settings.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("features.scene_routing.enabled must be a bool")
        self.enabled = enabled
        self.mode = scene_settings.get("mode", "shadow")
        if not isinstance(self.mode, str):
            raise ValueError("features.scene_routing.mode must be a string")
        if self.mode != "shadow":
            raise ValueError(f"features.scene_routing.mode must be 'shadow', got {self.mode!r}")
        self.provider = provider

    def route(self, routing_input: RoutingInput) -> dict[str, Any]:
        """Return routing metadata; all missing or bad evidence abstains safely."""
        if _is_motion_photo_bound(routing_input.metadata):
            return _default(ReasonCode.MOTION_PHOTO_BOUND_ASSET, "metadata_bound_asset")

        decode_reason = _decode_reason(routing_input.decode_state)
        if decode_reason is not None:
            return _default(decode_reason, "decode_unavailable")

        if not self.enabled:
            return _default(ReasonCode.ROUTER_DISABLED, "disabled")
        if self.provider is None:
            return _default(ReasonCode.MODEL_UNAVAILABLE, "provider_missing")
        if routing_input.image is None:
            return _default(ReasonCode.FEATURE_MISSING, "image_missing")

        try:
            output = self.provider(routing_input)
            return normalize_routing_record(output)
        except ValueError:
            return _default(ReasonCode.ROUTER_CONFLICT, "invalid_provider_record")
        except Exception:  # Provider errors must never break Stage 1.
            return _default(ReasonCode.MODEL_UNAVAILABLE, "provider_error")


def route_scene(
    settings: Mapping[str, Any] | Any,
    metadata: Mapping[str, Any],
    decode_state: DecodeState | str = DecodeState.OK,
    provider: RoutingProvider | None = None,
) -> dict[str, Any]:
    """Convenience entry point for callers that do not retain a router."""
    return SceneRouter(settings, provider).route(RoutingInput(metadata, decode_state))


def _scene_settings(settings: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """Accept either a feature mapping or ``Config.features`` without imports."""
    if hasattr(settings, "scene_routing"):
        value = settings.scene_routing
        value = value.as_dict() if hasattr(value, "as_dict") else value
        if isinstance(value, Mapping):
            return value
        raise ValueError("features.scene_routing must be an object")
    if hasattr(settings, "features"):
        return _scene_settings(settings.features)
    if isinstance(settings, Mapping):
        features = settings.get("features", settings)
        if isinstance(features, Mapping):
            if "scene_routing" not in features:
                return {}
            value = features["scene_routing"]
            if isinstance(value, Mapping):
                return value
    raise ValueError("features.scene_routing must be an object")


def _is_motion_photo_bound(metadata: Mapping[str, Any]) -> bool:
    return bool(metadata.get("motion_photo") or metadata.get("is_motion_photo")) or (
        metadata.get("file_kind") in {"jpg_motion", "mp4_paired"}
    ) or metadata.get("motion_partner_id") is not None


def _decode_reason(value: DecodeState | str) -> ReasonCode | None:
    if value == DecodeState.OK:
        return None
    if value == DecodeState.PARTIAL:
        return ReasonCode.DECODE_PARTIAL
    return ReasonCode.DECODE_FAILED


def _default(reason: ReasonCode, status: str) -> dict[str, Any]:
    return safe_default_record(reasons=[reason], model={"router_status": status})
