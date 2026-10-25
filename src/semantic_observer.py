"""Normalize local CV outputs into review-only semantic observations."""
from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
MODE = "shadow_review_only"
_PRIVATE_KEY_PARTS = ("path", "filename", "basename", "directory", "folder")
_WINDOWS_PATH = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\)")
_POSIX_PATH = re.compile(r"(?:^|[\s\"'])/(?:[^/\s\"']+/)+[^/\s\"']*")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _observation(code: str, state: str, source: str, value: Any,
                 *, applicability: str = "APPLICABLE") -> dict[str, Any]:
    return {
        "code": code,
        "state": state,
        "applicability": applicability,
        "source": source,
        "value": value,
        "authority": "observation_only",
    }


def _private_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _PRIVATE_KEY_PARTS)


def _path_like(value: str) -> bool:
    return bool(_WINDOWS_PATH.search(value) or _POSIX_PATH.search(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"encoding": "base64", "bytes": len(value),
                "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite": repr(value)}
    return value


def _decode_container(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = key.lower()
    if not lowered.endswith(("_json", "_meta")):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _flatten(prefix: str, value: Any,
             redactions: list[dict[str, str]]) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value):
            name = str(key)
            child = f"{prefix}.{name}" if prefix else name
            if _private_key(name):
                redactions.append({"field": child, "reason": "privacy_key"})
                continue
            rows.extend(_flatten(child, _decode_container(name, value[key]), redactions))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(f"{prefix}[{index}]", item, redactions))
        return rows
    if isinstance(value, str) and _path_like(value):
        redactions.append({"field": prefix, "reason": "path_value"})
        return []
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return [(prefix, _json_safe(value))]
    return [(prefix, {"python_type": type(value).__name__, "repr": repr(value)})]


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, Mapping) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def persisted_group_cv_context(
    members: Sequence[Mapping[str, Any]],
    *,
    group: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export every supplied persisted leaf except path/privacy-bearing values.

    Binary embeddings are losslessly base64 encoded; JSON text columns are
    expanded; weak, default-off, missing, and shadow-only fields are retained.
    """
    frames = []
    refs = []
    redactions: list[dict[str, str]] = []
    group_measurements = []
    if group is not None:
        for path, value in _flatten("", dict(group), redactions):
            ref = f"GROUP:{path}"
            group_measurements.append({"ref": ref, "value": value})
            refs.append(ref)
    for index, member in enumerate(members, start=1):
        alias = str(member.get("frame_alias") or f"F{index:02d}")
        measurements = []
        supplied = dict(member)
        supplied.pop("frame_alias", None)
        for path, value in _flatten("", supplied, redactions):
            ref = f"{alias}:{path}"
            measurements.append({"ref": ref, "value": value})
            refs.append(ref)
        frames.append({"frame_alias": alias, "measurements": measurements})
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "group_measurements": group_measurements,
        "frames": frames,
        "evidence_refs": refs,
        "redactions": redactions,
        "policy": {
            "drop_weak_signals": False,
            "drop_default_off_signals": False,
            "drop_shadow_signals": False,
            "redacts_filesystem_paths": True,
            "binary_encoding": "lossless_base64",
            "authority": "measurement_context_not_ground_truth",
        },
    }


def group_cv_context(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Package every supplied CV signal for group-level VLM synthesis.

    No signal is dropped merely because it is shadow-only or weak in isolation.
    Authority and applicability travel with the value so the VLM can combine
    weak signals without silently upgrading them into measured truth.
    """
    packed: list[dict[str, Any]] = []
    refs: list[str] = []
    for frame in frames:
        alias = str(frame.get("frame_alias") or "")
        if not alias:
            continue
        observations = []
        for raw in frame.get("observations") or []:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("code"), str):
                continue
            item = dict(raw)
            ref = f"{alias}:{item['code']}"
            item["ref"] = ref
            observations.append(item)
            refs.append(ref)
        packed.append({
            "frame_alias": alias,
            "observations": observations,
            "abstentions": list(frame.get("abstentions") or []),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "frames": packed,
        "evidence_refs": refs,
        "instruction": (
            "Consider all supplied measurements jointly, including weak and "
            "shadow-only signals; cite exact refs and do not upgrade their authority."
        ),
    }


def frame_observations(*, frame_alias: str, image_size: tuple[int, int],
                       eye_result: Mapping[str, Any],
                       yunet_faces: Sequence[Mapping[str, Any]],
                       face_quality: float | None, sharpness: float | None,
                       exposure_metrics: Mapping[str, Any],
                       exposure_class: Sequence[Any]) -> dict[str, Any]:
    """Return source-linked CV evidence without selecting or rejecting a frame."""
    width, height = image_size
    observations: list[dict[str, Any]] = []
    abstentions: list[str] = []

    eye_status = str(eye_result.get("status") or "UNKNOWN")
    observations.append(_observation(
        f"EYE_{eye_status}", eye_status, "mediapipe.face_landmarker.blendshapes",
        {
            "detected_face_count": eye_result.get("detected_face_count"),
            "detector_count_disagreement": bool(
                eye_result.get("detector_count_disagreement")
            ),
            "reasons": list(eye_result.get("reasons") or []),
            "face_states": [
                {
                    key: face.get(key) for key in (
                        "face_index", "state", "reasons",
                    )
                }
                for face in eye_result.get("faces") or []
            ],
        },
        applicability="UNKNOWN" if eye_status == "UNKNOWN" else "APPLICABLE",
    ))
    if eye_status in {"UNKNOWN", "HAS_MAYBE"}:
        abstentions.append("EYE_STATE_NOT_DECISIVE")

    quality = _finite(face_quality)
    observations.append(_observation(
        "FACE_LOCAL_QUALITY", "MEASURED" if quality is not None else "UNKNOWN",
        "quality.compute_face_quality", quality,
        applicability="APPLICABLE" if yunet_faces and quality is not None else "UNKNOWN",
    ))
    sharp = _finite(sharpness)
    observations.append(_observation(
        "GLOBAL_SHARPNESS", "MEASURED" if sharp is not None else "UNKNOWN",
        "quality.variance_of_laplacian", sharp,
        applicability="APPLICABLE" if sharp is not None else "UNKNOWN",
    ))

    exposure_state = str(exposure_class[0]) if exposure_class else "unknown"
    observations.append(_observation(
        "EXPOSURE_STATE", exposure_state.upper(), "exposure.classify",
        {
            "reason": exposure_class[1] if len(exposure_class) > 1 else "",
            "severity": exposure_class[2] if len(exposure_class) > 2 else None,
            "clip_hi": exposure_metrics.get("clip_hi"),
            "clip_lo": exposure_metrics.get("clip_lo"),
            "anchor_mass": exposure_metrics.get("anchor_mass"),
        },
    ))

    edge_faces = []
    for index, face in enumerate(yunet_faces):
        box = face.get("bbox") or []
        if len(box) < 4 or width <= 0 or height <= 0:
            continue
        x, y, box_width, box_height = map(float, box[:4])
        margins = {
            "left": x / width,
            "top": y / height,
            "right": (width - (x + box_width)) / width,
            "bottom": (height - (y + box_height)) / height,
        }
        if min(margins.values()) < 0.01:
            edge_faces.append({"face_index": index, "normalized_margins": margins})
    observations.append(_observation(
        "FACE_FRAME_EDGE_PROXIMITY",
        "NEAR_EDGE" if edge_faces else "NOT_OBSERVED",
        "yunet.face_bbox_vs_image_bounds",
        {"faces": edge_faces, "threshold": 0.01},
        applicability="APPLICABLE" if yunet_faces else "UNKNOWN",
    ))
    if edge_faces:
        abstentions.append("EDGE_PROXIMITY_IS_NOT_CROP_PROOF")

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "frame_alias": frame_alias,
        "observations": observations,
        "abstentions": sorted(set(abstentions)),
        "invariants": {
            "changes_keeper": False,
            "changes_sorting": False,
            "changes_routing": False,
            "deletion_authority": "none",
        },
    }
