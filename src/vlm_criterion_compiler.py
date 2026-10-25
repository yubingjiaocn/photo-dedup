"""Compile lossless persisted CV records into bounded model-usable criteria.

The audit exporter remains the source of truth.  This module deliberately keeps
compact scalar evidence exact while replacing storage-oriented collections with
typed summaries and an explicit source-family coverage ledger.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from . import pose_evidence

SCHEMA_VERSION = 2
LEGACY_CONTEXT_GUARD_BYTES = 131_072
MAX_NUMERIC_ARRAY = 16
REGION_TOP_K = 3
RECORD_TOP_K = 4
MAX_EXACT_FAMILY_BYTES = 1_800
MAX_EXACT_LEAVES = 32
MAX_GENERIC_STATS = 12
MAX_GENERIC_CATEGORIES = 8

_PRIVATE_KEY_PARTS = ("path", "filename", "basename", "directory", "folder")
_WINDOWS_PATH = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\)")
_POSIX_PATH = re.compile(r"(?:^|[\s\"'])/(?:[^/\s\"']+/)+[^/\s\"']*")
_INDEX = re.compile(r"\[\d+\]")
_STATE_KEYS = {
    "status", "state", "reason", "applicability", "authority", "eligible",
    "enabled", "review_only", "keeper_authority", "semantic_phase_authority",
    "semantic_scene_regime_authority", "semantic_subject_set_authority",
    "amodal_completeness", "eye_state", "catalog_semantics",
}
_STATIC_KEYS = {
    "schema_version", "method", "producer", "backend", "policy", "pose_format",
    "authority", "catalog_semantics",
}
_SPECIAL_SUMMARIES = {
    "quality_meta.local_region_set": "region_set",
    "quality_meta.local_quality": "region_set",
    "quality_meta.pose_evidence": "pose",
    "faces_json": "faces",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _stats(values: Sequence[Any]) -> dict[str, Any]:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    if not numbers:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
    }


def _delta(left: Any, right: Any) -> float | None:
    a, b = _finite(left), _finite(right)
    return b - a if a is not None and b is not None else None


def _private_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _PRIVATE_KEY_PARTS)


def _path_like(value: str) -> bool:
    return bool(_WINDOWS_PATH.search(value) or _POSIX_PATH.search(value))


def _decoded(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"python_type": type(value).__name__}


def _flatten(
    prefix: str,
    value: Any,
    exclusions: list[dict[str, str]],
) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for raw_key in sorted(value, key=str):
            key = str(raw_key)
            child = f"{prefix}.{key}" if prefix else key
            if _private_key(key):
                exclusions.append({"source": child, "reason": "privacy_key"})
                continue
            rows.extend(_flatten(child, value[raw_key], exclusions))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_flatten(f"{prefix}[{index}]", child, exclusions))
        return rows
    if isinstance(value, (bytes, bytearray, memoryview)):
        exclusions.append({"source": prefix, "reason": "raw_binary_storage"})
        return []
    if isinstance(value, str) and _path_like(value):
        exclusions.append({"source": prefix, "reason": "path_value"})
        return []
    return [(prefix, _safe_scalar(value))]


def _sequence_lengths(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_sequence_lengths(child, path))
    elif isinstance(value, (list, tuple)):
        result.append({
            "source": prefix,
            "count": len(value),
            "item_types": sorted({type(item).__name__ for item in value}),
        })
        for index, child in enumerate(value[:MAX_NUMERIC_ARRAY]):
            result.extend(_sequence_lengths(child, f"{prefix}[{index}]"))
    return result


def _needs_summary(value: Any) -> bool:
    try:
        size = len(_canonical(value))
    except (TypeError, ValueError):
        return True
    exclusions: list[dict[str, str]] = []
    leaves = _flatten("", value, exclusions)
    return (
        size > MAX_EXACT_FAMILY_BYTES
        or len(leaves) > MAX_EXACT_LEAVES
        or any(item["count"] > MAX_NUMERIC_ARRAY for item in _sequence_lengths(value))
    )


def _box_geometry(
    record: Mapping[str, Any],
    *,
    width: float | None,
    height: float | None,
    xywh: bool = False,
) -> dict[str, float] | None:
    box = record.get("bbox" if xywh else "box_normalized")
    normalized = not xywh
    if box is None:
        box = record.get("box")
        normalized = bool(
            isinstance(box, (list, tuple))
            and len(box) >= 4
            and all(_finite(value) is not None and 0 <= float(value) <= 1 for value in box[:4])
        )
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    values = [_finite(value) for value in box[:4]]
    if any(value is None for value in values):
        return None
    a, b, c, d = (float(value) for value in values)
    if xywh:
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        x1, y1, x2, y2 = a, b, c, d
    if not normalized:
        if not width or not height:
            return None
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    margins = (x1, y1, 1 - x2, 1 - y2)
    return {
        "area_fraction": (x2 - x1) * (y2 - y1),
        "minimum_edge_margin": min(margins),
        "touches_left": float(x1 <= 0.02),
        "touches_top": float(y1 <= 0.02),
        "touches_right": float(1 - x2 <= 0.02),
        "touches_bottom": float(1 - y2 <= 0.02),
    }


def _edge_summary(geometries: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    return {
        "near_any_edge_count": sum(
            geometry["minimum_edge_margin"] <= 0.02 for geometry in geometries
        ),
        "touch_left_count": int(sum(g["touches_left"] for g in geometries)),
        "touch_top_count": int(sum(g["touches_top"] for g in geometries)),
        "touch_right_count": int(sum(g["touches_right"] for g in geometries)),
        "touch_bottom_count": int(sum(g["touches_bottom"] for g in geometries)),
        "minimum_edge_margin": _stats(
            geometry["minimum_edge_margin"] for geometry in geometries
        ),
    }


def _region_collection(
    records: Any,
    *,
    width: float | None,
    height: float | None,
    catalog: bool,
) -> dict[str, Any]:
    rows = [row for row in records if isinstance(row, Mapping)] \
        if isinstance(records, (list, tuple)) else []
    geometries = [
        geometry for row in rows
        if (geometry := _box_geometry(row, width=width, height=height)) is not None
    ]
    classes = Counter(str(row.get("class_id", "unknown")) for row in rows)
    kinds = Counter(str(row.get("kind", "detection" if catalog else "unknown")) for row in rows)

    def top_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        musiq = _finite(row.get("musiq"))
        clipiqa = _finite(row.get("clipiqa"))
        confidence = _finite(row.get("confidence"))
        return (
            musiq if musiq is not None else 101.0,
            clipiqa if clipiqa is not None else 2.0,
            -(confidence if confidence is not None else -1.0),
        )

    top = []
    for row in sorted(rows, key=top_key)[:REGION_TOP_K]:
        geometry = _box_geometry(row, width=width, height=height)
        top.append({
            "region_id": row.get("region_id"),
            "parent_id": row.get("parent_id"),
            "kind": row.get("kind", "detection" if catalog else None),
            "class_id": row.get("class_id"),
            "confidence": _finite(row.get("confidence")),
            "area_fraction": geometry["area_fraction"] if geometry else None,
            "minimum_edge_margin": geometry["minimum_edge_margin"] if geometry else None,
            "musiq": _finite(row.get("musiq")),
            "clipiqa": _finite(row.get("clipiqa")),
        })
    return {
        "count": len(rows),
        "by_kind": dict(sorted(kinds.items())),
        "by_class_id": dict(sorted(classes.items())),
        "confidence": _stats(row.get("confidence") for row in rows),
        "area_fraction": _stats(g["area_fraction"] for g in geometries),
        "edges": _edge_summary(geometries),
        "musiq": _stats(row.get("musiq") for row in rows),
        "clipiqa": _stats(row.get("clipiqa") for row in rows),
        "top_regions": top,
        "top_k_limit": REGION_TOP_K,
        "invalid_box_count": len(rows) - len(geometries),
    }


def _region_summary(
    value: Any,
    *,
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    record = value if isinstance(value, Mapping) else {}
    regions = _region_collection(
        record.get("regions"), width=width, height=height, catalog=False,
    )
    catalog = _region_collection(
        record.get("catalog"), width=width, height=height, catalog=True,
    )
    subject_count = record.get("subject_count", record.get("detected_subject_count"))
    contradictions = []
    region_subjects = int(regions["by_kind"].get("subject", 0))
    catalog_count = catalog["count"]
    if isinstance(subject_count, int) and region_subjects and subject_count != region_subjects:
        contradictions.append("recorded_subject_count_differs_from_region_subject_count")
    if isinstance(subject_count, int) and catalog_count and subject_count != catalog_count:
        contradictions.append("recorded_subject_count_differs_from_catalog_count")
    if record.get("status") == "complete" and regions["count"] == 0:
        contradictions.append("complete_status_without_region_records")
    if regions["by_class_id"] and catalog["by_class_id"] \
            and regions["by_class_id"] != catalog["by_class_id"]:
        contradictions.append("region_and_catalog_class_counts_differ")
    unknowns = [
        key for key in ("eye_state", "amodal_completeness")
        if str(record.get(key, "")).lower() in {"", "unknown", "none"}
    ]
    authority = {
        key: record[key] for key in sorted(record)
        if "authority" in key or key in {
            "catalog_semantics", "regime_label_input", "evaluation_labels_used",
        }
    }
    return {
        "status": record.get("status"),
        "reason": record.get("reason"),
        "schema_version": record.get("schema_version"),
        "method": record.get("method"),
        "recorded_subject_count": subject_count,
        "regions": regions,
        "catalog": catalog,
        "contradictions": contradictions,
        "unknowns": unknowns,
        "applicability": record.get("status", "unknown"),
        "authority": authority or {"default": "measurement_context_not_ground_truth"},
        "summarized_sources": ["regions", "catalog", "box coordinates", "embedding offsets"],
    }


def _face_summary(
    value: Any,
    *,
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    rows = [row for row in value if isinstance(row, Mapping)] \
        if isinstance(value, (list, tuple)) else []
    geometries = [
        geometry for row in rows
        if (geometry := _box_geometry(
            row, width=width, height=height, xywh=True,
        )) is not None
    ]
    top = []
    for row in sorted(
        rows, key=lambda item: _finite(item.get("score")) or -1.0, reverse=True,
    )[:REGION_TOP_K]:
        geometry = _box_geometry(row, width=width, height=height, xywh=True)
        top.append({
            "score": _finite(row.get("score")),
            "area_fraction": geometry["area_fraction"] if geometry else None,
            "minimum_edge_margin": geometry["minimum_edge_margin"] if geometry else None,
        })
    return {
        "count": len(rows),
        "confidence": _stats(row.get("score") for row in rows),
        "area_fraction": _stats(g["area_fraction"] for g in geometries),
        "edges": _edge_summary(geometries),
        "top_faces": top,
        "top_k_limit": REGION_TOP_K,
        "invalid_box_count": len(rows) - len(geometries),
        "applicability": "available" if rows else "unknown_or_empty",
        "authority": "detector_observation_only",
        "summarized_sources": ["face records", "bbox coordinates"],
    }


def _pose_summary(
    value: Any,
    *,
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    record = value if isinstance(value, Mapping) else {}
    people = [person for person in record.get("people", []) if isinstance(person, Mapping)] \
        if isinstance(record.get("people"), list) else []
    geometries = [
        geometry for person in people
        if (geometry := _box_geometry(person, width=width, height=height)) is not None
    ]
    keypoint_counts = []
    keypoint_confidences = []
    top_people = []
    for person in people:
        points = person.get("keypoints")
        points = points if isinstance(points, list) else []
        confidences = [
            confidence for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 3
            and (confidence := _finite(point[2])) is not None
        ]
        usable = sum(confidence >= 0.6 for confidence in confidences)
        keypoint_counts.append(usable)
        keypoint_confidences.extend(confidences)
        geometry = _box_geometry(person, width=width, height=height)
        top_people.append({
            "box_confidence": _finite(person.get("box_confidence")),
            "area_fraction": geometry["area_fraction"] if geometry else None,
            "minimum_edge_margin": geometry["minimum_edge_margin"] if geometry else None,
            "keypoint_count": len(points),
            "usable_keypoint_count": usable,
            "keypoint_confidence": _stats(confidences),
        })
    top_people.sort(
        key=lambda item: (
            item["usable_keypoint_count"],
            item["box_confidence"] if item["box_confidence"] is not None else -1,
        ),
        reverse=True,
    )
    authority = {
        key: record[key] for key in sorted(record)
        if "authority" in key or key in {"review_only", "keeper_authority"}
    }
    return {
        "status": record.get("status"),
        "schema_version": record.get("schema_version"),
        "producer": record.get("producer"),
        "pose_format": record.get("pose_format"),
        "people_count": len(people),
        "box_confidence": _stats(person.get("box_confidence") for person in people),
        "area_fraction": _stats(g["area_fraction"] for g in geometries),
        "edges": _edge_summary(geometries),
        "usable_keypoint_count": _stats(keypoint_counts),
        "keypoint_confidence": _stats(keypoint_confidences),
        "top_people": top_people[:RECORD_TOP_K],
        "top_k_limit": RECORD_TOP_K,
        "invalid_box_count": len(people) - len(geometries),
        "unknowns": ["people"] if not people else [],
        "applicability": record.get("status", "unknown"),
        "authority": authority or {"semantic_phase_authority": False},
        "summarized_sources": ["people", "person boxes", "COCO keypoint coordinates"],
    }


def _generic_summary(value: Any) -> dict[str, Any]:
    exclusions: list[dict[str, str]] = []
    leaves = _flatten("", value, exclusions)
    numeric: dict[str, list[float]] = defaultdict(list)
    categorical: dict[str, Counter[str]] = defaultdict(Counter)
    states: dict[str, Any] = {}
    null_count = 0
    for path, raw in leaves:
        normalized = _INDEX.sub("[]", path)
        leaf_name = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        number = _finite(raw)
        if raw is None:
            null_count += 1
        elif number is not None:
            numeric[normalized].append(number)
        elif isinstance(raw, (str, bool)):
            categorical[normalized][str(raw)] += 1
        if leaf_name in _STATE_KEYS or "authority" in leaf_name:
            states[path] = raw
    numeric_stats = [
        {"source": path, **_stats(values)}
        for path, values in sorted(numeric.items())
    ][:MAX_GENERIC_STATS]
    categorical_stats = []
    for path, counts in sorted(categorical.items())[:MAX_GENERIC_CATEGORIES]:
        categorical_stats.append({
            "source": path,
            "count": sum(counts.values()),
            "top_values": [
                {"value": value, "count": count}
                for value, count in counts.most_common(RECORD_TOP_K)
            ],
        })
    return {
        "source_bytes": len(_canonical(value)) if not isinstance(
            value, (bytes, bytearray, memoryview),
        ) else len(value),
        "leaf_count": len(leaves),
        "null_count": null_count,
        "states": dict(sorted(states.items())[:MAX_NUMERIC_ARRAY]),
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "collections": _sequence_lengths(value)[:MAX_NUMERIC_ARRAY],
        "excluded_sources": exclusions,
        "applicability": states.get("status", states.get("state", "as_recorded")),
        "authority": {
            path: raw for path, raw in states.items() if "authority" in path.lower()
        } or {"default": "measurement_context_not_ground_truth"},
    }


def _entry(
    ref: str,
    family: str,
    representation: str,
    value: Any,
    *,
    applicability: Any = "as_recorded",
    authority: Any = "persisted_measurement",
) -> dict[str, Any]:
    return {
        "ref": ref,
        "source_family": family,
        "representation": representation,
        "value": value,
        "applicability": applicability,
        "authority": authority,
    }


def _family_entries(
    alias: str,
    family: str,
    value: Any,
    *,
    width: float | None,
    height: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    exclusions: list[dict[str, str]] = []
    special = _SPECIAL_SUMMARIES.get(family)
    if special == "region_set":
        summary = _region_summary(value, width=width, height=height)
    elif special == "pose":
        summary = _pose_summary(value, width=width, height=height)
    elif special == "faces":
        summary = _face_summary(value, width=width, height=height)
    elif family != "quality_meta.exposure" and _needs_summary(value):
        summary = _generic_summary(value)
    else:
        leaves = _flatten("", value, exclusions)
        entries = [
            _entry(
                f"{alias}:{family}{'.' if path and not path.startswith('[') else ''}{path}",
                family,
                "exact",
                scalar,
            )
            for path, scalar in leaves
        ]
        disposition = "exact" if entries else "non-evidence"
        return entries, {
            "scope": "frame",
            "frame": alias,
            "source_family": family,
            "disposition": disposition,
            "reason": "compact_scalar_leaves_preserved_exactly"
            if entries else "privacy_or_non_scalar_storage_only",
            "replacement_refs": [entry["ref"] for entry in entries],
            "excluded_sources": exclusions,
        }, value
    ref = f"{alias}:COMPILED:{family}"
    entries = [_entry(
        ref,
        family,
        "summary",
        summary,
        applicability=summary.get("applicability", "as_recorded"),
        authority=summary.get("authority", "measurement_context_not_ground_truth"),
    )]
    return entries, {
        "scope": "frame",
        "frame": alias,
        "source_family": family,
        "disposition": "summarized",
        "reason": (
            "storage_collection_compiled_to_bounded_measurements"
            if special else "large_or_deep_family_compiled_to_bounded_statistics"
        ),
        "replacement_refs": [ref],
        "summarized_sources": summary.get("summarized_sources", []),
        "excluded_sources": summary.get("excluded_sources", []),
    }, value


def _pair_ref(left: str, right: str, family: str) -> str:
    return f"PAIR:{left}:{right}:compiled_{family.replace('.', '_')}"


def _count_deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {
        key: int(right.get(key, 0)) - int(left.get(key, 0))
        for key in sorted(keys)
    }


def _region_pair(
    left_alias: str,
    right_alias: str,
    family: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_regions = left["regions"]
    right_regions = right["regions"]
    ref = _pair_ref(left_alias, right_alias, family)
    return {
        "ref": ref,
        "criterion": family,
        "metric": "compiled_region_set_comparison",
        "status": "measured",
        "value": {
            "region_count_delta": right_regions["count"] - left_regions["count"],
            "catalog_count_delta": right["catalog"]["count"] - left["catalog"]["count"],
            "class_count_deltas": _count_deltas(
                left_regions["by_class_id"], right_regions["by_class_id"],
            ),
            "kind_count_deltas": _count_deltas(
                left_regions["by_kind"], right_regions["by_kind"],
            ),
            "confidence_mean_delta": _delta(
                left_regions["confidence"]["mean"], right_regions["confidence"]["mean"],
            ),
            "area_mean_delta": _delta(
                left_regions["area_fraction"]["mean"],
                right_regions["area_fraction"]["mean"],
            ),
            "minimum_edge_margin_delta": _delta(
                left_regions["edges"]["minimum_edge_margin"]["min"],
                right_regions["edges"]["minimum_edge_margin"]["min"],
            ),
            "musiq_min_delta": _delta(
                left_regions["musiq"]["min"], right_regions["musiq"]["min"],
            ),
            "musiq_mean_delta": _delta(
                left_regions["musiq"]["mean"], right_regions["musiq"]["mean"],
            ),
            "clipiqa_min_delta": _delta(
                left_regions["clipiqa"]["min"], right_regions["clipiqa"]["min"],
            ),
            "clipiqa_mean_delta": _delta(
                left_regions["clipiqa"]["mean"], right_regions["clipiqa"]["mean"],
            ),
            "contradictions": sorted(set(left["contradictions"] + right["contradictions"])),
            "unknowns": sorted(set(left["unknowns"] + right["unknowns"])),
        },
        "source_refs": [
            f"{left_alias}:COMPILED:{family}",
            f"{right_alias}:COMPILED:{family}",
        ],
        "authority": "compiled_from_persisted_region_observations",
    }


def _face_pair(
    left_alias: str,
    right_alias: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    family = "faces_json"
    return {
        "ref": _pair_ref(left_alias, right_alias, family),
        "criterion": family,
        "metric": "compiled_face_detection_comparison",
        "status": "measured",
        "value": {
            "count_delta": right["count"] - left["count"],
            "confidence_mean_delta": _delta(
                left["confidence"]["mean"], right["confidence"]["mean"],
            ),
            "area_mean_delta": _delta(
                left["area_fraction"]["mean"], right["area_fraction"]["mean"],
            ),
            "minimum_edge_margin_delta": _delta(
                left["edges"]["minimum_edge_margin"]["min"],
                right["edges"]["minimum_edge_margin"]["min"],
            ),
        },
        "source_refs": [
            f"{left_alias}:COMPILED:{family}",
            f"{right_alias}:COMPILED:{family}",
        ],
        "authority": "detector_observation_only",
    }


def _pose_pair(
    left_alias: str,
    right_alias: str,
    left_summary: Mapping[str, Any],
    right_summary: Mapping[str, Any],
    left_raw: Any,
    right_raw: Any,
) -> dict[str, Any]:
    family = "quality_meta.pose_evidence"
    compared = pose_evidence.compare_poses(
        left_raw if isinstance(left_raw, Mapping) else None,
        right_raw if isinstance(right_raw, Mapping) else None,
    )
    return {
        "ref": _pair_ref(left_alias, right_alias, family),
        "criterion": family,
        "metric": "compiled_pose_comparison",
        "status": "measured" if compared.get("eligible") else "unavailable",
        "value": {
            "people_count_delta": (
                right_summary["people_count"] - left_summary["people_count"]
            ),
            "usable_keypoint_mean_delta": _delta(
                left_summary["usable_keypoint_count"]["mean"],
                right_summary["usable_keypoint_count"]["mean"],
            ),
            "pose_comparison": compared,
        },
        "source_refs": [
            f"{left_alias}:COMPILED:{family}",
            f"{right_alias}:COMPILED:{family}",
        ],
        "authority": "pose_observation_without_semantic_phase_authority",
    }


def _deduplicate_static_entries(
    frames: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, bytes], list[tuple[str, str]]] = defaultdict(list)
    for frame in frames:
        alias = frame["frame_alias"]
        for criterion in frame["criteria"]:
            if criterion["representation"] != "exact":
                continue
            source = criterion["ref"].split(":", 1)[1]
            leaf = source.rsplit(".", 1)[-1].split("[", 1)[0]
            if leaf not in _STATIC_KEYS:
                continue
            candidates[(source, _canonical(criterion["value"]))].append(
                (alias, criterion["ref"]),
            )
    replacements: dict[str, str] = {}
    shared = []
    for (source, encoded), occurrences in sorted(candidates.items()):
        if len(occurrences) < 2:
            continue
        ref = f"SHARED:{source}:{len(shared) + 1:02d}"
        shared.append({
            "ref": ref,
            "source": source,
            "value": json.loads(encoded),
            "frames": [alias for alias, _old_ref in occurrences],
            "representation": "exact_deduplicated_constant",
        })
        replacements.update({old_ref: ref for _alias, old_ref in occurrences})
    if not replacements:
        return []
    for frame in frames:
        frame["criteria"] = [
            criterion for criterion in frame["criteria"]
            if criterion["ref"] not in replacements
        ]
    for entry in ledger:
        refs = entry.get("replacement_refs", [])
        entry["replacement_refs"] = list(dict.fromkeys(
            replacements.get(ref, ref) for ref in refs
        ))
    return shared


def compile_model_context(
    members: Sequence[Mapping[str, Any]],
    *,
    group: Mapping[str, Any],
    transformation_version: str,
    binary_pairs: Sequence[Mapping[str, Any]],
    binary_refs: Sequence[str],
    exclusions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a bounded criterion context and complete coverage ledger."""
    frames: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, tuple[dict[str, Any], Any]]] = defaultdict(dict)
    direct_exact = {
        "width", "height",
        "file_kind", "scan_status", "scan_error", "quality_score", "face_count",
        "feature_status",
    }
    transformed = {
        "phash", "content_sha256", "dinov2_embedding", "local_quality_embedding",
    }
    non_evidence = {
        "size_bytes": "storage_metadata",
        "exif_datetime": "capture_timestamp_not_discriminator",
        "exif_timestamp": "capture_timestamp_not_discriminator",
        "file_id": "opaque_runtime_identifier_not_cv_evidence",
        "mtime_ns": "source_integrity_metadata_not_cv_evidence",
        "last_run_id": "opaque_runtime_identifier_not_cv_evidence",
        "thumb_status": "thumbnail_cache_metadata_not_source_cv_evidence",
        "thumb_max_px": "thumbnail_cache_metadata_not_source_cv_evidence",
        "thumb_bytes": "thumbnail_cache_metadata_not_source_cv_evidence",
        "thumb_source_size_bytes": "thumbnail_cache_metadata_not_source_cv_evidence",
        "thumb_source_mtime_ns": "thumbnail_cache_metadata_not_source_cv_evidence",
        "thumb_error": "thumbnail_cache_metadata_not_source_cv_evidence",
        "thumb_created_at": "thumbnail_cache_metadata_not_source_cv_evidence",
    }
    aliases_by_id = {
        int(row["file_id"]): str(row["frame_alias"])
        for row in members if row.get("file_id") is not None
    }
    for member in members:
        alias = str(member["frame_alias"])
        criteria: list[dict[str, Any]] = []
        width = _finite(member.get("width"))
        height = _finite(member.get("height"))
        for field in sorted(direct_exact):
            if field not in member:
                continue
            ref = f"{alias}:{field}"
            criteria.append(_entry(ref, field, "exact", _safe_scalar(member.get(field))))
            ledger.append({
                "scope": "frame", "frame": alias, "source_family": field,
                "disposition": "exact",
                "reason": "compact_scalar_preserved_exactly",
                "replacement_refs": [ref],
            })
        partner = member.get("motion_partner_id")
        partner_ref = f"{alias}:motion_partner"
        criteria.append(_entry(partner_ref, "motion_partner_id", "transformed", {
            "present": partner is not None,
            "group_frame_alias": aliases_by_id.get(int(partner))
            if isinstance(partner, int) else None,
        }))
        ledger.append({
            "scope": "frame", "frame": alias, "source_family": "motion_partner_id",
            "disposition": "summarized",
            "reason": "opaque_identifier_replaced_by_presence_and_in_group_alias",
            "replacement_refs": [partner_ref],
        })
        for field in transformed:
            if field in member:
                refs = [
                    m["ref"] for pair in binary_pairs if alias in pair["frames"]
                    for m in pair["measurements"] if m["criterion"] == field
                ]
                ledger.append({
                    "scope": "frame", "frame": alias, "source_family": field,
                    "disposition": "derived",
                    "reason": "raw_storage_replaced_by_pairwise_measurement",
                    "replacement_refs": refs,
                })
        for field, reason in sorted(non_evidence.items()):
            if field in member:
                ledger.append({
                    "scope": "frame", "frame": alias, "source_family": field,
                    "disposition": "non-evidence", "reason": reason,
                    "replacement_refs": [],
                })

        quality = _decoded(member.get("quality_meta"))
        if isinstance(quality, Mapping):
            for key in sorted(quality, key=str):
                family = f"quality_meta.{key}"
                entries, coverage, raw = _family_entries(
                    alias, family, quality[key], width=width, height=height,
                )
                criteria.extend(entries)
                ledger.append(coverage)
                if entries and entries[0]["representation"] == "summary":
                    summaries[family][alias] = (entries[0]["value"], raw)
        else:
            entries, coverage, _raw = _family_entries(
                alias, "quality_meta", quality, width=width, height=height,
            )
            criteria.extend(entries)
            ledger.append(coverage)

        evidence = _decoded(member.get("evidence_json"))
        if isinstance(evidence, Mapping):
            for key in sorted(evidence, key=str):
                family = f"evidence_json.{key}"
                if any(part in key.lower() for part in ("utility", "pair_margin", "keeper", "decision")):
                    ledger.append({
                        "scope": "frame", "frame": alias, "source_family": family,
                        "disposition": "non-evidence", "reason": "selection_anchor",
                        "replacement_refs": [],
                    })
                    continue
                entries, coverage, raw = _family_entries(
                    alias, family, evidence[key], width=width, height=height,
                )
                criteria.extend(entries)
                ledger.append(coverage)
                if entries and entries[0]["representation"] == "summary":
                    summaries[family][alias] = (entries[0]["value"], raw)
        else:
            entries, coverage, _raw = _family_entries(
                alias, "evidence_json", evidence, width=width, height=height,
            )
            criteria.extend(entries)
            ledger.append(coverage)

        faces = _decoded(member.get("faces_json"))
        entries, coverage, raw = _family_entries(
            alias, "faces_json", faces, width=width, height=height,
        )
        criteria.extend(entries)
        ledger.append(coverage)
        summaries["faces_json"][alias] = (entries[0]["value"], raw)
        frames.append({"frame_alias": alias, "criteria": criteria})

    group_criteria = []
    for field in ("group_type", "member_count", "clustered_member_count"):
        if field in group:
            ref = f"GROUP:{field}"
            group_criteria.append(_entry(ref, field, "exact", _safe_scalar(group[field])))
            ledger.append({
                "scope": "group", "source_family": field, "disposition": "exact",
                "reason": "compact_group_context_preserved_exactly",
                "replacement_refs": [ref],
            })
    for item in exclusions:
        scope = str(item.get("scope", "unknown"))
        field = str(item.get("field", "unknown"))
        ledger.append({
            "scope": scope,
            "source_family": field,
            "disposition": "non-evidence",
            "reason": str(item.get("reason", "explicit_non_evidence")),
            "replacement_refs": [],
        })

    shared = _deduplicate_static_entries(frames, ledger)
    pair_map = {
        tuple(pair["frames"]): {
            "frames": list(pair["frames"]),
            "measurements": list(pair["measurements"]),
        }
        for pair in binary_pairs
    }
    aliases = [str(member["frame_alias"]) for member in members]
    for left_index, left_alias in enumerate(aliases):
        for right_alias in aliases[left_index + 1:]:
            pair = pair_map.setdefault(
                (left_alias, right_alias),
                {"frames": [left_alias, right_alias], "measurements": []},
            )
            for family, by_alias in summaries.items():
                if left_alias not in by_alias or right_alias not in by_alias:
                    continue
                left_summary, left_raw = by_alias[left_alias]
                right_summary, right_raw = by_alias[right_alias]
                if family in {"quality_meta.local_region_set", "quality_meta.local_quality"}:
                    pair["measurements"].append(_region_pair(
                        left_alias, right_alias, family, left_summary, right_summary,
                    ))
                elif family == "faces_json":
                    pair["measurements"].append(_face_pair(
                        left_alias, right_alias, left_summary, right_summary,
                    ))
                elif family == "quality_meta.pose_evidence":
                    pair["measurements"].append(_pose_pair(
                        left_alias, right_alias, left_summary, right_summary,
                        left_raw, right_raw,
                    ))

    pairwise = [pair_map[key] for key in sorted(pair_map)]
    evidence_refs = [entry["ref"] for entry in group_criteria]
    evidence_refs.extend(item["ref"] for item in shared)
    evidence_refs.extend(
        criterion["ref"] for frame in frames for criterion in frame["criteria"]
    )
    evidence_refs.extend(
        measurement["ref"] for pair in pairwise for measurement in pair["measurements"]
    )
    context = {
        "schema_version": 3,
        "compiler_schema_version": SCHEMA_VERSION,
        "mode": "shadow_review_only",
        "context_transformation_version": transformation_version,
        "group_criteria": group_criteria,
        "shared_constants": shared,
        "frames": frames,
        "pairwise_measurements": pairwise,
        "evidence_refs": list(dict.fromkeys(evidence_refs)),
        "coverage_ledger": ledger,
        "policy": {
            "drop_weak_signals": False,
            "drop_default_off_signals": False,
            "drop_shadow_signals": False,
            "redacts_filesystem_paths": True,
            "binary_encoding": "forbidden_in_model_context",
            "max_numeric_array_length": MAX_NUMERIC_ARRAY,
            "region_top_k": REGION_TOP_K,
            "authority": "measurement_context_not_ground_truth",
        },
        "instruction": (
            "Use every criterion family through its exact or compiled refs. "
            "Weak/default-off/shadow-only values retain their recorded authority. "
            "Summaries replace storage leaves, not criterion families; consult the "
            "coverage ledger for every exclusion and replacement."
        ),
    }
    return context


def context_bytes(context: Mapping[str, Any]) -> int:
    return len(_canonical(context))


def coverage_report(derived_context: Mapping[str, Any]) -> dict[str, Any]:
    """L1 coverage only. Layer sizes and prompt gates are reported separately."""
    dispositions = Counter(
        str(entry.get("disposition"))
        for entry in derived_context.get("coverage_ledger", [])
        if isinstance(entry, Mapping)
    )
    return {
        "coverage": dict(sorted(dispositions.items())),
        "coverage_entries": len(derived_context.get("coverage_ledger", [])),
        "evidence_refs": len(derived_context.get("evidence_refs", [])),
    }
