"""Bounded offline VLM judgments stored as decision-neutral review sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    db,
    local_quality,
    root_scope,
    vlm_criterion_compiler,
    vlm_identity,
    vlm_prompt_context,
    vlm_visual_budget,
)
from .semantic_observer import persisted_group_cv_context
from .vlm_semantic import sanitize_group

SCHEMA_VERSION = 3
MODE = "shadow_review_only"
CONTEXT_TRANSFORMATION_VERSION = "three-layer-identity-cv-v3"
CACHE_DIR = "vlm-review"
MAX_GROUPS = 20
MAX_REPORT_GROUPS = 100
DEFAULT_LIMIT = 8
MAX_MEMBERS = 20
SHEET_MEMBERS = 6
MAX_MODEL_IMAGES = 8
MAX_MODEL_BYTES = 12_000_000
ALLOWED_MODELS = {
    "global.anthropic.claude-sonnet-5-5",
    "global.openai.gpt-5.6-terra",
}
_GROUP_FIELDS = (
    "group_id", "group_type", "keep_file_id", "group_created_at",
    "decision_state", "group_confidence", "policy_version", "decision_json",
    "member_count", "clustered_member_count",
)
_CONTEXT_GROUP_FIELDS = (
    "group_id", "group_type", "group_created_at", "member_count",
    "clustered_member_count",
)
_CV_MEMBER_FIELDS = (
    "file_id", "width", "height", "size_bytes", "mtime_ns", "exif_datetime",
    "exif_timestamp", "file_kind", "motion_partner_id", "scan_status",
    "scan_error", "last_run_id", "phash", "content_sha256",
    "dinov2_embedding", "local_quality_embedding", "quality_score",
    "quality_meta", "face_count", "faces_json", "feature_status",
    "evidence_json", "thumb_status", "thumb_max_px", "thumb_bytes",
    "thumb_source_size_bytes", "thumb_source_mtime_ns", "thumb_error",
    "thumb_created_at",
)
_RAW_BINARY_FIELDS = {
    "phash", "dinov2_embedding", "local_quality_embedding",
}
_OPAQUE_NON_EVIDENCE_MEMBER_FIELDS = {
    "file_id", "mtime_ns", "last_run_id",
    "thumb_source_mtime_ns", "thumb_created_at",
}
_OPAQUE_NON_EVIDENCE_GROUP_FIELDS = {
    "group_id", "group_created_at",
}
_MODEL_CONTEXT_EXCLUSIONS = (
    {"scope": "group", "field": "group_id", "reason": "opaque_runtime_identifier_not_cv_evidence"},
    {"scope": "group", "field": "group_created_at", "reason": "sidecar_creation_metadata_not_cv_evidence"},
    {"scope": "group", "field": "keep_file_id", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "group", "field": "decision_state", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "group", "field": "group_confidence", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "group", "field": "policy_version", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "group", "field": "decision_json", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "member", "field": "file_id", "reason": "opaque_runtime_identifier_not_cv_evidence"},
    {"scope": "member", "field": "mtime_ns", "reason": "source_integrity_metadata_not_cv_evidence"},
    {"scope": "member", "field": "last_run_id", "reason": "opaque_runtime_identifier_not_cv_evidence"},
    {"scope": "member", "field": "thumb_source_mtime_ns", "reason": "source_integrity_metadata_not_cv_evidence"},
    {"scope": "member", "field": "thumb_created_at", "reason": "thumbnail_creation_metadata_not_cv_evidence"},
    {"scope": "member", "field": "basename", "reason": "privacy_filename"},
    {"scope": "member", "field": "is_keep", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "member", "field": "decision", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "member", "field": "reason", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "member", "field": "member_confidence", "reason": "existing_decision_authority_not_cv_evidence"},
    {"scope": "member", "field": "user_override", "reason": "existing_decision_authority_not_cv_evidence"},
)

PROMPT = """You are an independent group-level photo judge for human review.
Do not assume the existing keeper is correct; it is not identified in the images or context.
Inspect every labelled full-frame sheet and useful crop. First determine the visible
scene/action regime and adapt the comparison criteria: a decisive moment, intentional
motion, stage lighting, multi-person interaction, and fireworks do not share one fixed
quality ranking. Do not force one winner when frames capture non-substitutable action,
expression, interaction, or scene phases; recommend multiple frames when each preserves a
distinct worthwhile moment, and give a grounded discriminator for every recommendation.
Consider every field in MODEL_CONTEXT_JSON, including weak, missing, default-off, and
shadow-only signals, while preserving each field's stated authority. Recommend the
frame(s) a human should inspect
first and return no more than three decisive distinctions. Every recommendation must be
grounded by at least one discriminator that names it as a subject and cites that same
frame/region and/or an exact CV ref. Distinct visible action, expression, interaction, or
scene phases are non-substitutable: do not collapse them to one winner merely because one
frame has higher technical quality. Recommend multiple frames when multiple worthwhile
phases exist, and independently ground every recommended frame. Every discriminator needs
visual_refs and/or exact cv_refs copied from MODEL_CONTEXT_JSON. A low-confidence but
grounded observation may only be an optional attention pointer; it must not ground or
justify any recommendation. Do not identify people or infer age, relationship,
personality, diagnosis, ethnicity, religion, sexuality, politics, hidden intent, or
internal emotion. Abstain when evidence is not reliable. Write every human-readable
reason in concise Simplified Chinese. Keep JSON keys, enum values, frame aliases and
region IDs exactly as the schema defines them in English.

Allowed discriminator codes: EXPRESSION_DIFFERENCE, POSE_ACTION_DIFFERENCE,
CAMERA_ENGAGEMENT, SCENE_INFORMATION_DIFFERENCE, UNIQUE_MOMENT,
EYE_STATE_DIFFERENCE, FACE_QUALITY_DIFFERENCE, TECHNICAL_QUALITY_DIFFERENCE,
EXPOSURE_DIFFERENCE, FRAMING_DIFFERENCE, COMPOSITE_ADVANTAGE,
NO_RELIABLE_DIFFERENCE.

Output exactly one JSON object, no Markdown:
{"schema_version":1,"group_id":GROUP_ID,"judgment":{"recommended":["F01"],
"confidence":"high","discriminators":[{"code":"COMPOSITE_ADVANTAGE",
"subjects":["F01"],"compared_to":["F02"],"reason":"concise visible distinction",
"confidence":"high","visual_refs":[{"frame":"F01","region":"face"}],
"cv_refs":["F01:exact.ref"]}]},"abstentions":[]}
"""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["file_id"]))
    for index, row in enumerate(ordered, start=1):
        row["frame_alias"] = f"F{index:02d}"
    return ordered


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = dict(rows[0])
    return {key: first.get(key) for key in _GROUP_FIELDS if key in first}


def _unit_vector(blob: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    source = {"storage": "little_endian_float16", "byte_length": None, "dimensions": None}
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None, {**source, "status": "missing"}
    raw = bytes(blob)
    source["byte_length"] = len(raw)
    if not raw or len(raw) % np.dtype("<f2").itemsize:
        return None, {**source, "status": "malformed"}
    vector = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    source["dimensions"] = int(vector.size)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or not math.isfinite(norm) or norm <= 0:
        return None, {**source, "status": "malformed"}
    return vector / norm, {**source, "status": "available"}


def _cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return similarity, float(1.0 - similarity)


def _pair_ref(left: str, right: str, metric: str) -> str:
    return f"PAIR:{left}:{right}:{metric}"


def _pairwise_binary_measurements(
    members: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    pairs: list[dict[str, Any]] = []
    refs: list[str] = []
    dino = [_unit_vector(row.get("dinov2_embedding")) for row in members]
    local = [local_quality.observations(row) for row in members]
    for left_index, left in enumerate(members):
        for right_index in range(left_index + 1, len(members)):
            right = members[right_index]
            left_alias = str(left["frame_alias"])
            right_alias = str(right["frame_alias"])
            measurements: list[dict[str, Any]] = []

            phash_ref = _pair_ref(left_alias, right_alias, "phash_hamming_distance")
            left_hash, right_hash = left.get("phash"), right.get("phash")
            hash_ok = (
                isinstance(left_hash, (bytes, bytearray, memoryview))
                and isinstance(right_hash, (bytes, bytearray, memoryview))
                and len(left_hash) == len(right_hash)
                and len(left_hash) > 0
            )
            hamming = (
                (
                    int.from_bytes(bytes(left_hash), "big")
                    ^ int.from_bytes(bytes(right_hash), "big")
                ).bit_count()
                if hash_ok else None
            )
            measurements.append({
                "ref": phash_ref,
                "criterion": "phash",
                "metric": "hamming_distance",
                "status": "measured" if hash_ok else "unavailable",
                "value": {
                    "hamming_distance": hamming,
                    "bit_length": len(bytes(left_hash)) * 8 if hash_ok else None,
                },
                "source_refs": [f"{left_alias}:phash", f"{right_alias}:phash"],
                "source": {
                    "left_byte_length": len(left_hash) if isinstance(
                        left_hash, (bytes, bytearray, memoryview),
                    ) else None,
                    "right_byte_length": len(right_hash) if isinstance(
                        right_hash, (bytes, bytearray, memoryview),
                    ) else None,
                },
            })
            refs.append(phash_ref)

            dino_ref = _pair_ref(
                left_alias, right_alias, "dinov2_cosine_similarity_distance",
            )
            left_dino, left_dino_source = dino[left_index]
            right_dino, right_dino_source = dino[right_index]
            dino_ok = (
                left_dino is not None
                and right_dino is not None
                and left_dino.shape == right_dino.shape
            )
            dino_similarity, dino_distance = (
                _cosine(left_dino, right_dino) if dino_ok else (None, None)
            )
            measurements.append({
                "ref": dino_ref,
                "criterion": "dinov2_embedding",
                "metric": "cosine_similarity_and_distance",
                "status": "measured" if dino_ok else "unavailable",
                "value": {
                    "cosine_similarity": dino_similarity,
                    "cosine_distance": dino_distance,
                },
                "source_refs": [
                    f"{left_alias}:dinov2_embedding",
                    f"{right_alias}:dinov2_embedding",
                ],
                "source": {
                    "left_status": left_dino_source["status"],
                    "right_status": right_dino_source["status"],
                    "left_dimensions": left_dino_source["dimensions"],
                    "right_dimensions": right_dino_source["dimensions"],
                    "left_byte_length": left_dino_source["byte_length"],
                    "right_byte_length": right_dino_source["byte_length"],
                },
            })
            refs.append(dino_ref)

            local_ref = _pair_ref(
                left_alias, right_alias, "local_quality_cosine_similarity_distance",
            )
            left_local, right_local = local[left_index], local[right_index]
            region_values = []
            for kind, class_id in sorted(left_local.keys() & right_local.keys()):
                similarity, distance = _cosine(
                    left_local[(kind, class_id)]["vector"],
                    right_local[(kind, class_id)]["vector"],
                )
                region_values.append({
                    "kind": kind,
                    "class_id": class_id,
                    "cosine_similarity": similarity,
                    "cosine_distance": distance,
                })
            ranked_regions = sorted(
                region_values, key=lambda item: item["cosine_distance"], reverse=True,
            )
            measurements.append({
                "ref": local_ref,
                "criterion": "local_quality_embedding",
                "metric": "cosine_similarity_and_distance",
                "status": "measured" if region_values else "unavailable",
                "value": {
                    "region_count": len(region_values),
                    "top_regions": ranked_regions[
                        :vlm_criterion_compiler.REGION_TOP_K
                    ],
                    "top_k_limit": vlm_criterion_compiler.REGION_TOP_K,
                    "minimum_cosine_similarity": min(
                        (item["cosine_similarity"] for item in region_values),
                        default=None,
                    ),
                    "maximum_cosine_distance": max(
                        (item["cosine_distance"] for item in region_values),
                        default=None,
                    ),
                    "mean_cosine_similarity": (
                        sum(item["cosine_similarity"] for item in region_values)
                        / len(region_values)
                        if region_values else None
                    ),
                },
                "source_refs": [
                    f"{left_alias}:local_quality_embedding",
                    f"{right_alias}:local_quality_embedding",
                    f"{left_alias}:quality_meta.local_quality",
                    f"{right_alias}:quality_meta.local_quality",
                ],
                "source": {
                    "left_byte_length": len(left.get("local_quality_embedding") or b""),
                    "right_byte_length": len(right.get("local_quality_embedding") or b""),
                    "left_region_count": len(left_local),
                    "right_region_count": len(right_local),
                },
            })
            refs.append(local_ref)

            sha_ref = _pair_ref(left_alias, right_alias, "content_sha256_equal")
            left_sha, right_sha = left.get("content_sha256"), right.get("content_sha256")
            sha_ok = isinstance(left_sha, str) and isinstance(right_sha, str)
            measurements.append({
                "ref": sha_ref,
                "criterion": "content_sha256",
                "metric": "exact_equality",
                "status": "measured" if sha_ok else "unavailable",
                "value": {"equal": left_sha == right_sha if sha_ok else None},
                "source_refs": [
                    f"{left_alias}:content_sha256",
                    f"{right_alias}:content_sha256",
                ],
            })
            refs.append(sha_ref)
            pairs.append({
                "frames": [left_alias, right_alias],
                "measurements": measurements,
            })
    return pairs, refs


def _model_cv_members(members: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    aliases_by_file_id = {
        int(row["file_id"]): str(row["frame_alias"])
        for row in members if row.get("file_id") is not None
    }
    for row in members:
        item = {
            key: value for key, value in row.items()
            if key not in _RAW_BINARY_FIELDS
            and key not in _OPAQUE_NON_EVIDENCE_MEMBER_FIELDS
            and key != "content_sha256"
        }
        partner_id = row.get("motion_partner_id")
        item.pop("motion_partner_id", None)
        item["motion_partner"] = {
            "present": partner_id is not None,
            "group_frame_alias": aliases_by_file_id.get(int(partner_id))
            if isinstance(partner_id, int) else None,
        }
        projected.append(item)
    return projected


def assert_model_context_safe(context: Mapping[str, Any]) -> None:
    """Reject raw storage, oversized numeric arrays, and oversized contexts."""
    def visit(value: Any, path: str) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"raw binary is forbidden in model context: {path}")
        if isinstance(value, Mapping):
            if str(value.get("encoding", "")).lower() == "base64":
                raise ValueError(f"base64 is forbidden in model context: {path}")
            for key, child in value.items():
                lowered = str(key).lower()
                if context.get("layer") == "L2" and (
                    lowered in {"bbox", "box", "boxes", "box_normalized", "landmarks", "keypoints",
                                "audit_context", "derived_context", "alias_map"}
                    or any(part in lowered for part in ("utility", "pair_margin", "thumbnail", "timestamp", "run_id"))
                ):
                    raise ValueError(f"non-prompt field is forbidden in L2: {path}.{key}")
                if "base64" in lowered:
                    raise ValueError(
                        f"base64 field is forbidden in model context: {path}.{key}"
                    )
                interface = {
                    "phash": "perceptual_hash_xor_popcount",
                    "dinov2_embedding": "cosine_on_persisted_little_endian_float16_vectors",
                    "local_quality_embedding": "cosine_on_matching_native_local_quality_regions",
                }
                if path == "measurement_interfaces" and key in interface and child == interface[key]:
                    continue
                if str(key) in _RAW_BINARY_FIELDS:
                    raise ValueError(
                        f"raw binary field is forbidden in model context: {path}.{key}"
                    )
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            if context.get("layer") == "L2":
                bound = 1024 if path == "evidence_refs" else 128 if path == "checked_families" else 20
                if len(value) > bound:
                    raise ValueError(f"array exceeds L2 bound: {path}")
            if (
                len(value) > vlm_criterion_compiler.MAX_NUMERIC_ARRAY
                and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in value
                )
            ):
                raise ValueError(
                    "numeric array exceeds model-context bound "
                    f"{vlm_criterion_compiler.MAX_NUMERIC_ARRAY}: {path}"
                )
            if path.endswith(("top_regions", "top_faces")) and len(
                value
            ) > vlm_criterion_compiler.REGION_TOP_K:
                raise ValueError(f"region top-k exceeds bound: {path}")
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite model scalar: {path}")
        elif isinstance(value, str) and context.get("layer") == "L2" and len(value) > 500:
            raise ValueError(f"unbounded model string: {path}")

    if context.get("layer") == "L2" and (context.get("schema_version") != 1 or
            not all(key in context for key in ("evidence", "pairs", "frames", "evidence_refs"))):
        raise ValueError("invalid L2 schema")
    visit(context, "")
    size = len(_canonical(context))
    limit = (vlm_prompt_context.MAX_BYTES if context.get("layer") == "L2"
             else vlm_criterion_compiler.LEGACY_CONTEXT_GUARD_BYTES)
    if context.get("layer") == "L2" and math.ceil((size + len(PROMPT.encode("utf-8"))) / 3) > vlm_prompt_context.MAX_TOKENS_ESTIMATE:
        raise ValueError("model context exceeds fail-closed per-group limit: total text estimate >15000 tokens")
    if size > limit:
        raise ValueError(
            "model context exceeds fail-closed per-group limit "
            f"{limit} bytes: {size}"
        )


def build_layers(
    rows: Sequence[Mapping[str, Any]], *, enforce_gate: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build L0 -> L1 -> L2. Only L2 is eligible for model invocation.

    Diagnostic reports may inspect a failed L2 with enforce_gate=False; no invocation
    or sidecar production path uses that option.
    """
    members = _rows(rows)
    if not members:
        raise ValueError("empty review group")
    mapping = vlm_identity.alias_map(members)
    identities = {vlm_identity.identity(value): alias for alias, value in mapping.items()}
    cv_members = [
        {"frame_alias": vlm_identity.identity(row),
         **{key: row.get(key) for key in _CV_MEMBER_FIELDS if key in row}}
        for row in members
    ]
    group = {key: members[0].get(key) for key in _CONTEXT_GROUP_FIELDS if key in members[0]}
    audit_members = [
        {**{key: value for key, value in row.items() if key not in _GROUP_FIELDS},
         "frame_alias": vlm_identity.identity(row)} for row in members
    ]
    audit = persisted_group_cv_context(audit_members, group=_group(members))
    audit["layer"] = "L0"
    for frame in audit["frames"]:
        frame["frame_identity"] = frame.pop("frame_alias")
    pairs, refs = _pairwise_binary_measurements(cv_members)
    derived = vlm_criterion_compiler.compile_model_context(
        cv_members, group=group, transformation_version=CONTEXT_TRANSFORMATION_VERSION,
        binary_pairs=pairs, binary_refs=refs, exclusions=_MODEL_CONTEXT_EXCLUSIONS,
    )
    derived["layer"] = "L1"
    for frame in derived["frames"]:
        frame["frame_identity"] = frame.pop("frame_alias")
    for item in derived["coverage_ledger"]:
        if "frame" in item:
            item["identity"] = item.pop("frame")
    required = {
        "exact_identity": ("content_sha256",), "near_duplicate": ("phash",),
        "global_similarity": ("dinov2_embedding",),
        "global_quality": ("quality_score", "quality_meta.musiq", "quality_meta.clipiqa"),
        "face_person": ("face_count", "faces_json", "quality_meta.face_quality"),
        "eyes": ("quality_meta.detectors",), "exposure": ("quality_meta.exposure",),
        "local_regions": ("quality_meta.local_region_set", "quality_meta.local_quality", "local_quality_embedding"),
        "pose_action_phase": ("quality_meta.pose_evidence",),
        "completeness_occlusion": ("quality_meta.completeness", "quality_meta.occlusion"),
        "scene_regime": ("quality_meta.routing",),
        "applicability_authority_conflicts": ("quality_meta.routing",),
        "selection_diagnostics": ("evidence_json.utility_score", "evidence_json.pair_margin"),
    }
    derived["family_coverage"] = {}
    for family, sources in required.items():
        entries = [item for item in derived["coverage_ledger"] if item["source_family"] in sources]
        refs = sorted({ref for entry in entries for ref in entry["replacement_refs"]})
        derived["family_coverage"][family] = {
            "disposition": "derived" if refs else "non-evidence",
            "replacement_refs": refs,
            "applicability": "as_recorded" if refs else "unavailable_or_not_evidence",
            "authority": "measurement_only" if refs else "none",
            "reason": "source_family_ledger" if entries else "not_persisted",
        }
    raw_decision = members[0].get("decision_json")
    if isinstance(raw_decision, str):
        raw_decision = json.loads(raw_decision)
    if isinstance(raw_decision, Mapping) and raw_decision.get("phase_selection"):
        derived["family_coverage"]["pose_action_phase"].update(
            disposition="non-evidence", applicability="uncompiled_group_shadow_evidence",
            reason="group_shadow_phase_records_mix_selector_authority_and_indexed_measurements; see L0",
        )
        derived["coverage_ledger"].append({
            "scope": "group", "source_family": "decision_json.phase_selection",
            "disposition": "non-evidence",
            "reason": "mixed_selector_and_shadow_records_require_identity_safe_extraction",
            "replacement_refs": [],
        })
    derived["scalar_comparisons"] = vlm_prompt_context.scalar_comparisons(derived)
    model = vlm_prompt_context.compile_prompt(derived, identities)
    if enforce_gate:
        assert_model_context_safe(model)
    return members, audit, derived, model


def build_contexts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    members, audit, _derived, model = build_layers(rows)
    return members, audit, model


def layer_size_report(audit_bytes: int, derived_bytes: int, prompt_context_bytes: int) -> dict[str, Any]:
    """Unambiguous byte counts/ratios; L0 and L1 have no prompt budget."""
    return {
        "audit_bytes": audit_bytes,
        "derived_bytes": derived_bytes,
        "prompt_context_bytes": prompt_context_bytes,
        "derived_to_audit_ratio": derived_bytes / audit_bytes if audit_bytes else None,
        "prompt_context_to_audit_ratio": prompt_context_bytes / audit_bytes if audit_bytes else None,
        "prompt_context_to_derived_ratio": prompt_context_bytes / derived_bytes if derived_bytes else None,
    }


def prompt_budget_report(prompt_context_bytes: int) -> dict[str, Any]:
    template_bytes = len(PROMPT.encode("utf-8"))
    total_tokens = math.ceil((prompt_context_bytes + template_bytes) / 3)
    return {
        "token_estimator": "ceil(canonical UTF-8 bytes / 3); not provider tokenization; excludes images",
        "prompt_context_tokens_estimate": math.ceil(prompt_context_bytes / 3),
        "prompt_context_tokens_estimate_limit": vlm_prompt_context.MAX_TOKENS_ESTIMATE,
        "prompt_context_within_tokens_estimate_limit": math.ceil(prompt_context_bytes / 3) <= vlm_prompt_context.MAX_TOKENS_ESTIMATE,
        "prompt_context_bytes_limit": vlm_prompt_context.MAX_BYTES,
        "prompt_context_within_bytes_limit": prompt_context_bytes <= vlm_prompt_context.MAX_BYTES,
        "prompt_template_bytes": template_bytes,
        "total_text_tokens_estimate": total_tokens,
        "total_text_tokens_estimate_limit": vlm_prompt_context.MAX_TOKENS_ESTIMATE,
        "total_text_within_tokens_estimate_limit": total_tokens <= vlm_prompt_context.MAX_TOKENS_ESTIMATE,
    }


def context_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure even over-budget groups without visuals, services, or model calls."""
    members, audit, derived, prompt = build_layers(rows, enforce_gate=False)
    sizes = layer_size_report(*(len(_canonical(layer)) for layer in (audit, derived, prompt)))
    return {
        "group_id": int(members[0]["group_id"]),
        "member_count": len(members),
        "transformation_version": CONTEXT_TRANSFORMATION_VERSION,
        **vlm_criterion_compiler.coverage_report(derived),
        **sizes,
        **prompt_budget_report(sizes["prompt_context_bytes"]),
    }


def aggregate_context_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Ratios use totals; per-group prompt limits never apply to summed bytes."""
    coverage: dict[str, int] = {}
    for report in reports:
        for disposition, count in report["coverage"].items():
            coverage[disposition] = coverage.get(disposition, 0) + int(count)
    tokens = [report["total_text_tokens_estimate"] for report in reports]
    median = statistics.median(tokens) if tokens else None
    maximum = max(tokens) if tokens else None
    return {
        "schema_version": 2,
        "mode": "context_measurement_only",
        "model_calls": 0,
        "transformation_version": CONTEXT_TRANSFORMATION_VERSION,
        "group_count": len(reports),
        **layer_size_report(*(sum(report[key] for report in reports) for key in
                              ("audit_bytes", "derived_bytes", "prompt_context_bytes"))),
        "coverage": dict(sorted(coverage.items())),
        "prompt_budget": {
            "scope": "per_group; aggregate sums are not compared with per-group limits",
            "token_estimator": prompt_budget_report(0)["token_estimator"],
            "prompt_context_tokens_estimate_sum": sum(r["prompt_context_tokens_estimate"] for r in reports),
            "prompt_context_tokens_estimate_median": statistics.median(r["prompt_context_tokens_estimate"] for r in reports) if reports else None,
            "prompt_context_tokens_estimate_max": max(r["prompt_context_tokens_estimate"] for r in reports) if reports else None,
            "prompt_context_tokens_estimate_per_group_limit": vlm_prompt_context.MAX_TOKENS_ESTIMATE,
            "prompt_context_all_groups_within_tokens_estimate_limit": all(r["prompt_context_within_tokens_estimate_limit"] for r in reports) if reports else None,
            "total_text_tokens_estimate_sum": sum(tokens),
            "prompt_context_bytes_per_group_limit": vlm_prompt_context.MAX_BYTES,
            "prompt_context_all_groups_within_bytes_limit": all(r["prompt_context_within_bytes_limit"] for r in reports) if reports else None,
            "total_text_tokens_estimate_median": median,
            "total_text_tokens_estimate_max": maximum,
            "total_text_tokens_estimate_median_limit": 8000,
            "total_text_tokens_estimate_max_limit": vlm_prompt_context.MAX_TOKENS_ESTIMATE,
            "total_text_median_within_tokens_estimate_limit": median <= 8000 if tokens else None,
            "total_text_all_groups_within_tokens_estimate_limit": all(r["total_text_within_tokens_estimate_limit"] for r in reports) if reports else None,
        },
        "groups": list(reports),
    }


def build_context(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compatibility helper returning only the prompt-safe model context."""
    members, _audit_context, model_context = build_contexts(rows)
    return members, model_context


def input_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    members, _audit_context, model_context = build_contexts(rows)
    identity = [{
        "frame_alias": row["frame_alias"], "file_id": int(row["file_id"]),
        "size_bytes": row.get("size_bytes"), "mtime_ns": row.get("mtime_ns"),
        "content_sha256": row.get("content_sha256"),
    } for row in members]
    return _sha_bytes(_canonical({
        "context_transformation_version": CONTEXT_TRANSFORMATION_VERSION,
        "prompt_sha256": _sha_bytes(PROMPT.encode("utf-8")),
        "model_context": model_context,
        "source_identity": identity,
    }))


def route_group(rows: Sequence[Mapping[str, Any]]) -> str:
    """Only byte-exact, auto-ready groups bypass VLM as decisive CV-only cases."""
    members = _rows(rows)
    group = _group(members)
    rivals = [row for row in members if not bool(row.get("is_keep"))]
    hashes = {row.get("content_sha256") for row in members}
    exact = (
        group.get("group_type") == "sha_exact"
        and group.get("decision_state") == "AUTO_READY"
        and bool(rivals)
        and all(row.get("reason") == "BYTE_IDENTICAL" for row in rivals)
        and None not in hashes and len(hashes) == 1
    )
    return "cv_only" if exact else "vlm"


def cache_path(output: Path, group_id: int) -> Path:
    return Path(output) / CACHE_DIR / "groups" / f"{int(group_id)}.json"


def _invariants(model_calls: int) -> dict[str, Any]:
    return {
        "changes_keeper": False, "changes_sorting": False,
        "changes_routing": False, "changes_decision_tables": False,
        "deletion_authority": "none", "model_calls": model_calls,
    }


def _project(sanitized: Mapping[str, Any], members: Sequence[Mapping[str, Any]],
             source: str, model_id: str | None) -> dict[str, Any]:
    alias_to_id = {str(row["frame_alias"]): int(row["file_id"]) for row in members}
    recommendation = dict(sanitized.get("recommendation") or {})
    recommended = [alias_to_id[a] for a in recommendation.get("recommended") or []
                   if a in alias_to_id]
    distinctions = []
    for claim in list(sanitized.get("accepted") or [])[:3]:
        distinctions.append({
            "code": claim["code"], "text": claim["observation"],
            "confidence": claim["confidence"],
            "targets": [{"file_id": alias_to_id[ref["frame"]], "region": ref["region"]}
                        for ref in claim["visual_refs"] if ref["frame"] in alias_to_id],
            "compare_to_file_ids": [alias_to_id[a] for a in claim["compared_to"]
                                    if a in alias_to_id],
            "cv_refs": list(claim["cv_refs"]),
        })
    attention_pointers = []
    for claim in list(sanitized.get("attention") or [])[:3]:
        attention_pointers.append({
            "code": claim["code"], "text": claim["observation"],
            "confidence": "low",
            "targets": [
                {"file_id": alias_to_id[ref["frame"]], "region": ref["region"]}
                for ref in claim["visual_refs"] if ref["frame"] in alias_to_id
            ],
            "compare_to_file_ids": [
                alias_to_id[alias] for alias in claim["compared_to"]
                if alias in alias_to_id
            ],
            "cv_refs": list(claim["cv_refs"]),
            "authority": "optional_attention_pointer_only",
        })
    return {
        "schema_version": SCHEMA_VERSION, "mode": MODE, "source": source,
        "model_id": model_id, "status": recommendation.get("status", "abstained"),
        "recommended_file_ids": recommended,
        "confidence": recommendation.get("confidence", "unknown"),
        "distinctions": distinctions, "max_distinctions": 3,
        "attention_pointers": attention_pointers, "max_attention_pointers": 3,
        "human_confirmation": "仅供人工审阅；不会改变既有 keeper、排序或删除状态。",
        "invariants": _invariants(1 if source == "vlm" else 0),
    }


def _cv_only(members: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    keepers = [int(row["file_id"]) for row in members if bool(row.get("is_keep"))]
    hashes = sorted({str(row.get("content_sha256")) for row in members})
    sanitized = {
        "recommendation": {"recommended": [], "confidence": "high", "status": "grounded"},
        "accepted": [], "rejected": [], "abstentions": [],
    }
    presentation = {
        "schema_version": SCHEMA_VERSION, "mode": MODE, "source": "cv_only",
        "model_id": None, "status": "grounded", "recommended_file_ids": keepers,
        "confidence": "high", "distinctions": [{
            "code": "BYTE_IDENTICAL", "text": "组内文件的持久化 SHA-256 完全相同；视觉内容无需 VLM 再判断。",
            "confidence": "high", "targets": [
                {"file_id": int(row["file_id"]), "region": "full_frame"} for row in members
            ], "compare_to_file_ids": [],
            "cv_refs": [f"{row['frame_alias']}:content_sha256" for row in members],
        }], "max_distinctions": 3,
        "attention_pointers": [], "max_attention_pointers": 3,
        "human_confirmation": "字节相同组直接使用决定性 CV；不会改变既有 keeper 或删除状态。",
        "invariants": _invariants(0),
    }
    sanitized["content_sha256"] = hashes
    return sanitized, presentation


def load_cached(output: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    path = cache_path(output, int(rows[0]["group_id"]))
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        return None
    if artifact.get("mode") != MODE or artifact.get("status") != "ready":
        return None
    if artifact.get("input_hash") != input_hash(rows):
        return None
    members, audit_context, derived_context, model_context = build_layers(rows)
    if artifact.get("alias_map") != vlm_identity.alias_map(members):
        return None
    if artifact.get("derived_context_sha256") != _sha_bytes(_canonical(derived_context)):
        return None
    route = route_group(members)
    if artifact.get("route") != route:
        return None
    if artifact.get("context_transformation_version") != CONTEXT_TRANSFORMATION_VERSION:
        return None
    if artifact.get("audit_context_sha256") != _sha_bytes(_canonical(audit_context)):
        return None
    if artifact.get("model_context_sha256") != _sha_bytes(_canonical(model_context)):
        return None
    if route == "cv_only":
        _sanitized, presentation = _cv_only(members)
        return presentation
    if artifact.get("prompt_sha256") != _sha_bytes(PROMPT.encode("utf-8")):
        return None
    model_id = artifact.get("model_id")
    raw = artifact.get("raw_judgment")
    if model_id not in ALLOWED_MODELS or not isinstance(raw, Mapping):
        return None
    if raw.get("schema_version") != 1 or int(raw.get("group_id", -1)) != int(members[0]["group_id"]):
        return None
    sanitized = sanitize_group(raw, [row["frame_alias"] for row in members],
                               model_context["evidence_refs"])
    return _project(sanitized, members, "vlm", str(model_id))


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10
        return ImageFont.load_default()


def _fit(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image, ImageOps

    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), (size[0] - 20, size[1] - 55))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, 50))
    return canvas


def _faces(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("faces_json")
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return [dict(face) for face in value if isinstance(face, Mapping)] if isinstance(value, list) else []


def _crop(image: Any, face: Mapping[str, Any]) -> Any | None:
    box = face.get("bbox")
    if not isinstance(box, list) or len(box) < 4:
        return None
    x, y, width, height = map(float, box[:4])
    pad = max(width, height) * 0.45
    bounds = (max(0, int(x - pad)), max(0, int(y - pad)),
              min(image.width, int(x + width + pad)),
              min(image.height, int(y + height + pad)))
    return image.crop(bounds) if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def prepare_visuals(output: Path, members: Sequence[Mapping[str, Any]],
                    sources: Mapping[int, Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    from PIL import Image, ImageDraw, ImageOps

    visual_plan = vlm_visual_budget.plan(members)
    group_id = int(members[0]["group_id"])
    root = Path(output) / CACHE_DIR / "assets" / str(group_id)
    root.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[dict[str, Any], Any]] = []
    for row in members:
        path = sources[int(row["file_id"])]
        stat = path.stat()
        size_matches = stat.st_size == int(row["size_bytes"])
        if not size_matches or _sha_file(path) != row["content_sha256"]:
            raise RuntimeError(f"source changed since inventory for file_id={row['file_id']}")
        with Image.open(path) as opened:
            loaded.append((dict(row), ImageOps.exif_transpose(opened).convert("RGB")))

    # A single sheet, with actual image pixels (not the padded tile) checked.
    assets: list[Path] = []
    columns = min(4, len(loaded))
    rows_n = (len(loaded) + columns - 1) // columns
    cell = (850, 850)
    sheet = Image.new("RGB", (columns * cell[0], rows_n * cell[1] + 70), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), f"GROUP {group_id} | full frames", fill="black", font=_font(28))
    tile_manifest = []
    for index, (row, image) in enumerate(loaded):
        fitted = ImageOps.contain(image, (cell[0] - 20, cell[1] - 55))
        if min(image.size) < 400 or min(fitted.size) < 400:
            raise vlm_visual_budget.VisualBudgetError("CONTACT_SHEET_SHORT_EDGE_BELOW_400")
        y, x = divmod(index, columns)
        sheet.paste(_fit(image, cell), (x * cell[0], 70 + y * cell[1]))
        draw.text((x * cell[0] + 15, 82 + y * cell[1]), row["frame_alias"],
                  fill="black", font=_font(28))
        tile_manifest.append({"identity": vlm_identity.identity(row),
                              "display_alias": row["frame_alias"],
                              "image_pixels": list(fitted.size)})
    sheet_path = root / "sheet-1.jpg"
    sheet.save(sheet_path, "JPEG", quality=90, optimize=True)
    assets.append(sheet_path)
    strip_manifests = []
    for strip_index, window in enumerate(visual_plan["windows"]):
        strip = Image.new("RGB", (640 * len(loaded), 640), "white")
        if strip.width > 8000:
            raise vlm_visual_budget.VisualBudgetError("STRIP_EXCEEDS_IMAGE_DIMENSION_BUDGET")
        cells = []
        for index, (row, image) in enumerate(loaded):
            box = (int(window[0] * image.width), int(window[1] * image.height),
                   int(window[2] * image.width), int(window[3] * image.height))
            crop = image.crop(box)
            fitted = ImageOps.contain(crop, (620, 585))
            if min(crop.size) < 256 or min(fitted.size) < 256:
                raise vlm_visual_budget.VisualBudgetError("CRITICAL_ROI_SHORT_EDGE_BELOW_256")
            strip.paste(_fit(crop, (640, 640)), (640 * index, 0))
            ImageDraw.Draw(strip).text((640 * index + 15, 10), row["frame_alias"],
                                      fill="black", font=_font(24))
            cells.append({"identity": vlm_identity.identity(row),
                          "image_pixels": list(fitted.size), "native_pixels": list(crop.size)})
        path = root / f"comparison-{strip_index + 1}.jpg"
        strip.save(path, "JPEG", quality=92, optimize=True)
        assets.append(path)
        strip_manifests.append(cells)

    manifest = []
    total = 0
    for path in assets:
        size = path.stat().st_size
        total += size
        with Image.open(path) as image:
            width, height = image.size
        manifest.append({"name": path.name, "sha256": _sha_file(path), "bytes": size,
                         "width": width, "height": height,
                         "visual_plan": visual_plan,
                         "cells": tile_manifest if path == assets[0]
                         else strip_manifests[assets.index(path) - 1]})
    if total > MAX_MODEL_BYTES:
        raise RuntimeError(f"visual payload exceeds {MAX_MODEL_BYTES} bytes")
    return assets, manifest


def _response_text(response: Mapping[str, Any]) -> str:
    parts = [block["text"] for block in response["output"]["message"]["content"]
             if isinstance(block, Mapping) and isinstance(block.get("text"), str)]
    text = "\n".join(parts).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def invoke_bedrock(model_id: str, group_id: int, assets: Sequence[Path],
                   model_context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_id not in ALLOWED_MODELS:
        raise ValueError(f"model is not an allowed Sonnet/Terra-class model: {model_id}")
    if model_context.get("layer") != "L2":
        raise ValueError("only L2 prompt-visible context may enter a model")
    import boto3
    from botocore.config import Config

    content: list[dict[str, Any]] = [{"text": PROMPT.replace("GROUP_ID", str(group_id))}]
    for path in assets:
        content.append({"text": f"VISUAL_ASSET {path.name}"})
        content.append({"image": {"format": "jpeg", "source": {"bytes": path.read_bytes()}}})
    assert_model_context_safe(model_context)
    content.append({
        "text": "MODEL_CONTEXT_JSON\n" + _canonical(model_context).decode("utf-8"),
    })
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          config=Config(retries={"max_attempts": 1, "mode": "standard"}))
    started = time.perf_counter()
    response = client.converse(modelId=model_id,
        messages=[{"role": "user", "content": content}], inferenceConfig={"maxTokens": 3000})
    parsed = json.loads(_response_text(response))
    metadata = {"latency_seconds": time.perf_counter() - started,
                "usage": response.get("usage"), "stop_reason": response.get("stopReason")}
    return parsed, metadata


def _write_artifact(output: Path, artifact: Mapping[str, Any]) -> Path:
    path = cache_path(output, int(artifact["group_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def process_group(output: Path, rows: Sequence[Mapping[str, Any]], sources: Mapping[int, Path],
                  *, model_id: str | None, prepare_only: bool = False,
                  invoke: Callable[[str, int, Sequence[Path], Mapping[str, Any]],
                                   tuple[dict[str, Any], dict[str, Any]]] = invoke_bedrock) -> dict[str, Any]:
    members, audit_context, derived_context, model_context = build_layers(rows)
    group_id = int(members[0]["group_id"])
    route = route_group(members)
    digest = input_hash(members)
    assets: list[Path] = []
    manifest: list[dict[str, Any]] = []
    model_meta: dict[str, Any] = {}
    raw: dict[str, Any] | None = None
    if route == "cv_only":
        sanitized, presentation = _cv_only(members)
        status = "ready"
    else:
        if model_id is None and not prepare_only:
            raise ValueError("--model-id is required for hard/high-similarity VLM groups")
        assets, manifest = prepare_visuals(output, members, sources)
        if prepare_only:
            sanitized, presentation, status = {}, {}, "prepared"
        else:
            assert_model_context_safe(model_context)
            raw, model_meta = invoke(model_id, group_id, assets, model_context)
            if raw.get("schema_version") != 1 or int(raw.get("group_id", -1)) != group_id:
                raise ValueError(f"VLM output failed schema/group gate for group {group_id}")
            sanitized = sanitize_group(raw, [row["frame_alias"] for row in members],
                                       model_context["evidence_refs"])
            presentation = _project(sanitized, members, "vlm", model_id)
            status = "ready"
    artifact = {
        "schema_version": SCHEMA_VERSION, "mode": MODE, "status": status,
        "group_id": group_id, "route": route, "input_hash": digest,
        "created_at": int(time.time()), "model_id": model_id if route == "vlm" else None,
        "context_transformation_version": CONTEXT_TRANSFORMATION_VERSION,
        "prompt_sha256": _sha_bytes(PROMPT.encode("utf-8")) if route == "vlm" else None,
        "audit_context_sha256": _sha_bytes(_canonical(audit_context)),
        "audit_context": audit_context,
        "alias_map": vlm_identity.alias_map(members),
        "derived_context_sha256": _sha_bytes(_canonical(derived_context)),
        "derived_context": derived_context,
        "model_context_sha256": _sha_bytes(_canonical(model_context)),
        "model_context": model_context,
        "visual_assets": manifest, "raw_judgment": raw,
        "validated_judgment": sanitized, "presentation": presentation,
        "model_metadata": model_meta, "invariants": _invariants(0 if route == "cv_only" or prepare_only else 1),
    }
    _write_artifact(output, artifact)
    return artifact


def _parse_ids(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except ValueError:
        value = [token for line in text.splitlines() for token in line.replace(",", " ").split()]
    if not isinstance(value, list):
        raise TypeError("group id file must be a JSON list or whitespace/comma-separated ids")
    return [int(item) for item in value]


def run_job(output: Path, *, group_ids: Sequence[int] = (), ids_file: Path | None = None,
            select: str = "hard", limit: int = DEFAULT_LIMIT, max_members: int = 12,
            model_id: str | None = None, prepare_only: bool = False, force: bool = False,
            invoke: Callable[[str, int, Sequence[Path], Mapping[str, Any]],
                             tuple[dict[str, Any], dict[str, Any]]] = invoke_bedrock) -> dict[str, Any]:
    output = Path(output).resolve()
    db_path = output / "inventory.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"inventory DB not found: {db_path}")
    if not 1 <= limit <= MAX_GROUPS:
        raise ValueError(f"limit must be between 1 and {MAX_GROUPS}")
    if not 2 <= max_members <= MAX_MEMBERS:
        raise ValueError(f"max_members must be between 2 and {MAX_MEMBERS}")
    requested = list(dict.fromkeys([int(gid) for gid in group_ids]
                                   + (_parse_ids(ids_file) if ids_file else [])))
    conn = sqlite3.connect(root_scope.readonly_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    scope = root_scope.recorded(conn)
    try:
        available = db.group_ids_in_scope(conn, scope=scope)
        candidates = requested or available
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"group ids are unavailable in this review scope: {missing}")
        selected: list[tuple[int, list[sqlite3.Row]]] = []
        for group_id in candidates:
            rows = db.group_page_by_id(conn, group_id, scope=scope)
            if not rows:
                continue
            if len(rows) > max_members:
                if requested:
                    raise ValueError(
                        f"group {group_id} has {len(rows)} members; --max-members is {max_members}"
                    )
                continue
            if requested or select == "all" or route_group(rows) == "vlm":
                selected.append((group_id, rows))
            if len(selected) >= limit:
                break
        if not selected:
            return {"processed": 0, "ready": 0, "prepared": 0, "skipped": 0, "groups": []}
        if any(route_group(rows) == "vlm" for _, rows in selected):
            if model_id is None and not prepare_only:
                raise ValueError("--model-id is required when selected groups route to VLM")
            if model_id is not None and model_id not in ALLOWED_MODELS:
                raise ValueError(f"model is not an allowed Sonnet/Terra-class model: {model_id}")
        results = []
        skipped = 0
        for group_id, rows in selected:
            if not force and load_cached(output, rows) is not None:
                skipped += 1
                results.append({"group_id": group_id, "status": "cached", "route": route_group(rows)})
                continue
            sources = {}
            for row in rows:
                record = db.get_file(conn, int(row["file_id"]), scope=scope)
                if record is None:
                    raise RuntimeError(f"file left review scope: {row['file_id']}")
                sources[int(row["file_id"])] = Path(record["path"])
            artifact = process_group(output, rows, sources, model_id=model_id,
                                     prepare_only=prepare_only, invoke=invoke)
            results.append({"group_id": group_id, "status": artifact["status"],
                            "route": artifact["route"]})
        return {"processed": len(results) - skipped,
                "ready": sum(row["status"] == "ready" for row in results),
                "prepared": sum(row["status"] == "prepared" for row in results),
                "skipped": skipped, "groups": results}
    finally:
        conn.close()


def report_contexts(
    output: Path,
    *,
    group_ids: Sequence[int] = (),
    ids_file: Path | None = None,
    select: str = "hard",
    limit: int = MAX_REPORT_GROUPS,
    max_members: int = MAX_MEMBERS,
) -> dict[str, Any]:
    """Report context bytes and family coverage without building visuals."""
    output = Path(output).resolve()
    db_path = output / "inventory.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"inventory DB not found: {db_path}")
    if not 1 <= limit <= MAX_REPORT_GROUPS:
        raise ValueError(f"report limit must be between 1 and {MAX_REPORT_GROUPS}")
    if not 2 <= max_members <= MAX_MEMBERS:
        raise ValueError(f"max_members must be between 2 and {MAX_MEMBERS}")
    requested = list(dict.fromkeys(
        [int(group_id) for group_id in group_ids]
        + (_parse_ids(ids_file) if ids_file else [])
    ))
    conn = sqlite3.connect(root_scope.readonly_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    scope = root_scope.recorded(conn)
    try:
        available = db.group_ids_in_scope(conn, scope=scope)
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"group ids are unavailable in this review scope: {missing}")
        reports = []
        for group_id in requested or available:
            rows = db.group_page_by_id(conn, group_id, scope=scope)
            if not rows or len(rows) > max_members:
                continue
            if requested or select == "all" or route_group(rows) == "vlm":
                reports.append(context_report(rows))
            if len(reports) >= limit:
                break
        return aggregate_context_reports(reports)
    finally:
        conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bounded review-only CV/VLM group judgments.")
    parser.add_argument("--output", type=Path, required=True,
                        help="existing review output containing inventory.sqlite")
    parser.add_argument("--group-id", type=int, action="append", default=[])
    parser.add_argument("--group-ids-file", type=Path)
    parser.add_argument("--select", choices=("hard", "all"), default="hard")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-members", type=int, default=12)
    parser.add_argument("--model-id", choices=sorted(ALLOWED_MODELS))
    parser.add_argument("--prepare-only", action="store_true",
                        help="write sheets/context but make no model call")
    parser.add_argument(
        "--report-context", action="store_true",
        help="report L0/L1/L2 bytes, L2 text gates and coverage; no visuals or model calls",
    )
    parser.add_argument("--force", action="store_true", help="replace a current matching cache")
    args = parser.parse_args(argv)
    if args.report_context:
        report = report_contexts(
            args.output, group_ids=args.group_id, ids_file=args.group_ids_file,
            select=args.select, limit=args.limit, max_members=args.max_members,
        )
    else:
        report = run_job(
            args.output, group_ids=args.group_id, ids_file=args.group_ids_file,
            select=args.select, limit=args.limit, max_members=args.max_members,
            model_id=args.model_id, prepare_only=args.prepare_only, force=args.force,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
