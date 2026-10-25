"""Validate evidence-bound VLM photo judgments before review presentation.

The VLM is the group-level synthesizer: it may combine the images with every
available CV observation, including shadow-only signals.  This module does not
trust free-form rationales.  Every surfaced discriminator must point to a
visible frame/region, a supplied CV field, or both.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

ALLOWED_CODES = {
    "EXPRESSION_DIFFERENCE",
    "POSE_ACTION_DIFFERENCE",
    "CAMERA_ENGAGEMENT",
    "SCENE_INFORMATION_DIFFERENCE",
    "UNIQUE_MOMENT",
    "EYE_STATE_DIFFERENCE",
    "FACE_QUALITY_DIFFERENCE",
    "TECHNICAL_QUALITY_DIFFERENCE",
    "EXPOSURE_DIFFERENCE",
    "FRAMING_DIFFERENCE",
    "COMPOSITE_ADVANTAGE",
    "NO_RELIABLE_DIFFERENCE",
}
ALLOWED_CONFIDENCE = {"high", "medium"}
OBSERVATION_CONFIDENCE = ALLOWED_CONFIDENCE | {"low"}
ALLOWED_REGIONS = {"full_frame", "face", "subject", "background"}
MAX_REGION_LABEL = 120
SENSITIVE = re.compile(
    r"\b(identity|identify|age|relationship|personality|diagnosis|ethnicity|"
    r"religion|sexuality|politics|internal emotion)\b",
    re.IGNORECASE,
)


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _valid_refs(refs: list[str] | None, allowed: set[str]) -> bool:
    return refs is not None and bool(refs) and set(refs) <= allowed


def _ref_aliases(ref: str, allowed_aliases: set[str]) -> set[str]:
    parts = ref.split(":")
    if parts[:1] == ["PAIR"] and len(parts) >= 4:
        return {alias for alias in parts[1:3] if alias in allowed_aliases}
    return {parts[0]} if parts and parts[0] in allowed_aliases else set()


def sanitize_group(group: Mapping[str, Any], aliases: Sequence[str],
                   cv_evidence_refs: Sequence[str] = ()) -> dict[str, Any]:
    """Fail closed on unsupported, ungrounded, or low-confidence judgments."""
    allowed_aliases = set(aliases)
    allowed_cv = set(cv_evidence_refs)
    raw_judgment = group.get("judgment")
    judgment = dict(raw_judgment) if isinstance(raw_judgment, Mapping) else {}
    selected = _string_list(judgment.get("recommended"))
    selected_valid = (
        selected is not None and bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= allowed_aliases
    )

    accepted: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    raw_claims = judgment.get("discriminators")
    legacy_observer = not isinstance(raw_claims, list)
    if legacy_observer:
        # Backward-compatible ingestion of the first observer probe.  New calls
        # should always emit judgment.discriminators with explicit grounding.
        raw_claims = group.get("semantic_claims") or []

    for index, raw in enumerate(raw_claims):
        claim = dict(raw) if isinstance(raw, Mapping) else {}
        reasons: list[str] = []
        code = claim.get("code")
        subjects = _string_list(claim.get("subjects"))
        compared = _string_list(claim.get("compared_to"))
        observation = claim.get("observation") or claim.get("reason")
        confidence = claim.get("confidence")
        visual_refs = claim.get("visual_refs")
        cv_refs = _string_list(claim.get("cv_refs"))

        if code not in ALLOWED_CODES:
            reasons.append("CODE_NOT_ALLOWED")
        if confidence not in OBSERVATION_CONFIDENCE:
            reasons.append("INVALID_CONFIDENCE")
        if subjects is None or not subjects or not set(subjects) <= allowed_aliases:
            reasons.append("INVALID_SUBJECT_ALIAS")
        if compared is None or not set(compared) <= allowed_aliases:
            reasons.append("INVALID_COMPARED_ALIAS")
        if not isinstance(observation, str) or not observation.strip() or len(observation) > 500:
            reasons.append("INVALID_OBSERVATION_TEXT")
        elif SENSITIVE.search(observation):
            reasons.append("SENSITIVE_OR_UNSUPPORTED_INFERENCE")

        clean_visual: list[dict[str, str]] = []
        if visual_refs is None and legacy_observer and subjects:
            clean_visual = [{"frame": frame, "region": "full_frame"}
                            for frame in subjects if frame in allowed_aliases]
        elif visual_refs is not None:
            if not isinstance(visual_refs, list):
                reasons.append("INVALID_VISUAL_REFS")
            else:
                for ref in visual_refs:
                    item = dict(ref) if isinstance(ref, Mapping) else {}
                    region = item.get("region")
                    if (item.get("frame") not in allowed_aliases
                            or not isinstance(region, str) or not region.strip()
                            or len(region) > MAX_REGION_LABEL or SENSITIVE.search(region)):
                        reasons.append("INVALID_VISUAL_REFS")
                        break
                    clean_visual.append({
                        "frame": item["frame"],
                        "region": region.strip(),
                        "region_kind": region if region in ALLOWED_REGIONS else "described_region",
                    })
        if cv_refs is not None and not set(cv_refs) <= allowed_cv:
            reasons.append("UNKNOWN_CV_REFERENCE")
        if not clean_visual and not _valid_refs(cv_refs, allowed_cv):
            reasons.append("UNGROUNDED_DISCRIMINATOR")

        if reasons:
            rejected.append({
                "claim_index": index,
                "reasons": list(dict.fromkeys(reasons)),
                "raw": claim,
            })
            continue
        clean_claim = {
            "code": code,
            "subjects": subjects,
            "compared_to": compared,
            "observation": observation.strip(),
            "confidence": confidence,
            "confidence_kind": "uncalibrated_model_judgment",
            "visual_refs": clean_visual,
            "cv_refs": cv_refs or [],
            "authority": (
                "optional_attention_pointer_only"
                if confidence == "low" else "human_review_recommendation_only"
            ),
        }
        if confidence == "low":
            attention.append(clean_claim)
        else:
            accepted.append(clean_claim)

    accepted = accepted[:3]
    attention = attention[:3]
    grounded_recommendations: set[str] = set()
    for claim in accepted:
        grounded = {ref["frame"] for ref in claim["visual_refs"]}
        for ref in claim["cv_refs"]:
            grounded.update(_ref_aliases(ref, allowed_aliases))
        grounded_recommendations.update(set(claim["subjects"]) & grounded)
    recommendation_rejections = []
    if selected is None:
        recommendation_rejections.append("INVALID_RECOMMENDATION_LIST")
    elif not selected:
        recommendation_rejections.append("EMPTY_RECOMMENDATION")
    elif len(selected) != len(set(selected)):
        recommendation_rejections.append("DUPLICATE_RECOMMENDATION")
    elif not set(selected) <= allowed_aliases:
        recommendation_rejections.append("INVALID_RECOMMENDATION_ALIAS")
    elif not set(selected) <= grounded_recommendations:
        recommendation_rejections.append("UNGROUNDED_RECOMMENDATION")
    recommendation_ok = selected_valid and not recommendation_rejections
    recommendation = {
        "recommended": selected if recommendation_ok else [],
        "confidence": judgment.get("confidence")
        if judgment.get("confidence") in ALLOWED_CONFIDENCE else "unknown",
        "status": "grounded" if recommendation_ok and accepted else "abstained",
    }
    if recommendation_rejections:
        recommendation["rejections"] = recommendation_rejections
    return {
        "group_id": group.get("group_id"),
        "recommendation": recommendation,
        "accepted": accepted,
        "attention": attention,
        "rejected": rejected,
        "abstentions": list(group.get("abstentions") or []),
        "invariants": {
            "changes_keeper": False,
            "changes_sorting": False,
            "changes_routing": False,
            "deletion_authority": "none",
        },
    }
