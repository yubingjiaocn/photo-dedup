"""P0 selective decision policy: automate only safe, explainable removals."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from . import exposure

PROFILES = {
    "conservative": {"margin_min": 0.14, "exposure_review": "UNKNOWN"},
    "balanced": {"margin_min": 0.08, "exposure_review": "MAYBE"},
    "aggressive": {"margin_min": 0.04, "exposure_review": "MAYBE"},
}
VALID_DECISIONS = {"KEEP", "AUTO_REMOVE", "MAYBE", "UNKNOWN"}
POLICY_VERSION = "p0-safety-v2"


def parse_meta(raw: str | None) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _critical_available(member: Mapping[str, Any]) -> bool:
    meta = parse_meta(member.get("quality_meta"))
    exp = meta.get("exposure")
    return (
        member.get("phash") is not None
        and member.get("dinov2_embedding") is not None
        and member.get("quality_score") is not None
        and isinstance(exp, dict)
        and "clip_hi" in exp and "anchor_mass" in exp
    )


def _exposure(member: Mapping[str, Any]) -> tuple[str, str, float]:
    exp = parse_meta(member.get("quality_meta")).get("exposure")
    if not isinstance(exp, dict):
        return "unknown", "FEATURE_MISSING: exposure", 1.0
    return exposure.classify(exp)


def decide_group(
    members: Sequence[Mapping[str, Any]], keeper: int, scores: Mapping[int, float],
    group_type: str, *, profile: str = "balanced",
    group_trusted: bool = True,
    safe_duplicates: Mapping[int, bool] | None = None,
) -> Dict[str, Any]:
    """Decide each member; keeper and safe_duplicates keys are local indices."""
    policy = PROFILES.get(profile)
    if policy is None:
        raise ValueError(f"unknown decision profile: {profile}")
    decisions: Dict[int, Dict[str, Any]] = {}
    decisions[keeper] = _record("KEEP", 1.0, "GROUP_KEEPER", scores[keeper], 0.0)
    keeper_exp, _, _ = _exposure(members[keeper])
    direct_duplicates = safe_duplicates or {}
    for idx, member in enumerate(members):
        if idx == keeper:
            continue
        margin = float(scores[keeper] - scores[idx])
        exp_state, exp_reason, _severity = _exposure(member)
        if not _critical_available(member):
            decisions[idx] = _record("UNKNOWN", 0.0, "FEATURE_MISSING", scores[idx], margin)
        elif direct_duplicates.get(idx, False):
            # Byte-identical (SHA-256 match) - only automatic removal path
            decisions[idx] = _record("AUTO_REMOVE", 1.0, "BYTE_IDENTICAL", scores[idx], margin)
        elif not group_trusted:
            # Non-SHA visual groups that failed trust still preserve their specific reason
            if group_type == "phash_near":
                decisions[idx] = _record("MAYBE", 0.0, "PHASH_NEAR_UNTRUSTED", scores[idx], margin)
            elif group_type == "similar_scene":
                decisions[idx] = _record("MAYBE", 0.0, "SIMILAR_SCENE_UNTRUSTED", scores[idx], margin)
            else:
                decisions[idx] = _record("MAYBE", 0.0, "GROUP_IMPURE", scores[idx], margin)
        elif group_type == "sha_exact":
            # SHA-exact group but this specific pair doesn't share SHA - should not happen
            # if logic is correct, but handle defensively
            decisions[idx] = _record("MAYBE", 0.0, "SHA_GROUP_NO_MATCH", scores[idx], margin)
        elif group_type == "phash_near":
            # pHash near-duplicate without byte identity - requires review
            decisions[idx] = _record("MAYBE", 0.0, "PHASH_NEAR_DUP_ONLY", scores[idx], margin)
        elif group_type == "similar_scene":
            decisions[idx] = _record("MAYBE", 0.0, "SUBJECTIVE_ONLY: similar_scene", scores[idx], margin)
        elif _face_conflict(members[keeper], member):
            decisions[idx] = _record("MAYBE", 0.0, "FACE_COUNT_MISMATCH", scores[idx], margin)
        elif exp_state == "reject" and keeper_exp != "reject":
            review_state = policy["exposure_review"]
            decisions[idx] = _record(review_state, 0.0,
                                     f"EXPOSURE_UNCALIBRATED_{profile.upper()}: {exp_reason}",
                                     scores[idx], margin)
        elif margin < policy["margin_min"]:
            decisions[idx] = _record("MAYBE", 0.0, "LOW_MARGIN", scores[idx], margin)
        elif exp_state == "maybe":
            decisions[idx] = _record("MAYBE", 0.0, f"EXPOSURE_BORDERLINE: {exp_reason}", scores[idx], margin)
        else:
            decisions[idx] = _record("MAYBE", 0.0, "RELATIVE_ONLY", scores[idx], margin)
    # Non-bypassable group floor: exactly/at least one explicit keeper.
    assert any(v["decision"] == "KEEP" for v in decisions.values())
    assert sum(v["decision"] == "AUTO_REMOVE" for v in decisions.values()) <= len(members) - 1
    state = "REVIEW_REQUIRED" if any(v["decision"] in {"MAYBE", "UNKNOWN"} for v in decisions.values()) else "AUTO_READY"
    return {"members": decisions, "state": state, "profile": profile, "policy_version": POLICY_VERSION}


def _record(decision: str, confidence: float, reason: str, score: float, margin: float) -> Dict[str, Any]:
    return {
        "decision": decision, "confidence": float(confidence), "reason": reason,
        "evidence": {"utility_score": float(score), "pair_margin": float(margin)},
    }


def _face_conflict(keeper: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    ka, ca = int(keeper.get("face_count") or 0), int(candidate.get("face_count") or 0)
    return ka != ca or ka >= 2 or ca >= 2


def detector_extensions() -> Dict[str, Any]:
    """Reserved extension points; unvalidated detectors remain disabled/unknown."""
    return {
        "closed_eyes": {"enabled": False, "state": "unknown", "interface": "Detector.detect(image, faces)"},
        "ofiq": {"enabled": False, "state": "unknown", "interface": "Detector.score(image, faces)"},
    }
