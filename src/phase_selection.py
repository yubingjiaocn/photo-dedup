"""Deterministic logical-phase protection and explainable keeper selection.

This module consumes cached metadata only. It never opens source media, never
physically splits a database group, and never creates deletion authority.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from . import cluster_layers as CL
from . import quality as Q
from .phase_coverage import diagnose_phase_coverage, diagnostic_review_reasons, keeper_budget
from .pose_evidence import protect_pose_variants


@dataclass(frozen=True)
class Phase:
    phase_id: int
    members: tuple[int, ...]
    boundary_reasons: tuple[str, ...]
    uncertain: bool
    uncertainty_reasons: tuple[str, ...] = ()


def _meta(member: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(member.get("quality_meta") or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _embedding(member: Mapping[str, Any]) -> np.ndarray | None:
    blob = member.get("dinov2_embedding")
    if not blob:
        return None
    try:
        value = Q.blob_to_embedding(blob)
    except (ValueError, TypeError):
        return None
    norm = float(np.linalg.norm(value))
    if value.size == 0 or not math.isfinite(norm) or norm == 0:
        return None
    return value / norm


def _subject_center(member: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return an explicit subject center or a real detected-face center.

    The production extractor does not emit a generic subject center. A dominant
    face is useful evidence when present, but is not described as a proxy for a
    non-face subject.
    """
    meta = _meta(member)
    center = meta.get("subject_center")
    if isinstance(center, list) and len(center) == 2:
        try:
            x, y = float(center[0]), float(center[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                return x, y
        except (ValueError, TypeError):
            pass
    faces = CL.parse_faces(member.get("faces_json"))
    return CL.dominant_face_center(
        faces, int(member.get("width") or 0), int(member.get("height") or 0), 0.6
    )


def segment_phases(
    members: Sequence[Mapping[str, Any]], *, max_gap_seconds: int = 4,
    embedding_boundary: float = 0.93, position_shift: float = 0.22,
    group_type: str = "burst", minimum_phase_members: int = 2,
) -> list[Phase]:
    """Annotate robust logical phases inside one existing database group.

    Sparse photo bursts are not edited video. A boundary therefore needs either
    a strong signal or locally exceptional embedding change with independent
    support. Adjacent candidate cuts are peak-suppressed so ordinary detector
    flicker cannot manufacture one-frame phases. Hard/strong endpoint evidence
    may still preserve a meaningful singleton.
    """
    if not members:
        return []
    order = sorted(range(len(members)), key=lambda i: (
        members[i].get("exif_timestamp") is None,
        members[i].get("exif_timestamp") or 0,
        int(members[i].get("id") or i),
    ))
    evidence: list[dict[str, Any]] = []
    for left, right in pairwise(order):
        a, b = members[left], members[right]
        reasons: list[str] = []
        uncertainty: set[str] = set()
        ta, tb = a.get("exif_timestamp"), b.get("exif_timestamp")
        gap = None
        if ta is None or tb is None:
            uncertainty.add("TIME_MISSING")
        else:
            gap = int(tb) - int(ta)
            if gap > max_gap_seconds:
                reasons.append("TIME_GAP")
        ea, eb = _embedding(a), _embedding(b)
        similarity = None
        if ea is None or eb is None or ea.shape != eb.shape:
            uncertainty.add("EMBEDDING_MISSING")
        else:
            similarity = float(np.dot(ea, eb))
        ca, cb = _subject_center(a), _subject_center(b)
        position_delta = None
        if ca is not None and cb is not None:
            position_delta = max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1]))
            if position_delta > position_shift:
                reasons.append("DETECTED_SUBJECT_POSITION_CHANGE")
        elif int(a.get("face_count") or 0) > 0 or int(b.get("face_count") or 0) > 0:
            uncertainty.add("FACE_POSITION_MISSING")
        face_a, face_b = int(a.get("face_count") or 0), int(b.get("face_count") or 0)
        evidence.append({
            "left": left, "right": right, "similarity": similarity,
            "position_delta": position_delta, "face_delta": abs(face_a - face_b),
            "max_face_count": max(face_a, face_b), "gap": gap,
            "reasons": reasons, "uncertainty": uncertainty,
        })

    similarities = [float(item["similarity"]) for item in evidence
                    if item["similarity"] is not None]
    local_baseline = float(np.median(similarities)) if similarities else None
    accepted: dict[int, list[str]] = {}
    strong: set[int] = set()
    for edge, item in enumerate(evidence):
        similarity = item["similarity"]
        reasons = list(item["reasons"])
        embedding_exception = bool(
            similarity is not None and similarity < embedding_boundary
            and (local_baseline is None or similarity < local_baseline - 0.035)
        )
        strong_embedding = bool(
            similarity is not None
            and similarity < min(0.88, embedding_boundary - 0.05)
        )
        # In perceptual-near stage bursts, a sustained sub-.955 transition can
        # still represent a real pose/tableau change even when face position is
        # stable. It remains subject to minimum-run suppression below.
        phash_tableau_change = bool(
            group_type == "phash_near" and similarity is not None
            and similarity < 0.955
        )
        position_change = "DETECTED_SUBJECT_POSITION_CHANGE" in reasons
        moderate_position = bool(
            item["position_delta"] is not None and item["position_delta"] > 0.05
        )
        low_face_context = item["max_face_count"] <= 3
        face_support = item["face_delta"] >= 1 and low_face_context
        face_strong = item["face_delta"] >= 2 and low_face_context
        time_gap = "TIME_GAP" in reasons
        if embedding_exception:
            reasons.append("LOCAL_BASELINE_EMBEDDING_CHANGE")
        consensus = embedding_exception and (position_change or face_support)
        sparse_action_change = bool(
            group_type == "burst" and embedding_exception
            and similarity is not None and similarity < 0.90 and moderate_position
        )
        strong_single = time_gap or strong_embedding or phash_tableau_change or sparse_action_change or bool(
            embedding_exception and face_strong
            and similarity is not None and similarity < 0.91
        )
        if group_type == "phash_near":
            accept = time_gap or strong_embedding or phash_tableau_change or bool(
                embedding_exception and position_change
            )
        else:
            accept = strong_single or consensus
        if accept and len(members) >= 3:
            accepted[edge] = list(dict.fromkeys(reasons))
            if strong_single:
                strong.add(edge)

    def strength(edge: int) -> tuple[int, float]:
        similarity = evidence[edge]["similarity"]
        return edge in strong, 1.0 - float(similarity) if similarity is not None else 0.0

    # Minimum segment length / non-maximum suppression. Remove the weaker of
    # two cuts surrounding a singleton. Endpoint singletons survive only with
    # strong evidence, preserving real sparse action states.
    changed = True
    while changed and accepted:
        changed = False
        boundaries = [-1, *sorted(accepted), len(order) - 1]
        for pos in range(1, len(boundaries)):
            if boundaries[pos] - boundaries[pos - 1] >= minimum_phase_members:
                continue
            left_edge = boundaries[pos - 1] if boundaries[pos - 1] >= 0 else None
            right_edge = boundaries[pos] if boundaries[pos] < len(order) - 1 else None
            candidates = [edge for edge in (left_edge, right_edge) if edge is not None]
            removable = [edge for edge in candidates if edge not in strong]
            if not removable:
                continue
            remove = min(removable, key=strength) if len(candidates) == 2 else removable[0]
            del accepted[remove]
            changed = True
            break

    phases: list[Phase] = []
    current = [order[0]]
    boundary_reasons: list[str] = ["LOGICAL_PHASE_START"]
    uncertainty: set[str] = set()
    for edge, item in enumerate(evidence):
        right = item["right"]
        pair_uncertainty = set(item["uncertainty"])
        if edge in accepted:
            phases.append(Phase(len(phases), tuple(current), tuple(boundary_reasons),
                                bool(uncertainty), tuple(sorted(uncertainty))))
            current = [right]
            boundary_reasons = accepted[edge]
            uncertainty = pair_uncertainty
        else:
            current.append(right)
            uncertainty.update(pair_uncertainty)
    phases.append(Phase(len(phases), tuple(current), tuple(boundary_reasons),
                        bool(uncertainty), tuple(sorted(uncertainty))))
    return phases

def _bounded(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


_UTILITY_WEIGHTS = {
    "quality": 0.40, "face_clarity": 0.25, "exposure": 0.15,
    "subject_completeness": 0.12, "occlusion_inverse": 0.08,
}


def _score(member: Mapping[str, Any]) -> tuple[float, dict[str, float | None], list[str]]:
    """Return one authoritative, availability-normalised utility score.

    Optional fields are omitted, not filled with invented neutral proxies. The
    remaining declared weights are renormalised. This keeps ranking useful on
    the production schema while exposing exactly which evidence was absent.
    """
    meta = _meta(member)
    quality = _bounded((float(member["quality_score"]) / 100.0)
                       if member.get("quality_score") is not None else None)
    face_count = int(member.get("face_count") or 0)
    face = _bounded(meta.get("face_quality")) if face_count > 0 else None
    exposure = meta.get("exposure") if isinstance(meta.get("exposure"), dict) else None
    exposure_ok = None
    if exposure is not None:
        try:
            exposure_ok = _bounded(1.0 - min(
                1.0, float(exposure["clip_hi"]) + float(exposure["clip_lo"])
            ))
        except (KeyError, ValueError, TypeError):
            exposure_ok = None
    completeness = _bounded(meta.get("subject_completeness"))
    occlusion = _bounded(meta.get("occlusion"))
    evidence: dict[str, float | None] = {
        "quality": quality,
        "face_clarity": face,
        "exposure": exposure_ok,
        "subject_completeness": completeness,
        "occlusion_inverse": None if occlusion is None else 1.0 - occlusion,
    }
    weights = _UTILITY_WEIGHTS
    available = {key: value for key, value in evidence.items() if value is not None}
    denominator = sum(weights[key] for key in available)
    total = (sum(weights[key] * value for key, value in available.items()) / denominator
             if denominator else 0.0)
    missing = [f"{key.upper()}_MISSING" for key, value in evidence.items() if value is None]
    return total, evidence, missing


def _score_group(
    members: Sequence[Mapping[str, Any]], score_policy: str,
    score_change_margin: float = 0.06,
) -> tuple[dict[int, float], dict[str, Any]]:
    if not isinstance(score_policy, str) or score_policy not in {
        "per_member", "common_evidence", "guarded_common",
    }:
        raise ValueError("score_policy must be per_member, common_evidence or guarded_common")
    if (isinstance(score_change_margin, bool)
            or not isinstance(score_change_margin, (int, float))
            or not math.isfinite(score_change_margin) or not 0 <= score_change_margin <= 1):
        raise ValueError("score_change_margin must be a finite number in [0, 1]")
    scored = [_score(member) for member in members]
    legacy = {idx: item[0] for idx, item in enumerate(scored)}
    shared = [key for key in _UTILITY_WEIGHTS
              if scored and all(item[1][key] is not None for item in scored)]
    excluded = [key for key in _UTILITY_WEIGHTS if key not in shared
                and any(item[1][key] is not None for item in scored)]
    context = {
        "policy": score_policy, "shared_components": shared,
        "excluded_noncommon_components": excluded,
        "effective_policy": "per_member", "comparison_only": True,
        "required_primary_gain": score_change_margin, "primary_gain": None,
        "fallback": None if shared or not scored else "NO_SHARED_COMPONENTS_LEGACY_UNCHANGED",
        "semantics": "shared_observed_components_not_subject_or_face_truth",
    }
    if score_policy == "per_member" or not shared:
        return legacy, context
    denominator = sum(_UTILITY_WEIGHTS[key] for key in shared)
    scores = {idx: sum(_UTILITY_WEIGHTS[key] * item[1][key] for key in shared) / denominator
              for idx, item in enumerate(scored)}
    if score_policy == "guarded_common":
        def rank(idx: int, values: Mapping[int, float]) -> tuple[float, int, int]:
            return (values[idx], int(members[idx].get("size_bytes") or 0),
                    -int(members[idx].get("id") or idx))
        old_anchor = max(legacy, key=lambda idx: rank(idx, legacy))
        new_anchor = max(scores, key=lambda idx: rank(idx, scores))
        gain = scores[new_anchor] - scores[old_anchor]
        context.update({"primary_gain": gain, "primary_before_member_index": old_anchor,
                        "primary_after_member_index": new_anchor})
        if old_anchor == new_anchor or gain < score_change_margin:
            context["fallback"] = ("NO_PRIMARY_RANK_CHANGE" if old_anchor == new_anchor
                                   else "INSUFFICIENT_PRIMARY_GAIN")
            return legacy, context
    context.update({"effective_policy": "common_evidence", "comparison_only": False})
    return scores, context


def utility_scores(
    members: Sequence[Mapping[str, Any]], *, score_policy: str = "per_member",
    score_change_margin: float = 0.06,
) -> dict[int, float]:
    """Expose actual scores; common_evidence compares only shared observations.

    Missing face/optional detections are neither a reward nor a quality penalty.
    A group with no shared observations explicitly retains the legacy ranking.
    guarded_common changes the scoring basis only when the best primary
    candidate changes with at least score_change_margin observed utility gain.
    Shared/excluded context fields describe that comparison; effective_policy
    identifies the score map actually used. No subject identity, frontality or
    completeness is inferred by either policy.
    """
    return _score_group(members, score_policy, score_change_margin)[0]


def _phase_variation(
    members: Sequence[Mapping[str, Any]], phase: Phase, diversity_similarity: float,
) -> tuple[bool, dict[str, Any]]:
    embeddings = [(idx, _embedding(members[idx])) for idx in phase.members]
    valid = [(idx, emb) for idx, emb in embeddings if emb is not None]
    min_similarity = None
    if len(valid) >= 2:
        min_similarity = min(
            float(np.dot(left[1], right[1]))
            for pos, left in enumerate(valid) for right in valid[pos + 1:]
        )
    face_counts = {int(members[idx].get("face_count") or 0) for idx in phase.members}
    timestamps = [members[idx].get("exif_timestamp") for idx in phase.members]
    valid_times = [int(value) for value in timestamps if value is not None]
    span = max(valid_times) - min(valid_times) if len(valid_times) >= 2 else 0
    variable = len(phase.members) >= 3 and (
        (min_similarity is not None and min_similarity < diversity_similarity)
        or len(face_counts) > 1
    )
    return variable, {
        "min_pair_similarity": min_similarity,
        "face_count_states": sorted(face_counts),
        "time_span_seconds": span,
        "valid_embedding_count": len(valid),
    }


def select_phase_keepers(
    members: Sequence[Mapping[str, Any]], phases: Sequence[Phase], *,
    keepers_per_phase: int = 1, max_group_keepers: int = 3,
    diversity_similarity: float = 0.965, mmr_quality_weight: float = 0.7,
    phase_requirements: Mapping[str, Any] | None = None,
    score_policy: str = "per_member", score_change_margin: float = 0.06,
    diversity_policy: str = "mmr", diversity_quality_slack: float = 0.04,
    pose_policy: str = "off", pose_displacement_threshold: float = 0.4,
    local_quality_policy: str = "off", local_quality_similarity: float = 0.9,
    local_quality_penalty: float = 0.08,
    instance_recovery_policy: str = "off",
) -> dict[str, Any]:
    """Protect logical phases with variation-aware budget and deterministic MMR.

    Size never grants a keeper. At most one extra variation keeper and one
    primary-signal uncertainty keeper are proposed per logical phase, then a
    group-level cap prevents detector flicker from recreating keep-all. The cap
    limits the budget; it does not create it. Optional scoring evidence only
    raises review, never keep-all. The explicit consensus-pose policy may
    protect one extra contrasting pose in a small group, always for review.
    """
    if not isinstance(instance_recovery_policy, str) or instance_recovery_policy not in {"off", "review_only"}:
        raise ValueError("instance_recovery_policy must be off or review_only")
    if not isinstance(local_quality_policy, str) or local_quality_policy not in {"off", "dominance", "region_set"}:
        raise ValueError("local_quality_policy must be off, dominance or region_set")
    from .local_quality import _finite, adjust_scores
    if not _finite(local_quality_similarity, 0, 1) or not _finite(local_quality_penalty, 0, 0.25):
        raise ValueError("invalid local quality similarity/penalty")
    if not isinstance(pose_policy, str) or pose_policy not in {"off", "consensus", "stable_actor"}:
        raise ValueError("pose_policy must be off, consensus or stable_actor")
    if (isinstance(pose_displacement_threshold, bool)
            or not isinstance(pose_displacement_threshold, (int, float))
            or not math.isfinite(pose_displacement_threshold) or pose_displacement_threshold <= 0):
        raise ValueError("pose_displacement_threshold must be positive and finite")
    if not isinstance(diversity_policy, str) or diversity_policy not in {"mmr", "quality_banded"}:
        raise ValueError("diversity_policy must be mmr or quality_banded")
    if (isinstance(diversity_quality_slack, bool)
            or not isinstance(diversity_quality_slack, (int, float))
            or not 0 <= diversity_quality_slack <= 1):
        raise ValueError("diversity_quality_slack must be finite and in [0, 1]")
    if max_group_keepers < 1:
        raise ValueError("max_group_keepers must be >= 1")
    if not 0.0 <= mmr_quality_weight <= 1.0:
        raise ValueError("mmr_quality_weight must be in [0, 1]")
    selections: list[dict[str, Any]] = []
    all_keepers: list[int] = []
    scores, scoring_context = _score_group(members, score_policy, score_change_margin)
    local_context = None
    if local_quality_policy == "dominance":
        scores, local_context = adjust_scores(members, phases, scores,
            similarity=local_quality_similarity, penalty=local_quality_penalty)
    if local_quality_policy == "region_set":
        from .region_set_quality import adjust_scores as adjust_set_scores
        scores, local_context = adjust_set_scores(members, phases, scores,
            similarity=local_quality_similarity, penalty=local_quality_penalty)
    local_changed = bool(local_context and local_context["changed_members"])
    for phase in phases:
        target = max(1, keepers_per_phase)
        reasons = ["LOGICAL_PHASE_MINIMUM_KEEPER"]
        variable, variation = _phase_variation(members, phase, diversity_similarity)
        if variable and len(phase.members) > target:
            target += 1
            reasons.append("OBSERVED_VARIATION_EXTRA_KEEPER")
        critical_uncertainty = bool(set(phase.uncertainty_reasons) & {
            "TIME_MISSING", "EMBEDDING_MISSING", "FACE_POSITION_MISSING",
        })
        if critical_uncertainty and len(phase.members) > target:
            target += 1
            reasons.append("PRIMARY_BOUNDARY_EVIDENCE_EXTRA_KEEPER")
        scored = {idx: _score(members[idx]) for idx in phase.members}
        optional_missing = any(
            any(code in {"SUBJECT_COMPLETENESS_MISSING", "OCCLUSION_INVERSE_MISSING"}
                for code in scored[idx][2])
            for idx in phase.members
        )
        ranked_utilities = sorted((scores[idx] for idx in phase.members), reverse=True)
        utility_margin = (ranked_utilities[0] - ranked_utilities[1]
                          if len(ranked_utilities) > 1 else 1.0)
        # Missing completeness/occlusion is ubiquitous in production. Review it
        # only where it can plausibly change a face-bearing, close-score choice.
        optional_gap_review = optional_missing and utility_margin < 0.08 and any(
            int(members[idx].get("face_count") or 0) > 0 for idx in phase.members
        )
        if optional_missing:
            reasons.append("OPTIONAL_SCORING_EVIDENCE_MISSING")
        if optional_gap_review:
            reasons.append("OPTIONAL_EVIDENCE_LOW_MARGIN_REVIEW")
        candidates = sorted(phase.members, key=lambda idx: (
            scores[idx], int(members[idx].get("size_bytes") or 0),
            -int(members[idx].get("id") or idx),
        ), reverse=True)
        chosen: list[int] = []
        selection_steps: list[dict[str, Any]] = []
        while candidates and len(chosen) < min(target, len(phase.members)):
            if not chosen:
                pick = candidates.pop(0)
                selection_steps.append({"member": pick, "method": "utility", "score": scores[pick]})
            else:
                ranks: dict[int, tuple[float, float, int, int]] = {}
                details: dict[int, tuple[float, float]] = {}
                for idx in candidates:
                    emb = _embedding(members[idx])
                    similarities = []
                    for selected in chosen:
                        other = _embedding(members[selected])
                        if emb is not None and other is not None and emb.shape == other.shape:
                            similarities.append(float(np.dot(emb, other)))
                    max_similarity = max(similarities) if similarities else 0.0
                    mmr_score = (
                        mmr_quality_weight * scores[idx]
                        - (1.0 - mmr_quality_weight) * max_similarity
                    )
                    ranks[idx] = (
                        mmr_score, scores[idx], int(members[idx].get("size_bytes") or 0),
                        -int(members[idx].get("id") or idx),
                    )
                    details[idx] = (mmr_score, max_similarity)
                pick = max(candidates, key=ranks.__getitem__)
                method = "mmr"
                if diversity_policy == "quality_banded":
                    embeddings = [_embedding(members[i]) for i in [*chosen, *candidates]]
                    comparable = all(e is not None and e.shape == embeddings[0].shape
                                     for e in embeddings) if embeddings[0] is not None else False
                    if comparable:
                        floor = max(scores[i] for i in candidates) - diversity_quality_slack
                        eligible = [i for i in candidates if scores[i] >= floor]
                        diverse = min(eligible, key=lambda i: (
                            details[i][1], -scores[i],
                            -int(members[i].get("size_bytes") or 0),
                            int(members[i].get("id") or i),
                        ))
                        # A utility budget bounds the quality/diversity tradeoff;
                        # cosine gain is ranking evidence, not an action label.
                        if details[pick][1] - details[diverse][1] > 1e-6:
                            pick, method = diverse, "quality_banded_diversity"
                            reasons.append("QUALITY_BANDED_DIVERSITY")
                candidates.remove(pick)
                mmr_score, max_similarity = details[pick]
                selection_steps.append({
                    "member": pick, "method": method, "score": mmr_score,
                    "utility_score": scores[pick], "max_selected_similarity": max_similarity,
                })
                reasons.append("MMR_QUALITY_DIVERSITY")
            chosen.append(pick)
        all_keepers.extend(chosen)
        selections.append({
            "phase_id": phase.phase_id,
            "members": list(phase.members),
            "keepers": chosen,
            "reason_codes": list(dict.fromkeys([
                *phase.boundary_reasons, *phase.uncertainty_reasons, *reasons,
            ])),
            "review_required": bool(critical_uncertainty or optional_gap_review or any(
                step["method"] == "quality_banded_diversity" for step in selection_steps
            )),
            "critical_uncertainty": critical_uncertainty,
            "optional_gap_review": optional_gap_review,
            "utility_margin": utility_margin,
            "variation": variation,
            "selection_steps": selection_steps,
            "evidence": {
                str(idx): {"utility_score": scores[idx], **scored[idx][1], "missing": scored[idx][2]}
                for idx in phase.members
            },
        })
    unique_keepers = list(dict.fromkeys(all_keepers))
    # Logical phases annotate evidence rather than guarantee every transition.
    # The base budget excludes keep-all. The explicit pose policy can protect
    # one additional contrasting pose in groups of 2–6, including both of a pair.
    group_budget = keeper_budget(len(members), max_group_keepers)
    if len(unique_keepers) > group_budget:
        pool = list(unique_keepers)
        bounded: list[int] = []
        banded_budget_change = False
        while pool and len(bounded) < group_budget:
            if not bounded:
                pick = max(pool, key=lambda idx: (
                    scores[idx], int(members[idx].get("size_bytes") or 0),
                    -int(members[idx].get("id") or idx),
                ))
            else:
                def rank(idx: int) -> tuple[float, float, int, int]:
                    emb = _embedding(members[idx])
                    sims = []
                    for selected in bounded:
                        other = _embedding(members[selected])
                        if emb is not None and other is not None and emb.shape == other.shape:
                            sims.append(float(np.dot(emb, other)))
                    similarity = max(sims) if sims else 0.0
                    value = mmr_quality_weight * scores[idx] - (1 - mmr_quality_weight) * similarity
                    return (value, scores[idx], int(members[idx].get("size_bytes") or 0),
                            -int(members[idx].get("id") or idx))
                pick = max(pool, key=rank)
                if diversity_policy == "quality_banded":
                    embs = {i: _embedding(members[i]) for i in [*pool, *bounded]}
                    first = embs[bounded[0]]
                    if first is not None and all(e is not None and e.shape == first.shape for e in embs.values()):
                        similarities = {i: max(float(np.dot(embs[i], embs[j])) for j in bounded) for i in pool}
                        floor = max(scores[i] for i in pool) - diversity_quality_slack
                        eligible = [i for i in pool if scores[i] >= floor]
                        diverse = min(eligible, key=lambda i: (
                            similarities[i], -scores[i], -int(members[i].get("size_bytes") or 0),
                            int(members[i].get("id") or i),
                        ))
                        if similarities[pick] - similarities[diverse] > 1e-6:
                            pick = diverse
                            banded_budget_change = True
            bounded.append(pick)
            pool.remove(pick)
        unique_keepers = bounded
        keeper_set = set(unique_keepers)
        for item in selections:
            item["keepers"] = [idx for idx in item["keepers"] if idx in keeper_set]
            if banded_budget_change and item["keepers"]:
                item["reason_codes"].append("QUALITY_BANDED_BUDGET_ALLOCATION")
                item["review_required"] = True
            if not item["keepers"]:
                item["reason_codes"].append("LOGICAL_PHASE_ANNOTATED_NOT_SEPARATELY_RETAINED")
                item["review_required"] = True
        budget_limited = True
    else:
        budget_limited = False
    pose_context = None
    pose_added = False
    if pose_policy in {"consensus", "stable_actor"} and unique_keepers:
        if pose_policy == "stable_actor":
            from .actor_set_pose import protect_actor_variants
            unique_keepers, pose_context = protect_actor_variants(
                members, unique_keepers, scores, displacement_threshold=pose_displacement_threshold)
        else:
            unique_keepers, pose_context = protect_pose_variants(
                members, unique_keepers, scores, displacement_threshold=pose_displacement_threshold,
            )
        extra = pose_context["added_keeper"]
        pose_added = extra is not None
        if pose_added:
            pose_context["base_keeper_budget"] = group_budget
            group_budget = max(group_budget, len(unique_keepers))
            for item in selections:
                if extra in item["members"]:
                    item["keepers"].append(extra)
                    item["reason_codes"].append("POSE_VARIANT_KEEPER")
                    item["review_required"] = True
                    item["selection_steps"].append({"member": extra, "method": "pose_variant", "semantic_phase_authority": False})
    # Diagnose after selection: external phase evidence cannot steer keepers.
    # Metadata-only callers need no stable IDs. Use an explicit index namespace
    # for the entire partition if any IDs are missing; never mix namespaces.
    stable_ids = all(member.get("id") is not None for member in members)
    if phase_requirements is not None and not stable_ids:
        raise ValueError("external phase requirements require stable member IDs")
    member_ids = ([int(member["id"]) for member in members] if stable_ids else
                  list(range(len(members))))
    keeper_ids = [member_ids[idx] for idx in unique_keepers]
    shadow_requirements = {
        "source": "shadow", "evidence_ref": "phase_selection.supplied_logical_phases",
        "member_id_space": "file_id" if stable_ids else "member_index",
        "confidence": "uncalibrated", "phases": [
            {"phase_id": str(phase.phase_id),
             "member_ids": [member_ids[idx] for idx in phase.members],
             "boundary_reasons": list(phase.boundary_reasons),
             "uncertainty_reasons": list(phase.uncertainty_reasons)} for phase in phases],
    }
    diagnostics = [diagnose_phase_coverage(member_ids, keeper_ids, group_budget, shadow_requirements)]
    if phase_requirements is not None:
        diagnostics.append(diagnose_phase_coverage(member_ids, keeper_ids, group_budget, phase_requirements))
    coverage_reasons = diagnostic_review_reasons(diagnostics)
    scoring_review = (score_policy == "guarded_common"
                      and scoring_context["effective_policy"] == "common_evidence")
    selection_reasons = [*coverage_reasons, *(["KEEPER_SCORE_COMPARABILITY_CHANGE"] if scoring_review else []),
                         *(["POSE_VARIANT_KEEPER"] if pose_added else []),
                         *(["LOCAL_QUALITY_DOMINANCE"] if local_changed else [])]
    result = {
        **({"scoring_context": scoring_context} if score_policy != "per_member" else {}),
        **({"diversity_context": {"policy": diversity_policy, "quality_slack": diversity_quality_slack}}
           if diversity_policy != "mmr" else {}),
        **({"pose_coverage": pose_context} if pose_context is not None else {}),
        **({"local_quality_context": local_context} if local_context is not None else {}),
        "keepers": unique_keepers,
        "phase_coverage_diagnostics": diagnostics,
        "reason_codes": selection_reasons,
        "mandatory_review": bool(coverage_reasons) or pose_added or local_changed,
        "utility_scores": scores,
        "phases": selections,
        "group_keeper_budget": group_budget,
        "budget_limited": budget_limited,
        "review_required": scoring_review or local_changed or bool(coverage_reasons) or any(item["review_required"] for item in selections),
        "review_phase_count": sum(item["review_required"] for item in selections),
        "semantics": "logical_phase_protection_within_existing_db_group",
        "physical_group_split": False,
        "authority": "shadow_review_only",
    }
    if local_quality_policy == "region_set":
        from .region_set_quality import protect_selection
        baseline = select_phase_keepers(members, phases,
            keepers_per_phase=keepers_per_phase, max_group_keepers=max_group_keepers,
            diversity_similarity=diversity_similarity, mmr_quality_weight=mmr_quality_weight,
            phase_requirements=phase_requirements, score_policy=score_policy,
            score_change_margin=score_change_margin, diversity_policy=diversity_policy,
            diversity_quality_slack=diversity_quality_slack, pose_policy=pose_policy,
            pose_displacement_threshold=pose_displacement_threshold, local_quality_policy="off")
        result = protect_selection(result, baseline, phases, local_context)
    if instance_recovery_policy == "review_only":
        from .instance_recovery import review_context
        context = review_context(members)
        result["instance_recovery_context"] = context
        if context["proposed_groups"]:
            result["mandatory_review"] = True
            result["review_required"] = True
            result["reason_codes"].append("INSTANCE_RECOVERY_PROPOSAL_REVIEW")
    return result
