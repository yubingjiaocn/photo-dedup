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
    weights = {
        "quality": 0.40, "face_clarity": 0.25, "exposure": 0.15,
        "subject_completeness": 0.12, "occlusion_inverse": 0.08,
    }
    available = {key: value for key, value in evidence.items() if value is not None}
    denominator = sum(weights[key] for key in available)
    total = (sum(weights[key] * value for key, value in available.items()) / denominator
             if denominator else 0.0)
    missing = [f"{key.upper()}_MISSING" for key, value in evidence.items() if value is None]
    return total, evidence, missing


def utility_scores(members: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Expose the exact score map used by selection and decision evidence."""
    return {idx: _score(member)[0] for idx, member in enumerate(members)}


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
) -> dict[str, Any]:
    """Protect logical phases with variation-aware budget and deterministic MMR.

    Size never grants a keeper. At most one extra variation keeper and one
    primary-signal uncertainty keeper are proposed per logical phase, then a
    group-level cap prevents detector flicker from recreating keep-all. The cap
    limits the budget; it does not create it. Optional scoring evidence only
    raises review, never keep-all.

    At most one extra
    variation keeper and one primary-signal uncertainty keeper are added per
    logical phase.
    """
    if max_group_keepers < 1:
        raise ValueError("max_group_keepers must be >= 1")
    if not 0.0 <= mmr_quality_weight <= 1.0:
        raise ValueError("mmr_quality_weight must be in [0, 1]")
    selections: list[dict[str, Any]] = []
    all_keepers: list[int] = []
    scores = utility_scores(members)
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
                candidates.remove(pick)
                mmr_score, max_similarity = details[pick]
                selection_steps.append({
                    "member": pick, "method": "mmr", "score": mmr_score,
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
            "review_required": bool(critical_uncertainty or optional_gap_review),
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
    # Logical phases are evidence annotations, not a promise to keep every
    # member or every detector transition. Bound group retention to at most
    # three representatives (and never all members for a non-singleton group).
    # This preserves multi-stage narratives without recreating keep-all.
    group_budget = min(max_group_keepers, max(1, 1 + int(math.log2(max(1, len(members))))))
    if len(members) > 1:
        group_budget = min(group_budget, len(members) - 1)
    if len(unique_keepers) > group_budget:
        pool = list(unique_keepers)
        bounded: list[int] = []
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
            bounded.append(pick)
            pool.remove(pick)
        unique_keepers = bounded
        keeper_set = set(unique_keepers)
        for item in selections:
            item["keepers"] = [idx for idx in item["keepers"] if idx in keeper_set]
            if not item["keepers"]:
                item["reason_codes"].append("LOGICAL_PHASE_ANNOTATED_NOT_SEPARATELY_RETAINED")
                item["review_required"] = True
        budget_limited = True
    else:
        budget_limited = False
    return {
        "keepers": unique_keepers,
        "utility_scores": scores,
        "phases": selections,
        "group_keeper_budget": group_budget,
        "budget_limited": budget_limited,
        "review_required": any(item["review_required"] for item in selections),
        "review_phase_count": sum(item["review_required"] for item in selections),
        "semantics": "logical_phase_protection_within_existing_db_group",
        "physical_group_split": False,
        "authority": "shadow_review_only",
    }
