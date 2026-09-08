"""Deterministic phase segmentation and explainable multi-keeper selection.

This module consumes cached metadata only.  It neither opens source media nor
creates deletion authority. Missing evidence increases keepers and review.
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
) -> list[Phase]:
    """Split an ordered visual group when cached temporal/visual state changes."""
    if not members:
        return []
    order = sorted(range(len(members)), key=lambda i: (
        members[i].get("exif_timestamp") is None,
        members[i].get("exif_timestamp") or 0,
        int(members[i].get("id") or i),
    ))
    phases: list[Phase] = []
    current = [order[0]]
    boundary_reasons: list[str] = ["PHASE_START"]
    uncertain = False
    for left, right in pairwise(order):
        a, b = members[left], members[right]
        reasons: list[str] = []
        ta, tb = a.get("exif_timestamp"), b.get("exif_timestamp")
        if ta is None or tb is None:
            reasons.append("TIME_MISSING")
            uncertain = True
        elif int(tb) - int(ta) > max_gap_seconds:
            reasons.append("TIME_GAP")
        ea, eb = _embedding(a), _embedding(b)
        if ea is None or eb is None or ea.shape != eb.shape:
            reasons.append("EMBEDDING_MISSING")
            uncertain = True
        elif float(np.dot(ea, eb)) < embedding_boundary:
            reasons.append("EMBEDDING_CHANGE")
        ca, cb = _subject_center(a), _subject_center(b)
        if ca is None or cb is None:
            reasons.append("SUBJECT_POSITION_MISSING")
            uncertain = True
        elif max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])) > position_shift:
            reasons.append("SUBJECT_POSITION_CHANGE")
        face_a, face_b = int(a.get("face_count") or 0), int(b.get("face_count") or 0)
        if face_a != face_b:
            reasons.append("FACE_COUNT_CHANGE")
        hard = [r for r in reasons if r in {
            "TIME_GAP", "EMBEDDING_CHANGE", "SUBJECT_POSITION_CHANGE", "FACE_COUNT_CHANGE"
        }]
        if hard:
            phases.append(Phase(len(phases), tuple(current), tuple(boundary_reasons), uncertain))
            current = [right]
            boundary_reasons = hard
            uncertain = any(r.endswith("MISSING") for r in reasons)
        else:
            current.append(right)
            uncertain = uncertain or any(r.endswith("MISSING") for r in reasons)
    phases.append(Phase(len(phases), tuple(current), tuple(boundary_reasons), uncertain))
    return phases


def _score(member: Mapping[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    meta = _meta(member)
    quality = float(member.get("quality_score") or 0.0) / 100.0
    face = float(meta.get("face_quality") or 0.0)
    exposure = meta.get("exposure") if isinstance(meta.get("exposure"), dict) else {}
    exposure_ok = 1.0 - min(1.0, float(exposure.get("clip_hi", 0.5)) + float(exposure.get("clip_lo", 0.5)))
    completeness = float(meta.get("subject_completeness", 0.5))
    occlusion = float(meta.get("occlusion", 0.5))
    evidence = {
        "quality": max(0.0, min(1.0, quality)),
        "face_clarity": max(0.0, min(1.0, face)),
        "exposure": max(0.0, min(1.0, exposure_ok)),
        "subject_completeness": max(0.0, min(1.0, completeness)),
        "occlusion_inverse": max(0.0, min(1.0, 1.0 - occlusion)),
    }
    missing = []
    for key, source in (("FACE_CLARITY_MISSING", "face_quality"),
                        ("EXPOSURE_MISSING", "exposure"),
                        ("SUBJECT_COMPLETENESS_MISSING", "subject_completeness"),
                        ("OCCLUSION_MISSING", "occlusion")):
        if source not in meta:
            missing.append(key)
    total = (0.40 * evidence["quality"] + 0.25 * evidence["face_clarity"] +
             0.15 * evidence["exposure"] + 0.12 * evidence["subject_completeness"] +
             0.08 * evidence["occlusion_inverse"])
    return total, evidence, missing


def select_phase_keepers(
    members: Sequence[Mapping[str, Any]], phases: Sequence[Phase], *,
    keepers_per_phase: int = 1, large_phase_size: int = 6,
    diversity_similarity: float = 0.965,
) -> dict[str, Any]:
    """Select at least one keeper per phase with quality + greedy diversity."""
    selections = []
    all_keepers: list[int] = []
    for phase in phases:
        target = max(1, keepers_per_phase)
        reasons = ["PHASE_MINIMUM_KEEPER"]
        if len(phase.members) >= large_phase_size:
            target += 1
            reasons.append("LARGE_PHASE_EXTRA_KEEPER")
        if phase.uncertain:
            target += 1
            reasons.append("MISSING_FEATURES_EXTRA_KEEPER")
        scored = {idx: _score(members[idx]) for idx in phase.members}
        candidates = sorted(phase.members, key=lambda idx: (
            scored[idx][0], int(members[idx].get("size_bytes") or 0), -idx
        ), reverse=True)
        chosen: list[int] = []
        while candidates and len(chosen) < min(target, len(phase.members)):
            if not chosen:
                pick = candidates.pop(0)
            else:
                diversity_values = {}
                for idx in candidates:
                    emb = _embedding(members[idx])
                    similarities = []
                    for selected in chosen:
                        other = _embedding(members[selected])
                        if emb is not None and other is not None and emb.shape == other.shape:
                            similarities.append(float(np.dot(emb, other)))
                    novelty = 1.0 - max(similarities) if similarities else 1.0
                    diversity_values[idx] = (novelty, scored[idx][0], -idx)
                pick = max(candidates, key=diversity_values.__getitem__)
                candidates.remove(pick)
                if all(_embedding(members[i]) is not None for i in [pick, *chosen]):
                    max_sim = max(float(np.dot(_embedding(members[pick]), _embedding(members[i]))) for i in chosen)
                    if max_sim < diversity_similarity:
                        reasons.append("DIVERSITY_KEEPER")
            chosen.append(pick)
        all_keepers.extend(chosen)
        selections.append({
            "phase_id": phase.phase_id,
            "members": list(phase.members),
            "keepers": chosen,
            "reason_codes": list(dict.fromkeys([*phase.boundary_reasons, *reasons])),
            "review_required": phase.uncertain,
            "evidence": {str(idx): {"utility_score": scored[idx][0], **scored[idx][1],
                                    "missing": scored[idx][2]} for idx in phase.members},
        })
    return {
        "keepers": list(dict.fromkeys(all_keepers)),
        "phases": selections,
        "review_required": any(item["review_required"] for item in selections),
        "authority": "shadow_review_only",
    }
