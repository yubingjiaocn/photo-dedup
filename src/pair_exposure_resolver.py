"""Fail-closed, label-free two-photo candidate; no provider or write interface.

Families are provenance-distinct, NOT a claim of statistical independence.
Each family contributes at most one vote; disagreement within it abstains.
"""
from __future__ import annotations

import math

import numpy as np

from . import photo_exposure

VERSION = "pair-exposure-resolver-v3"
DINO_DIM = 768
DINO_MIN_NORM = 1e-12  # persisted vectors are unnormalized; reject zero/degenerate
CRITERIA = {
    "sharpness": ("quality_meta.sharpness", 1),
    "gradient": ("descriptor.quality", 1),
    "clipiqa": ("quality_meta.clipiqa", 1),
    "local_clipiqa": ("COMPILED:quality_meta.local_region_set.regions.clipiqa.mean", 1),
    "musiq": ("quality_meta.musiq", 1),
    "local_musiq": ("COMPILED:quality_meta.local_region_set.regions.musiq.mean", 1),
    "face_quality": ("quality_meta.face_quality", 1),
    "entropy": ("quality_meta.exposure.entropy_nonclip", 1),
    "mass_usable": ("quality_meta.exposure.mass_usable", 1),
    "clip_hi": ("quality_meta.exposure.clip_hi", -1),
    "clip_lo": ("quality_meta.exposure.clip_lo", -1),
    "edge_margin": ("COMPILED:faces_json.edges.minimum_edge_margin.mean", 1),
    "near_edge_faces": ("COMPILED:faces_json.edges.near_any_edge_count", -1),
    "eye_open": ("quality_meta.eye_detection.open_score", 1),
}
FAMILIES = {
    "detail": ("sharpness", "gradient"),
    "clipiqa": ("clipiqa", "local_clipiqa"),
    "musiq": ("musiq", "local_musiq"),
    "exposure": ("entropy", "mass_usable", "clip_hi", "clip_lo"),
    "face_quality": ("face_quality",),
    "face_composition": ("edge_margin", "near_edge_faces"),
    "eye_open": ("eye_open",),
}
# quality_score is a MUSIQ alias, deliberately not an input or fallback vote.
PROFILE_A = {
    "max_raw_dino_distance": 0.045, "max_rgb_mae": 0.09,
    "max_dhash": 8, "min_available_votes": 6, "min_vote_margin": 4,
}
# Small, predeclared discrete frontier; only A may be locked in this experiment.
PROFILES = {
    "A": PROFILE_A,
    "strict": {**PROFILE_A, "max_raw_dino_distance": 0.03, "max_rgb_mae": 0.06},
    "wide_diagnostic": {**PROFILE_A, "max_raw_dino_distance": 0.055, "max_dhash": 12},
}


def decode_dino(blob):
    """Decode the exact persisted <f2 x 768 contract; never guess dtype/shape."""
    if not isinstance(blob, bytes) or len(blob) != DINO_DIM * 2:
        raise ValueError("DINO requires exactly 1536 bytes")
    vector = np.frombuffer(blob, dtype="<f2").astype(np.float64)
    if vector.shape != (DINO_DIM,) or not np.isfinite(vector).all():
        raise ValueError("DINO dimension/nonfinite failure")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= DINO_MIN_NORM:
        raise ValueError("DINO must have a finite nonzero norm")
    return vector / norm


def raw_dino_distance(left_blob, right_blob):
    left, right = decode_dino(left_blob), decode_dino(right_blob)
    return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))


def identity_record(identity):
    prefix, sha = identity.split("@", 1)
    if not prefix.startswith("I") or not prefix[1:].isdigit() or int(prefix[1:]) <= 0 or len(sha) != 64:
        raise ValueError("invalid frame identity")
    int(sha, 16)
    return {"file_id": int(prefix[1:]), "content_sha256": sha}


def _number(value):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise TypeError("invalid numeric measurement")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite measurement")
    return value


def feature_audit(frames, evidence):
    if len(frames) != 2 or len({f["identity"] for f in frames}) != 2:
        raise ValueError("requires two distinct frame identities")
    for frame in frames:
        identity_record(frame["identity"])
        _number(frame["time"])
        desc = frame["descriptor"]
        for key, shape in (("rgb", (64, 64, 3)), ("dhash", (8, 8))):
            array = np.asarray(desc[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("invalid pixel descriptor")
        if desc["dhash"].dtype != np.bool_:
            raise ValueError("dHash must be boolean")
        if np.min(desc["rgb"]) < 0 or np.max(desc["rgb"]) > 1 or _number(desc["aspect"]) <= 0:
            raise ValueError("invalid descriptor range")
    pair = photo_exposure.pair_metrics(*frames)
    if pair is None or not all(math.isfinite(_number(x)) for x in pair.values()):
        raise ValueError("unavailable pair comparison")
    # Rounded request pair measurements are never a substitute for raw BLOBs.
    blobs = evidence["raw_dino_blobs"]
    distance = raw_dino_distance(*(blobs[f["identity"]] for f in frames))
    signs, status = {}, {}
    for name, (criterion, direction) in CRITERIA.items():
        values = evidence.get(criterion)
        if name == "gradient":
            raw = [f["descriptor"]["quality"] for f in frames]
        elif values is None or any(f["identity"] not in values or values[f["identity"]] is None for f in frames):
            signs[name], status[name] = 0, "unavailable"
            continue
        else:
            raw = [values[f["identity"]] for f in frames]
        left, right = map(_number, raw)
        delta = (left - right) * direction
        signs[name] = 1 if delta > 0 else -1 if delta < 0 else 0
        status[name] = "measured" if delta else "tie"
    votes, family_status = {}, {}
    for family, members in FAMILIES.items():
        active = {signs[m] for m in members if signs[m]}
        votes[family] = next(iter(active)) if len(active) == 1 else 0
        family_status[family] = "vote" if votes[family] else "conflict" if active else "unavailable_or_tie"
    return {"pair": pair, "raw_dino_cosine_distance": distance,
            "metric_signs": signs, "metric_status": status,
            "family_votes": votes, "family_status": family_status}


def resolve(frames, evidence, profile_name="A"):
    """Only local_select_one or vlm_required; malformed input fails closed.

    Invalid identities cannot form a safe exposure list: input_valid is false
    and callers must recover the original pair, never interpret [] as resolved.
    """
    result = {"resolver_version": VERSION, "profile": profile_name,
              "route": "vlm_required", "local_selected": [], "vlm_required": True,
              "vlm_exposed": [], "input_valid": False,
              "selection_contract_if_vlm": "v3_unchanged"}
    try:
        result["vlm_exposed"] = [identity_record(f["identity"]) for f in frames]
        profile = PROFILES[profile_name]
        audit = feature_audit(frames, evidence)
        votes = list(audit["family_votes"].values())
        first, second = votes.count(1), votes.count(-1)
        near = (audit["raw_dino_cosine_distance"] <= profile["max_raw_dino_distance"]
                and audit["pair"]["mae"] <= profile["max_rgb_mae"]
                and audit["pair"]["dhash"] <= profile["max_dhash"])
        result.update(input_valid=True, thresholds=dict(profile), feature_audit=audit,
                      decision_evidence={"near_pair": near, "first_votes": first,
                                         "second_votes": second, "available_votes": first + second,
                                         "winner_margin": abs(first - second)})
        if near and first + second >= profile["min_available_votes"] and abs(first - second) >= profile["min_vote_margin"]:
            winner = 0 if first > second else 1
            result.update(route="local_select_one", vlm_required=False, vlm_exposed=[],
                          local_selected=[identity_record(frames[winner]["identity"])])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError, AttributeError) as exc:
        result["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    return result
