"""Review-only ensemble evidence. Detector regions are observations, never identities.

L0 owns boxes/keypoints. L1 owns correspondence candidates and explicit unknowns;
L2 is a compact projection without geometry or old selection authority. No DB writes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

from .vlm_identity import identity

VERSION = "subject-set-review-v1"
CRITERIA = ("ensemble_completeness", "all_key_faces_eyes_occlusion",
            "synchronized_action_formation", "key_interaction", "stage_lighting",
            "distinct_action_phases")


def _object(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("invalid region")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise ValueError("invalid region")
    if not 0 <= value[0] < value[2] <= 1 or not 0 <= value[1] < value[3] <= 1:
        raise ValueError("invalid region")
    return list(value)


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1e-12, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def correspondence(left, right):
    """Conservative reciprocal geometry candidates; no person identity claim.

    Geometry alone cannot resolve crossings, duplicate detections or camera motion.
    Every unmatched observation is retained; region IDs are frame-local only.
    """
    pairs, conflicts = [], []
    scores = [[iou(a["box"], b["box"]) if a["class_id"] == b["class_id"] else 0
               for b in right] for a in left]
    for i, a in enumerate(left):
        order = sorted(range(len(right)), key=lambda j: scores[i][j], reverse=True)
        if not order or scores[i][order[0]] < .5:
            conflicts.append({"observation": a["ref"], "state": "unknown", "reason": "UNMATCHED"})
            continue
        j = order[0]
        column = sorted((scores[k][j] for k in range(len(left))), reverse=True)
        margin = scores[i][j] - (scores[i][order[1]] if len(order) > 1 else 0)
        reverse_margin = column[0] - (column[1] if len(column) > 1 else 0)
        if scores[i][j] != column[0] or min(margin, reverse_margin) < .15:
            conflicts.append({"observation": a["ref"], "state": "conflict", "reason": "AMBIGUOUS"})
            continue
        pairs.append({"observations": [a["ref"], right[j]["ref"]], "iou": scores[i][j],
                      "confidence": "medium", "authority": "visual_correspondence_candidate_not_identity"})
    matched = {p["observations"][1] for p in pairs}
    conflicts.extend({"observation": b["ref"], "state": "unknown", "reason": "UNMATCHED"}
                     for b in right if b["ref"] not in matched)
    return {"pairs": pairs, "conflicts": conflicts}


def compile_evidence(members, *, regime, scope=None):
    """Scope is a SHA-bound review annotation, not human/selector authority.

    Each scope entry may mark candidate role, a known detector omission and visible
    clipping. Unknown scope is a blocker even when detector counts agree.
    """
    if regime not in {"group", "stage"} or not 2 <= len(members) <= 6:
        raise ValueError("bounded group/stage evidence needs 2..6 frames")
    ids = [identity(r) for r in members]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stable identity")
    members = copy.deepcopy(members)
    scope = copy.deepcopy(scope or {})
    if set(scope) - set(ids):
        raise ValueError("stale scope identity")
    raw, frames, lookup, blockers = [], [], {}, []
    for row, key in zip(members, ids):
        q = _object(row.get("quality_meta"))
        regions = q.get("local_region_set", {}).get("regions", [])
        annotation = scope.get(key, {})
        subjects = []
        raw.append({"frame": key, "regions": regions, "pose": q.get("pose_evidence"),
                    "faces": _object(row.get("faces_json")), "scope": annotation})
        for r in regions:
            if r.get("kind") != "subject":
                continue
            ref = key + "/" + r["region_id"]
            if ref in lookup:
                raise ValueError("duplicate region ref")
            box = _box(r["box_normalized"])
            role = annotation.get("roles", {}).get(r["region_id"], "unknown")
            if role not in {"key_candidate", "bystander_candidate", "unknown"}:
                raise ValueError("invalid role")
            faces = [f for f in regions if f.get("kind") == "face" and f.get("parent_id") == r["region_id"]]
            poses = [p for p in q.get("pose_evidence", {}).get("people", []) if iou(box, _box(p["box"])) >= .5]
            pose = {"state": "unknown", "reason": "no_unique_pose_join"}
            if len(poses) == 1:
                points = poses[0].get("keypoints", [])
                pose = {"state": "derived", "visible_keypoints": sum(p[2] >= .5 for p in points),
                        "total_keypoints": len(points), "semantic_phase": "unknown"}
            s = {"ref": ref, "region_id": r["region_id"], "class_id": r.get("class_id"),
                 "detection_confidence": r.get("confidence"), "role": role,
                 "role_authority": "review_annotation_candidate" if role != "unknown" else "unknown",
                 "body_visibility": "detector_extent_only_not_complete",
                 "face_visibility": "detected" if faces else "unknown_not_absent",
                 "eye_state": "unknown", "eye_reason": "no_subject_bound_eye_join", "face_quality": [{k: f.get(k) for k in ("musiq", "clipiqa")} for f in faces],
                 "local_quality": {k: r.get(k) for k in ("musiq", "clipiqa")}, "pose": pose,
                 "cropping": "edge_touch_candidate" if min(box[0], box[1], 1-box[2], 1-box[3]) < .01 else "unknown",
                 "occlusion": "unknown", "unique_action_phase": "unknown"}
            subjects.append(s)
            lookup[ref] = {**s, "box": box, "face_boxes": [_box(f["box_normalized"]) for f in faces]}
        reasons = []
        if (annotation.get("coverage") != "verified_complete"
                or annotation.get("authority") not in {"human_verified", "independent_subject_set_validation"}):
            reasons.append("ENSEMBLE_COVERAGE_UNVERIFIED")
        if annotation.get("detector_missing"):
            reasons.append("DETECTOR_MISSING_SUBJECTS")
        if any(s["role"] == "unknown" for s in subjects):
            reasons.append("SUBJECT_ROLE_UNKNOWN")
        if any(s["role"] == "bystander_candidate" for s in subjects):
            reasons.append("BYSTANDER_CONTAMINATION")
        if annotation.get("clipping_conflict"):
            reasons.append("BODY_EXTENT_CONFLICT")
        blockers.extend(reasons)
        frames.append({"frame": key, "detected_subject_count": len(subjects),
                       "true_subject_count": "unknown", "subjects": subjects,
                       "ensemble_completeness": "unknown", "formation": "unknown",
                       "interaction": "unknown", "stage_lighting": "not_semantically_assessed",
                       "conflicts": reasons})
    matches = []
    anchor = [lookup[s["ref"]] for s in frames[0]["subjects"]]
    for frame in frames[1:]:
        matches.append(correspondence(anchor, [lookup[s["ref"]] for s in frame["subjects"]]))
    if len({f["detected_subject_count"] for f in frames}) != 1:
        blockers.append("SUBJECT_COUNT_MISMATCH")
    if any(m["conflicts"] for m in matches):
        blockers.append("CORRESPONDENCE_UNKNOWN_OR_CONFLICT")
    l0 = {"layer": "L0", "version": VERSION, "frames": raw}
    l1 = {"layer": "L1", "version": VERSION, "regime": regime, "frames": frames,
          "correspondence": matches, "blockers": sorted(set(blockers)),
          "criteria_priority": list(CRITERIA), "multi_phase_policy": "allow_multiple_non_substitutable_members_and_phases",
          "authority": "review_only_candidate", "coverage_ledger": [
              {"family": k, "disposition": "derived" if k in {"subject_count", "local_quality", "pose", "correspondence"} else "non-evidence",
               "reason": "observations_not_identity_or_preference"}
              for k in ("subject_count", "body_visibility", "face_visibility", "eyes", "local_quality", "pose",
                        "cropping_occlusion", "correspondence", "ensemble_completeness", "formation_interaction", "unique_phase")],
          "irreplaceable_candidates": [],
          "irreplaceable_candidate_contract": {
              "key": "stable_frame_ref + subject_ref or phase_ref",
              "authority": "observation_only_not_recommendation",
              "phase_order": "unknown_without_source_bound_evidence"}}
    return l0, l1, lookup


def prompt_evidence(l1):
    if l1.get("layer") != "L1" or l1.get("version") != VERSION:
        raise ValueError("L2 requires current L1")
    return {"layer": "L2", "version": VERSION, "regime": l1["regime"],
            "criteria_priority": l1["criteria_priority"], "policy": l1["multi_phase_policy"],
            "frames": l1["frames"], "correspondence": l1["correspondence"],
            "blockers": l1["blockers"], "coverage_ledger": l1["coverage_ledger"],
            "irreplaceable_candidates": l1["irreplaceable_candidates"]}


def cache_key(l0):
    return hashlib.sha256(json.dumps({"version": VERSION, "audit": l0}, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def cache_valid(packet, l0):
    return (isinstance(packet, Mapping) and packet.get("version") == VERSION
            and packet.get("cache_key") == cache_key(l0))


def visual_plan(l1, lookup, sizes, *, max_regions=2):
    """Rank body regions by within-track quality/geometry change, never face area.

    Blocked plans still expose proposed windows for diagnosis; they cannot render
    approved subject strips. Every observation is accounted for in the ledger.
    """
    if max_regions not in (2, 3):
        raise ValueError("subject budget must be 2 or 3")
    tracks = []
    for subject in l1["frames"][0]["subjects"]:
        refs = [subject["ref"]]
        for match in l1["correspondence"]:
            candidates = [p for p in match["pairs"] if p["observations"][0] == refs[0]]
            if len(candidates) == 1:
                refs.append(candidates[0]["observations"][1])
        if len(refs) != len(l1["frames"]) or any(lookup[r]["role"] != "key_candidate" for r in refs):
            continue
        boxes = [lookup[r]["box"] for r in refs]
        window = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        qualities = [lookup[r]["local_quality"].get("musiq") for r in refs]
        quality_delta = max(qualities)-min(qualities) if all(isinstance(q, (int, float)) for q in qualities) else 0
        score = quality_delta / 100 + max(1-iou(boxes[0], b) for b in boxes)
        native = [[math.floor((window[2]-window[0])*sizes[r.split('/')[0]][0]),
                   math.floor((window[3]-window[1])*sizes[r.split('/')[0]][1])] for r in refs]
        tracks.append({"refs": refs, "window": window, "discriminating_score": score,
                       "native_sizes": native, "size_gate": min(min(n) for n in native) >= 256,
                       "region": "body", "face_strip": "unavailable_without_native_256px_and_unique_face_join"})
    ordered = sorted(tracks, key=lambda t: (-t["discriminating_score"], t["refs"][0]))
    selected = [t for t in ordered if t["size_gate"]][:max_regions]
    selected_refs = {r for t in selected for r in t["refs"]}
    blockers = list(l1["blockers"])
    if len(selected) < 2:
        blockers.append("FEWER_THAN_TWO_KEY_SUBJECT_REGIONS")
    if any(not t["size_gate"] for t in tracks):
        blockers.append("ROI_NATIVE_SHORT_EDGE_BELOW_256")
    if any(min(size) < 400 for size in sizes.values()):
        blockers.append("SHEET_NATIVE_SHORT_EDGE_BELOW_400")
    ledger = [{"ref": ref, "role": obs["role"], "shown": False,
               "proposed": ref in selected_refs,
               "reason": "blocked_plan" if ref in selected_refs and blockers else
               "selected" if ref in selected_refs else
               "bystander_downweighted" if obs["role"] == "bystander_candidate" else "budget_or_unresolved",
               "face_coverage": "not_shown", "whole_frame_context": True} for ref, obs in lookup.items()]
    if not blockers:
        for entry in ledger:
            entry["shown"] = entry["proposed"]
    return {"status": "attention_only" if blockers else "candidate", "blockers": sorted(set(blockers)),
            "windows": selected, "max_regions": max_regions, "sheet_min_short_edge": 400,
            "strip_min_short_edge": 256, "coverage_ledger": ledger,
            "full_subject_coverage": False, "production_authority": False}
