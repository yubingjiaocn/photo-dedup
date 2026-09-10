"""Add-only actor-local pose protection after complete, stable region-set tracking."""

from __future__ import annotations
import copy
import json
from collections import Counter
from . import pose_evidence as pose
from .local_quality import _iou
from .region_set_quality import decode, match

IDENTITY_SIMILARITY = 0.90


def _pose_record(member):
    try:
        r = json.loads(member.get("quality_meta") or "{}").get("pose_evidence")
        if (
            not isinstance(r, dict)
            or r.get("status") != "ok"
            or not isinstance(r.get("producer"), str)
            or not r["producer"]
        ):
            return None
        if not isinstance(r.get("shape"), list) or len(r["shape"]) != 2:
            return None
        if not all(pose._finite(v) and v > 0 for v in r["shape"]):
            return None
        return r
    except (ValueError, TypeError, AttributeError):
        return None


def _assigned_pose(region, regions, record):
    """Pose-to-instance association is reciprocal and geometry-unique, never largest."""
    if record is None:
        return None
    people = record.get("people", [])
    if not isinstance(people, list):
        return None
    valid = {p["index"] for p in pose._candidates(record)}
    candidates = []
    for i, p in enumerate(people):
        if i not in valid:
            continue
        overlap = _iou(region["box_normalized"], p["box"])
        candidates.append((overlap, i))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] < 0.5:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
        return None
    chosen = people[candidates[0][1]]
    reverse = sorted(
        (_iou(r["box_normalized"], chosen["box"]), r["region_id"])
        for r in regions.values()
        if r["kind"] == "subject" and r["class_id"] == 0
    )
    if not reverse or reverse[-1][1] != region["region_id"]:
        return None
    if len(reverse) > 1 and reverse[-1][0] - reverse[-2][0] < 0.1:
        return None
    x1, y1, x2, y2 = chosen["box"]
    w, h = record["shape"]
    bw = x2 - x1
    bh = y2 - y1
    if bw * w < 96 or bh * h < 192 or bw * bh < 0.015:
        return None
    # Coordinate transform only: never upsample pixels, boost confidence or invent joints.
    if any(
        not isinstance(p, list) or len(p) != 3 or not all(pose._finite(v) for v in p)
        for p in chosen["keypoints"]
    ):
        return None
    local = copy.deepcopy(chosen)
    local["box"] = [0.0, 0.0, 1.0, 1.0]
    local["keypoints"] = [
        [(x - x1) / bw, (y - y1) / bh, c] for x, y, c in chosen["keypoints"]
    ]
    return {
        "producer": record["producer"],
        "status": "ok",
        "shape": [bw * w, bh * h],
        "people": [local],
    }


def protect_actor_variants(members, keepers, scores, *, displacement_threshold=0.4):
    if not pose._finite(displacement_threshold) or displacement_threshold <= 0:
        raise ValueError("actor displacement threshold must be positive")
    result = list(keepers)
    evidence = {
        "policy": "stable_set_actor_local",
        "participant_policy": "all-observed-set / any-resolved-actor",
        "added_keeper": None,
        "comparisons": [],
        "max_extra_keepers": 1,
        "displacement_threshold": displacement_threshold,
        "identity_similarity": IDENTITY_SIMILARITY,
        "semantic_phase_authority": False,
        "refusals": {},
        "existing_keepers_preserved": True,
        "largest_subject_priority": False,
    }
    if not 2 <= len(members) <= 6 or not keepers or len(keepers) == len(members):
        return result, evidence
    if len(scores) != len(members) or any(
        not pose._finite(scores[i]) for i in range(len(members))
    ):
        raise ValueError("one finite score per member required")
    if len(set(keepers)) != len(keepers) or any(
        type(i) is not int or not 0 <= i < len(members) for i in keepers
    ):
        raise ValueError("invalid keepers")
    sets = [decode(m) for m in members]
    if any(s is None for s in sets):
        evidence["refusals"] = {"INCOMPLETE_OBSERVED_SET": 1}
        return result, evidence
    # One anchor track set for the whole group; pairwise permutations must agree with it.
    anchor = sets[0]
    tracks = []
    refusals = Counter()
    if sum(r["kind"] == "subject" for r in anchor.values()) < 2:
        evidence["refusals"] = {"MULTI_SUBJECT_SUPPORT_REQUIRED": 1}
        return result, evidence
    for s in sets:
        pairs, why = match(anchor, s, IDENTITY_SIMILARITY)
        if pairs is None:
            evidence["refusals"] = {why: 1}
            return result, evidence
        tracks.append(
            {a["region_id"]: b["region_id"] for a, b in pairs if a["kind"] == "subject"}
        )
    records = [_pose_record(m) for m in members]
    accepted = {}
    for candidate in range(len(members)):
        if candidate in keepers:
            continue
        if scores[candidate] < max(scores[k] for k in keepers) - 0.15:
            refusals["GLOBAL_QUALITY_FLOOR"] += 1
            continue
        comparisons = []
        for k in keepers:
            pairs, why = match(sets[candidate], sets[k], IDENTITY_SIMILARITY)
            if pairs is None:
                refusals[why] += 1
                break
            expected = {tracks[candidate][a]: tracks[k][a] for a in tracks[0]}
            if any(
                expected.get(a["region_id"]) != b["region_id"]
                for a, b in pairs
                if a["kind"] == "subject"
            ):
                refusals["TRACK_CYCLE_INCONSISTENT"] += 1
                break
            if any(
                a["musiq"] < b["musiq"] - 10 or a["clipiqa"] < b["clipiqa"] - 0.15
                for a, b in pairs
            ):
                refusals["ANY_REGION_QUALITY_DROP"] += 1
                break
            actors = []
            for a, b in pairs:
                if a["kind"] != "subject" or a["class_id"] != 0:
                    continue
                pa = _assigned_pose(a, sets[candidate], records[candidate])
                pb = _assigned_pose(b, sets[k], records[k])
                measured = pose.compare_poses(
                    pa, pb, participant_policy="visible_participant"
                )
                if (
                    measured["eligible"]
                    and measured["displacement"] >= displacement_threshold
                ):
                    actors.append(
                        {
                            "candidate_region": a["region_id"],
                            "keeper_region": b["region_id"],
                            **measured,
                        }
                    )
            if not actors:
                refusals["NO_RESOLVED_ACTOR_CHANGE"] += 1
                break
            comparisons.append({"keeper": k, "set_bijection": True, "actors": actors})
        if len(comparisons) == len(keepers):
            accepted[candidate] = comparisons
    if accepted:
        extra = max(accepted, key=lambda i: (scores[i], -i))
        result.append(extra)
        evidence.update(added_keeper=extra, comparisons=accepted[extra])
    evidence["refusals"] = dict(refusals)
    return result, evidence
