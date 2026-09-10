"""Complete symmetric native region-set comparisons, never semantic subject truth."""

from __future__ import annotations
import json
from collections import Counter
import numpy as np
from .local_quality import DIM, STRIDE, _finite, _iou

MAX_SUBJECTS = 12


def decode(member):
    try:
        meta = json.loads(member.get("quality_meta") or "{}")["local_region_set"]
        blob = member.get("local_quality_embedding")
        if meta.get("schema_version") != 2 or meta.get("status") != "complete":
            return None
        regions = meta["regions"]
        if (
            not isinstance(blob, (bytes, bytearray, memoryview))
            or not 0 < len(blob) <= 24 * STRIDE
        ):
            return None
        if not isinstance(regions, list) or not 1 <= len(regions) <= 24:
            return None
        result = {}
        for r in regions:
            rid = r["region_id"]
            box = r["box_normalized"]
            offset = r["embedding_offset"]
            if (
                not isinstance(rid, str)
                or rid in result
                or r["kind"] not in {"subject", "face"}
                or type(r["class_id"]) is not int
                or r["class_id"] not in {0, 15, 16}
                or len(box) != 4
                or not all(_finite(v, 0, 1) for v in box)
                or not box[0] < box[2]
                or not box[1] < box[3]
                or type(offset) is not int
                or offset < 0
                or offset % STRIDE
                or offset + STRIDE > len(blob)
                or not _finite(r["confidence"], 0.6, 1)
                or len(r["native_size"]) != 2
                or not all(_finite(v, 96, 100000) for v in r["native_size"])
                or not _finite(r["musiq"], 0, 100)
                or not _finite(r["clipiqa"], 0, 1)
            ):
                return None
            v = np.frombuffer(blob, dtype="<f2", offset=offset, count=DIM).astype(
                np.float32
            )
            norm = float(np.linalg.norm(v))
            if not np.all(np.isfinite(v)) or not np.isfinite(norm) or norm <= 0:
                return None
            result[rid] = {**r, "vector": v / norm}
        subjects = {k: v for k, v in result.items() if v["kind"] == "subject"}
        if not 1 <= len(subjects) <= MAX_SUBJECTS or meta.get("subject_count") != len(
            subjects
        ):
            return None
        parents = []
        for r in result.values():
            if r["kind"] == "subject" and r.get("parent_id") is not None:
                return None
            if r["kind"] == "face":
                parent = r.get("parent_id")
                if (
                    parent not in subjects
                    or subjects[parent]["class_id"] != 0
                    or r["class_id"] != 0
                    or parent in parents
                ):
                    return None
                fb, pb = r["box_normalized"], subjects[parent]["box_normalized"]
                if not (
                    pb[0] <= (fb[0] + fb[2]) / 2 <= pb[2]
                    and pb[1] <= (fb[1] + fb[3]) / 2 <= pb[3]
                ):
                    return None
                parents.append(parent)
        return result
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        BufferError,
        OverflowError,
    ):
        return None


def match(left, right, similarity=0.97):
    """Full bijection with mutual margins; reversing inputs inverts it."""
    a = [r for r in left.values() if r["kind"] == "subject"]
    b = [r for r in right.values() if r["kind"] == "subject"]
    if (
        not a
        or not b
        or len(a) != len(b)
        or Counter(r["class_id"] for r in a) != Counter(r["class_id"] for r in b)
    ):
        return None, "SET_COVERAGE_UNSTABLE"
    costs = np.full((len(a), len(b)), -2.0)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            sim = float(x["vector"] @ y["vector"])
            overlap = _iou(x["box_normalized"], y["box_normalized"])
            if x["class_id"] == y["class_id"] and sim >= similarity and overlap >= 0.5:
                costs[i, j] = 0.8 * sim + 0.2 * overlap
    rows = np.argmax(costs, axis=1)
    cols = np.argmax(costs, axis=0)
    pairs = []
    for i, j in enumerate(rows):
        if costs[i, j] < 0 or cols[j] != i:
            return None, "ASSOCIATION_UNKNOWN"
        row = np.sort(costs[i])
        col = np.sort(costs[:, j])
        if (len(row) > 1 and row[-1] - row[-2] < 0.03) or (
            len(col) > 1 and col[-1] - col[-2] < 0.03
        ):
            return None, "ASSOCIATION_AMBIGUOUS"
        x, y = a[i], b[j]
        pairs.append((x, y))
        fa = [r for r in left.values() if r.get("parent_id") == x["region_id"]]
        fb = [r for r in right.values() if r.get("parent_id") == y["region_id"]]
        if len(fa) != len(fb):
            return None, "FACE_COVERAGE_UNSTABLE"
        pairs.extend(zip(fa, fb))
    if len(pairs) != len(left) or len(pairs) != len(right):
        return None, "REGION_COVERAGE_UNSTABLE"
    return pairs, "COMPLETE_SYMMETRIC_MATCH"


def _global_vector(m):
    try:
        v = np.frombuffer(m["dinov2_embedding"], dtype="<f2").astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if v.size == DIM and n > 0 and np.all(np.isfinite(v)) else None
    except (KeyError, ValueError, TypeError):
        return None


def adjust_scores(members, phases, scores, *, similarity=0.97, penalty=0.08):
    if not _finite(similarity, 0.9, 1) or not _finite(penalty, 0, 0.25):
        raise ValueError("region-set parameters out of range")
    sets = [decode(m) for m in members]
    vectors = [_global_vector(m) for m in members]
    out = dict(scores)
    comparisons = []
    refusals = Counter()
    frames = []
    for i, s in enumerate(sets):
        frames.append(
            {
                "member": i,
                "region_count": len(s) if s else 0,
                "subject_count": sum(r["kind"] == "subject" for r in s.values())
                if s
                else 0,
                "worst_musiq": min(r["musiq"] for r in s.values()) if s else None,
                "worst_clipiqa": min(r["clipiqa"] for r in s.values()) if s else None,
                "eye_state": "unknown",
                "semantic_subject_coverage": "unknown",
            }
        )
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i == j:
                continue
            if a is None or b is None:
                refusals["NO_COMPLETE_RELIABLE_SET"] += 1
                continue
            pairs, reason = match(a, b, similarity)
            if pairs is None:
                refusals[reason] += 1
                continue
            face_parents = {
                r.get("parent_id") for r in a.values() if r["kind"] == "face"
            }
            if any(
                r["kind"] == "subject"
                and r["class_id"] == 0
                and rid not in face_parents
                for rid, r in a.items()
            ):
                refusals["HUMAN_FACE_QUALITY_UNAVAILABLE"] += 1
                continue
            x, y = vectors[i], vectors[j]
            if x is None or y is None or x.shape != y.shape or float(x @ y) < 0.995:
                refusals["CONTEXT_OR_INTERACTION_UNCERTAIN"] += 1
                continue
            gm = [members[k].get("quality_score") for k in (i, j)]
            gc = [
                json.loads(members[k].get("quality_meta") or "{}").get("clipiqa")
                for k in (i, j)
            ]
            if (
                not all(_finite(v, 0, 100) for v in gm)
                or not all(_finite(v, 0, 1) for v in gc)
                or gm[1] < gm[0] - 0.5
                or gc[1] < gc[0] - 0.01
            ):
                refusals["GLOBAL_BASELINE_QUALITY_UNCERTAIN_OR_WORSE"] += 1
                continue
            if any(
                q["musiq"] < p["musiq"] or q["clipiqa"] < p["clipiqa"] for p, q in pairs
            ):
                refusals["ANY_REGION_WORSENED"] += 1
                continue
            wm = min(q["musiq"] for _, q in pairs) - min(p["musiq"] for p, _ in pairs)
            wc = min(q["clipiqa"] for _, q in pairs) - min(
                p["clipiqa"] for p, _ in pairs
            )
            if (
                wm < 1
                or wc < 0.02
                or not any(
                    q["musiq"] - p["musiq"] >= 1 and q["clipiqa"] - p["clipiqa"] >= 0.05
                    for p, q in pairs
                )
            ):
                refusals["WORST_REGION_NOT_IMPROVED"] += 1
                continue
            out[i] = max(0, float(scores[i]) - penalty)
            comparisons.append(
                {
                    "worse": i,
                    "better": j,
                    "matched_regions": len(pairs),
                    "subject_count": frames[i]["subject_count"],
                    "worst_musiq_gain": wm,
                    "worst_clipiqa_gain": wc,
                }
            )
    return out, {
        "policy": "symmetric_region_set",
        "similarity": similarity,
        "max_utility_penalty": penalty,
        "frames": frames,
        "comparisons": comparisons,
        "refusals": dict(refusals),
        "changed_members": [i for i in out if out[i] != scores[i]],
        "eye_state_authority": False,
        "amodal_completeness_authority": False,
        "interaction_semantic_authority": False,
        "no_reliable_region_path": "preserve_whole_frame_baseline",
    }


def protect_selection(candidate, baseline, phases, context):
    """Every removed keeper needs a retained set witness; preserve phase coverage."""
    old = set(baseline["keepers"])
    new = set(candidate["keepers"])
    reason = None
    if old == new:
        reason = "NO_KEEPER_CHANGE"
    elif any(set(p.members) & old and not set(p.members) & new for p in phases):
        reason = "BASELINE_PHASE_WOULD_BE_LOST"
    elif any(
        not any(c["worse"] == i and c["better"] in new for c in context["comparisons"])
        for i in old - new
    ):
        reason = "REMOVED_KEEPER_HAS_NO_RETAINED_SET_WITNESS"
    if reason:
        context = {
            **context,
            "proposed_changed_members": context["changed_members"],
            "changed_members": [],
            "selection_applied": False,
            "selection_guard": reason,
        }
        return {**baseline, "local_quality_context": context}
    return {
        **candidate,
        "local_quality_context": {
            **context,
            "selection_applied": True,
            "selection_guard": "SUPPORTED_SUBSTITUTIONS_ONLY",
        },
    }
