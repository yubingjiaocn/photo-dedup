"""L1 -> L2 projection. No raw member or audit record is accepted here.

The complete coverage ledger and unrounded measurements remain in L1. L2 uses
only display labels and compact discriminating values. Absence/summary is explicit,
not a claim that missing detectors found no difference.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from statistics import median
from typing import Any

MAX_BYTES = 45_000
MAX_TOKENS_ESTIMATE = 15_000
MAX_ARRAY = 20
_NON_EVIDENCE = re.compile(
    r"utility|pair_margin|keeper|decision|timestamp|datetime|mtime|thumb|run_id|"
    r"schema_version|backend|producer|interface|embedding_offset|native_size|"
    r"iqa_input|iqa_scale|scan_|feature_status|file_kind|size_bytes|method|policy|"
    r"source_bytes|leaf_count|excluded_sources|collections|histogram|top_regions|top_faces|top_people"
)
_GEOMETRY = re.compile(r"(?:^|[.\[])box(?:es|_normalized)?(?:$|[.\[])|bbox|landmark|keypoints")


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def token_estimate(value: Any) -> int:
    """Deterministic byte/3 estimate, not provider billed tokens."""
    return math.ceil(len(encoded(value)) / 3)


def _leaves(value: Any, path: str = "") -> dict[str, Any]:
    if _NON_EVIDENCE.search(path) or _GEOMETRY.search(path):
        return {}
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            result.update(_leaves(child, f"{path}.{key}" if path else str(key)))
        return result
    if isinstance(value, (list, tuple)):
        # Categorical conflict/applicability lists are short states, not geometry.
        if len(value) <= 4 and all(isinstance(v, str) and len(v) <= 100 for v in value):
            return {path: "; ".join(value)} if value else {}
        result = {}
        for i, child in enumerate(value):
            if isinstance(child, Mapping):
                result.update(_leaves(child, f"{path}[{i}]"))
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return {path: value}
    if isinstance(value, str) and len(value) <= 160:
        return {path: value}
    return {}


def scalar_comparisons(derived: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for frame in derived["frames"]:
        for entry in frame["criteria"]:
            family = entry["source_family"]
            for leaf, value in _leaves(entry["value"]).items():
                suffix = entry["ref"].split(":", 1)[1] + (f".{leaf}" if leaf else "")
                if not _NON_EVIDENCE.search(suffix) and not _GEOMETRY.search(suffix):
                    metrics[(family, suffix)][frame["frame_identity"]] = value
    result = []
    for (family, metric), values in sorted(metrics.items()):
        values = {frame["frame_identity"]: values.get(frame["frame_identity"])
                  for frame in derived["frames"]}
        row = {"family": family, "metric": metric, "values": values,
               "authority": "persisted_measurement_not_preference",
               "applicability": "as_recorded", "conflicts": []}
        numbers = {k: float(v) for k, v in values.items()
                   if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)}
        if numbers and len(numbers) == len(values):
            best, center = max(numbers.values()), median(numbers.values())
            row.update(rank={k: 1 + sum(x > v for x in numbers.values()) for k, v in numbers.items()},
                       delta_best={k: v - best for k, v in numbers.items()},
                       delta_median={k: v - center for k, v in numbers.items()},
                       direction="numeric_only_not_preference")
        row["discriminating"] = len({encoded(v) for v in values.values()}) > 1
        result.append(row)
    return result


def compile_prompt(derived: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    """Accept only L1; aliases map stable identities to labels, never join data."""
    if derived.get("layer") != "L1":
        raise ValueError("L2 requires L1 derived evidence")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in derived["scalar_comparisons"]:
        if row["discriminating"]:
            families[row["family"]].append(row)
    evidence = []
    dispositions = {}
    for item in derived["coverage_ledger"]:
        family = item["source_family"]
        dispositions.setdefault(family, set()).add(item["disposition"])
    checked = {}
    for family in sorted(dispositions):
        candidates = families.get(family, [])
        if _NON_EVIDENCE.search(family):
            continue
        if not candidates:
            checked[family] = "no_discriminating_scalar_or_unavailable; see L1"
            continue
        # Preserve all scalar discriminators; size gate rejects rather than silently truncates.
        for row in candidates:
            ref = f"D{len(evidence) + 1:03d}"
            compact = {"ref": ref, "criterion": row["metric"],
                       "values": {aliases[k]: round(v, 3) if isinstance(v, float) else v
                                  for k, v in row["values"].items()}}
            if "rank" in row and family in {"quality_score", "quality_meta.musiq", "quality_meta.clipiqa", "quality_meta.sharpness", "quality_meta.face_quality"}:
                compact["rank"] = {aliases[k]: v for k, v in row["rank"].items()}
                compact["delta_best"] = {aliases[k]: round(v, 3) for k, v in row["delta_best"].items()}
            evidence.append(compact)
        checked[family] = f"{len(candidates)} discriminating scalars"
    pairs = []
    for pair in derived["pairwise_measurements"]:
        for measurement in pair["measurements"]:
            if measurement["criterion"] not in {
                "phash", "content_sha256", "dinov2_embedding", "local_quality_embedding",
            }:
                continue
            values = _leaves(measurement["value"])
            pairs.append({"ref": f"P{len(pairs) + 1:03d}",
                          "frames": [aliases[k] for k in pair["frames"]],
                          "criterion": measurement["criterion"],
                          "status": measurement["status"],
                          "value": {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in values.items()}})
    components = [{alias} for alias in aliases.values()]
    for item in pairs:
        if item["criterion"] == "content_sha256" and item["value"].get("equal") is True:
            touched = [group for group in components if group.intersection(item["frames"])]
            merged = set().union(*touched)
            components = [group for group in components if group not in touched] + [merged]
    identical_sets = sorted(sorted(group) for group in components if len(group) > 1)
    pair_policy = "upper_triangle"
    if len(aliases) > 6:
        by_criterion = defaultdict(list)
        for item in pairs:
            by_criterion[item["criterion"]].append(item)
        bounded = []
        for criterion, items in sorted(by_criterion.items()):
            if criterion == "content_sha256":
                continue  # All equivalence classes are explicit in identical_sets.
            if criterion == "phash":
                nearest = {}
                for alias in aliases.values():
                    available = [item for item in items if alias in item["frames"]
                                 and item["value"].get("hamming_distance") is not None]
                    if available:
                        item = min(available, key=lambda x: (x["value"]["hamming_distance"], x["frames"]))
                        nearest[item["ref"]] = item
                bounded.extend(nearest.values() or items[:1])
                continue
            key = {"phash": "hamming_distance", "dinov2_embedding": "cosine_distance",
                   "local_quality_embedding": "maximum_cosine_distance",
                   "content_sha256": "equal"}[criterion]
            measured = [item for item in items if item["value"].get(key) is not None]
            if measured:
                ordered = sorted(measured, key=lambda item: (item["value"][key], item["frames"]))
                bounded.append(ordered[0])
                if ordered[-1] != ordered[0]:
                    bounded.append(ordered[-1])
            else:
                bounded.append(items[0])
        pairs = bounded
        pair_policy = "large_group_phash_nearest_neighbors_and_cosine_extremes; complete_pairs_in_L1"
    applicability = {}
    for row in derived["scalar_comparisons"]:
        if any(part in row["metric"].lower() for part in ("applicability", "authority", "conflict", ".state", ".status")):
            values = list(row["values"].values())
            applicability[row["metric"]] = (values[0] if all(v == values[0] for v in values)
                                            else {aliases[k]: v for k, v in row["values"].items()})
    context = {
        "layer": "L2", "schema_version": 1,
        "frames": list(aliases.values()),
        "authority": "measurements_not_preferences; unknown applicability stays unknown",
        "applicability_and_conflicts": applicability,
        "scene_regime": "unknown_unless_explicit_in_evidence",
        "evidence": {item.pop("ref"): item for item in evidence},
        "pairs": {item.pop("ref"): item for item in pairs},
        "pair_policy": pair_policy,
        "identical_sets": identical_sets,
        "region_policy": "aggregate_only; no_cross_file_registration_available",
        "checked_families": [f"{family}: {state}" for family, state in checked.items()],
        "evidence_refs": [],
    }
    for ref, item in context["evidence"].items():
        context["evidence_refs"].extend(f"{alias}:{ref}" for alias in item["values"])
    for ref, item in context["pairs"].items():
        context["evidence_refs"].append(f"PAIR:{':'.join(item['frames'])}:{ref}")
    return context
