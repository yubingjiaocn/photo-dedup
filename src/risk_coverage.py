"""Deterministic selective prediction and stratified silent-region auditing.

The policy consumes cached group evidence only.  It assigns review attention;
it never grants deletion authority or opens source media.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .phase_coverage import diagnostic_review_reasons

REVIEW_PRIMARY = "REVIEW_PRIMARY"
SAFE_SILENT = "SAFE_SILENT"
DIAGNOSTIC_SAMPLE = "DIAGNOSTIC_SAMPLE"

# Contributions are deliberately monotone: adding conflict/missing evidence or
# reducing margins cannot lower risk.  The constants are transparent tune-set
# policy, not a fitted probability model.
_REASON_WEIGHTS = {
    "GROUP_NOT_BYTE_IDENTICAL": 0.04,
    "LARGE_GROUP": 0.08,
    "LONG_TIME_SPAN": 0.08,
    "LOW_EMBEDDING_PURITY": 0.13,
    "LOCAL_CHANGE_SIGNAL": 0.13,
    "FACE_COUNT_CONFLICT": 0.10,
    "IDENTITY_OR_POSITION_CONFLICT": 0.12,
    "LOW_QUALITY_MARGIN": 0.12,
    "CRITICAL_EVIDENCE_MISSING": 0.10,
    "V2_V3_DISAGREEMENT": 0.13,
    "MULTI_KEEPER_BUDGET": 0.09,
    "LOGICAL_MULTI_PHASE": 0.10,
    "ALGORITHM_CHANGED_KEEPER": 0.08,
    "BASELINE_KEEPER_DISPLACED": 0.08,
}


def risk_score(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable monotone score and explainable reason codes."""
    reasons: list[str] = []
    if evidence.get("group_type") != "sha_exact":
        reasons.append("GROUP_NOT_BYTE_IDENTICAL")
    if int(evidence.get("member_count", 0)) >= 5:
        reasons.append("LARGE_GROUP")
    if float(evidence.get("time_span_seconds") or 0) > 8:
        reasons.append("LONG_TIME_SPAN")
    purity = evidence.get("min_pair_similarity")
    if purity is not None and float(purity) < 0.94:
        reasons.append("LOW_EMBEDDING_PURITY")
    local = evidence.get("min_adjacent_similarity")
    if local is not None and float(local) < 0.95:
        reasons.append("LOCAL_CHANGE_SIGNAL")
    if int(evidence.get("face_count_states", 1)) > 1:
        reasons.append("FACE_COUNT_CONFLICT")
    if bool(evidence.get("position_conflict")):
        reasons.append("IDENTITY_OR_POSITION_CONFLICT")
    margin = evidence.get("quality_margin")
    if margin is None or float(margin) < 0.06:
        reasons.append("LOW_QUALITY_MARGIN")
    if int(evidence.get("critical_missing_count", 0)) > 0:
        reasons.append("CRITICAL_EVIDENCE_MISSING")
    if bool(evidence.get("v2_v3_disagreement")):
        reasons.append("V2_V3_DISAGREEMENT")
    if int(evidence.get("keeper_count", 1)) > 1:
        reasons.append("MULTI_KEEPER_BUDGET")
    if int(evidence.get("logical_phase_count", 1)) > 1:
        reasons.append("LOGICAL_MULTI_PHASE")
    if bool(evidence.get("changed")):
        reasons.append("ALGORITHM_CHANGED_KEEPER")
    baseline = set(evidence.get("baseline_keeper_ids", ()))
    candidate = set(evidence.get("candidate_keeper_ids", ()))
    if baseline and not baseline <= candidate:
        reasons.append("BASELINE_KEEPER_DISPLACED")
    score = min(1.0, sum(_REASON_WEIGHTS[reason] for reason in reasons))
    impact = max(1, int(evidence.get("member_count", 1)) - int(evidence.get("keeper_count", 1)))
    expected_value = score * (1.0 + math.log2(impact + 1.0))
    top = sorted(reasons, key=lambda item: (-_REASON_WEIGHTS[item], item))[:3]
    explanation = "; ".join(reason.replace("_", " ").lower() for reason in top)
    coverage_reasons = diagnostic_review_reasons(evidence.get("phase_coverage_diagnostics", []))
    # Structural infeasibility is a review constraint, not a fitted risk weight.
    reasons.extend(code for code in coverage_reasons if code not in reasons)
    if coverage_reasons:
        explanation = "; ".join([*coverage_reasons, explanation]).rstrip("; ")
    return {
        "mandatory_review": bool(coverage_reasons),
        "risk_score": round(score, 6),
        "expected_review_value": round(expected_value, 6),
        "reason_codes": reasons,
        "risk_reason": explanation or "byte-identical or high-agreement control",
    }


def assign_primary_budget(rows: Sequence[dict[str, Any]], budget: float = 0.10) -> list[dict[str, Any]]:
    """Keep the ranked budget plus mandatory diagnostic reviews, with overflow.

    Mandatory cases never evict previously ranked review work or become silent
    due to the review budget. This changes attention only, not keeper authority.
    """
    if not 0 <= budget <= 1:
        raise ValueError("budget must be in [0, 1]")
    count = math.floor(len(rows) * budget)
    ranked = sorted(rows, key=lambda row: (
        -float(row["expected_review_value"]), -float(row["risk_score"]),
        str(row["dataset"]), int(row["group_id"]),
    ))
    chosen = {(row["dataset"], row["group_id"]) for row in ranked[:count]}
    for row in rows:
        budgeted = (row["dataset"], row["group_id"]) in chosen
        mandatory = bool(row.get("mandatory_review"))
        row["review_budget_overflow"] = mandatory and not budgeted
        row["selective_output"] = REVIEW_PRIMARY if budgeted or mandatory else SAFE_SILENT
    return ranked


def frontier(rows: Sequence[Mapping[str, Any]], budgets: Iterable[float],
             error_keys: set[tuple[str, int]] | None = None) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (
        -float(row["expected_review_value"]), -float(row["risk_score"]),
        str(row["dataset"]), int(row["group_id"]),
    ))
    total_members = sum(int(row["member_count"]) for row in ranked)
    labelled = error_keys is not None
    denominator = len(error_keys or ())
    output = []
    for budget in budgets:
        if not 0 <= budget <= 1:
            raise ValueError("budget must be in [0, 1]")
        budget_count = math.floor(len(ranked) * budget)
        selected = [row for index, row in enumerate(ranked)
                    if index < budget_count or row.get("mandatory_review")]
        count = len(selected)
        captured = sum((row["dataset"], int(row["group_id"])) in (error_keys or set())
                       for row in selected)
        output.append({
            "review_budget": budget,
            "review_budget_overflow_groups": count - budget_count,
            "mandatory_review_groups": sum(bool(row.get("mandatory_review")) for row in ranked),
            "review_groups": count,
            "review_group_rate": count / len(ranked) if ranked else 0.0,
            "review_members": sum(int(row["member_count"]) for row in selected),
            "review_member_rate": (
                sum(int(row["member_count"]) for row in selected) / total_members
                if total_members else 0.0
            ),
            "silent_coverage": 1.0 - count / len(ranked) if ranked else 0.0,
            "error_capture_recall": captured / denominator if labelled and denominator else None,
            "error_capture_numerator": captured if labelled else None,
            "error_capture_denominator": denominator if labelled else None,
            "label_scope": "human_tune_subset" if labelled else "unlabelled",
        })
    return output


def one_sided_wilson_upper(errors: int, audited: int, confidence: float = 0.95) -> float | None:
    """One-sided Wilson upper bound; exact zero-error bound for zero errors."""
    if audited <= 0:
        return None
    if not 0 <= errors <= audited:
        raise ValueError("errors must be between zero and audited")
    alpha = 1.0 - confidence
    if errors == 0:
        return 1.0 - alpha ** (1.0 / audited)
    # 95% one-sided normal quantile. Other confidence levels intentionally
    # unsupported rather than silently approximated.
    if confidence != 0.95:
        raise ValueError("only 0.95 confidence is supported")
    z = 1.6448536269514722
    p = errors / audited
    denominator = 1.0 + z * z / audited
    center = p + z * z / (2 * audited)
    radius = z * math.sqrt(p * (1 - p) / audited + z * z / (4 * audited * audited))
    return min(1.0, (center + radius) / denominator)


def add_deciles(rows: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (
        float(row["risk_score"]), str(row["dataset"]), int(row["group_id"])
    ))
    total = len(ordered)
    for index, row in enumerate(ordered):
        row["risk_decile"] = min(10, 1 + (index * 10 // max(1, total)))


def _size_bin(value: int) -> str:
    return "2-3" if value <= 3 else ("4-5" if value <= 5 else "6+")


def stratified_audit_sample(rows: Sequence[dict[str, Any]], sample_size: int,
                            seed: str = "risk-coverage-v4") -> dict[str, Any]:
    """Deterministically sample silent groups over risk/type/size/change strata.

    Sparse exact composite strata are collapsed to decile x change for sampling,
    while all requested dimensions remain in each manifest row for analysis.
    """
    silent = [row for row in rows if row["selective_output"] == SAFE_SILENT]
    sample_size = min(sample_size, len(silent))
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in silent:
        key = f"d{row['risk_decile']}|{'changed' if row['changed'] else 'unchanged'}"
        strata[key].append(row)
    selected: list[dict[str, Any]] = []
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    pools = {key: sorted(value, key=lambda row: (row["dataset"], row["group_id"]))
             for key, value in strata.items()}
    # One per stratum first, then proportional-ish round robin.
    for key in sorted(pools):
        if len(selected) >= sample_size:
            break
        pool = pools[key]
        pick = pool.pop(rng.randrange(len(pool)))
        selected.append(pick)
    while len(selected) < sample_size:
        available = [(len(pool), key) for key, pool in pools.items() if pool]
        if not available:
            break
        _, key = max(available, key=lambda item: (item[0], item[1]))
        pool = pools[key]
        selected.append(pool.pop(rng.randrange(len(pool))))
    selected_keys = {(row["dataset"], row["group_id"]) for row in selected}
    counts = Counter(f"d{row['risk_decile']}|{'changed' if row['changed'] else 'unchanged'}"
                     for row in selected)
    population = Counter(f"d{row['risk_decile']}|{'changed' if row['changed'] else 'unchanged'}"
                         for row in silent)
    manifest_rows = []
    for row in sorted(selected, key=lambda item: (-item["risk_score"], item["dataset"], item["group_id"])):
        key = f"d{row['risk_decile']}|{'changed' if row['changed'] else 'unchanged'}"
        n, k = population[key], counts[key]
        row["selective_output"] = DIAGNOSTIC_SAMPLE
        manifest_rows.append({
            "audit_id": f"{row['dataset']}:G{row['group_id']}",
            "dataset": row["dataset"], "group_id": row["group_id"],
            "member_ids": row["member_ids"], "candidate_keeper_ids": row["candidate_keeper_ids"],
            "risk_score": row["risk_score"], "risk_decile": row["risk_decile"],
            "risk_reason": row["risk_reason"], "reason_codes": row["reason_codes"],
            "group_type": row["group_type"], "size_bin": _size_bin(row["member_count"]),
            "changed": row["changed"], "stratum": key,
            "stratum_population": n, "stratum_sample": k,
            "inclusion_probability": k / n, "analysis_weight": n / k,
            "label_status": "unlabelled", "verdict": None,
        })
    return {
        "seed": seed, "silent_population": len(silent), "sample_count": len(manifest_rows),
        "selected_keys": sorted(f"{dataset}:G{group}" for dataset, group in selected_keys),
        "strata": [{"stratum": key, "population": population[key], "sample": counts[key]}
                   for key in sorted(population)],
        "groups": manifest_rows,
    }
