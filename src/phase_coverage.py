"""Metadata-only phase coverage diagnostics, never keeper/deletion authority.

Requirements describe a complete disjoint partition of a group. Provenance and
confidence are declarations from the caller, not inferred/calibrated here. Even
low-confidence shadow partitions remain visible; merging them requires separate
evidence and must not silently erase a coverage conflict.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

PHASE_COVERAGE_BUDGET_CONFLICT = "PHASE_COVERAGE_BUDGET_CONFLICT"
PHASE_COVERAGE_GAP = "PHASE_COVERAGE_GAP"


def keeper_budget(member_count: int, max_group_keepers: int = 3) -> int:
    """The existing logarithmic/max-keepers/n-1 selection cap (zero if empty)."""
    if type(member_count) is not int or member_count < 0:
        raise ValueError("member_count must be a nonnegative integer")
    if type(max_group_keepers) is not int or max_group_keepers < 1:
        raise ValueError("max_group_keepers must be a positive integer")
    if not member_count:
        return 0
    budget = min(max_group_keepers, 1 + int(math.log2(member_count)))
    return min(budget, member_count - 1) if member_count > 1 else budget


def _ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(type(i) is not int for i in value):
        raise ValueError(f"{name} must contain integer member IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate member IDs")
    return list(value)


def diagnose_phase_coverage(
    member_ids: Sequence[int], keeper_ids: Sequence[int], budget: int,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare declared phase requirements with an actual frozen keeper budget.

    No phase producer is promoted to human/trusted status, and no confidence
    label grants keep-all. A structural conflict or observed coverage gap forces
    review only. Inputs must be complete; malformed evidence raises ValueError.
    """
    members = _ids(member_ids, "member_ids")
    keepers = _ids(keeper_ids, "keeper_ids")
    if not set(keepers) <= set(members):
        raise ValueError("keeper_ids must belong to member_ids")
    if type(budget) is not int or not 0 <= budget <= len(members):
        raise ValueError("budget must be an integer within the group size")
    if members and (not keepers or budget < 1):
        raise ValueError("nonempty groups require a keeper and positive budget")
    if len(keepers) > budget:
        raise ValueError("keeper count exceeds stated budget")
    if not isinstance(requirements, Mapping):
        raise ValueError("requirements must be a mapping")
    if requirements.get("source") not in {"shadow", "human", "trusted", "analyst_interpretation"}:
        raise ValueError("unsupported phase evidence source")
    if requirements.get("confidence") not in {"uncalibrated", "low", "high"}:
        raise ValueError("explicit phase confidence required")
    if not isinstance(requirements.get("evidence_ref"), str) or not requirements["evidence_ref"].strip():
        raise ValueError("nonempty evidence_ref required")
    phases = requirements.get("phases")
    if not isinstance(phases, (list, tuple)):
        raise ValueError("phases must be a complete partition")
    seen: set[int] = set()
    phase_ids: set[str] = set()
    uncovered = []
    coverage = []
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("each phase must be a mapping")
        pid = phase.get("phase_id")
        if not isinstance(pid, str) or not pid or pid in phase_ids:
            raise ValueError("phase IDs must be nonempty unique strings")
        phase_ids.add(pid)
        ids = _ids(phase.get("member_ids"), "phase member_ids")
        if not ids or seen.intersection(ids):
            raise ValueError("phases must be nonempty and disjoint")
        seen.update(ids)
        retained = [i for i in keepers if i in ids]
        coverage.append({"phase_id": pid, "member_ids": ids, "keeper_ids": retained})
        if not retained:
            uncovered.append(pid)
    if seen != set(members):
        raise ValueError("phases must partition all and only the group members")
    conflict = len(phases) > budget
    reasons = ([PHASE_COVERAGE_BUDGET_CONFLICT] if conflict else
               [PHASE_COVERAGE_GAP] if uncovered else [])
    return {
        "schema_version": 1, "policy": "A_BUDGET_UNCHANGED_REVIEW",
        "authority": "shadow_review_only", "auto_remove_authority": "BYTE_IDENTICAL_ONLY",
        "member_ids": members, "keeper_ids": keepers, "keeper_budget": budget,
        "required_phase_count": len(phases), "budget_conflict": conflict,
        "minimum_extra_keepers": max(0, len(phases) - budget),
        "keep_all_extra_keepers": len(members) - len(keepers),
        "uncovered_phase_ids": uncovered, "phase_coverage": coverage,
        "mandatory_review": bool(reasons), "reason_codes": reasons,
        "evidence": copy.deepcopy(dict(requirements)),
    }


def diagnostic_review_reasons(diagnostics: Sequence[Mapping[str, Any]]) -> list[str]:
    """Only diagnostic outputs with an explicit conflict/gap force review."""
    return list(dict.fromkeys(
        code for item in diagnostics if item.get("mandatory_review")
        for code in item.get("reason_codes", [])
        if code in {PHASE_COVERAGE_BUDGET_CONFLICT, PHASE_COVERAGE_GAP}
    ))
