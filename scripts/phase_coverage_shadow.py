#!/usr/bin/env python3
"""Replay coverage diagnostics on frozen JSON; no DB, media or embeddings.

The before/after A/B here means diagnostic-off/on, not policy A/B. Keeper
selection and logical partitioning are frozen. Optional external requirements
are a separate evidence lane, never a training set or replacement partition.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase_coverage import diagnose_phase_coverage, diagnostic_review_reasons, keeper_budget
from src.risk_coverage import DIAGNOSTIC_SAMPLE, REVIEW_PRIMARY, assign_primary_budget, frontier, risk_score


def replay(rows: list[dict[str, Any]], requirements: list[dict[str, Any]] | None = None,
           primary_budget: float = .10) -> dict[str, Any]:
    """Apply default A to each frozen group and expose mandatory queue overflow.

    Existing diagnostic samples are retained rather than resampled; if one is
    promoted to mandatory review its original bucket remains in before_output.
    This is not a new probability audit or a recalculated population risk rate.
    """
    extra = {}
    keys = {(row["dataset"], row["group_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate group keys")
    for item in requirements or []:
        key = (item["dataset"], item["group_id"])
        if key not in keys or key in extra:
            raise ValueError("requirements key missing or duplicated")
        extra[key] = item["requirements"]
    after = copy.deepcopy(rows)
    for old, row in zip(rows, after, strict=True):
        members, keepers = row["member_ids"], row["candidate_keeper_ids"]
        if row["member_count"] != len(members):
            raise ValueError("member_count mismatch")
        if row["logical_phase_count"] != len(row["logical_phases"]):
            raise ValueError("logical_phase_count mismatch")
        default_budget = 1 if row.get("group_type") == "sha_exact" else keeper_budget(len(members))
        budget = row.get("group_keeper_budget", default_budget)
        row["keeper_budget_source"] = ("cached_explicit" if "group_keeper_budget" in old else
                                      "baseline_default_reconstructed")
        shadow = {"source": "shadow", "confidence": "uncalibrated",
                  "evidence_ref": "frozen-json.logical_phases", "phases": [
                      {"phase_id": str(p["phase_id"]), "member_ids": p["member_ids"],
                       "reason_codes": p.get("reason_codes", [])} for p in row["logical_phases"]]}
        diagnostics = [diagnose_phase_coverage(members, keepers, budget, shadow)]
        key = (row["dataset"], row["group_id"])
        if key in extra:
            diagnostics.append(diagnose_phase_coverage(members, keepers, budget, extra[key]))
        row["group_keeper_budget"] = budget
        row["phase_coverage_diagnostics"] = diagnostics
        row["risk_evidence"]["phase_coverage_diagnostics"] = diagnostics
        row.update(risk_score(row["risk_evidence"]))
        row["before_output"] = old["selective_output"]
        row["review_required"] = bool(old["review_required"] or row["mandatory_review"])
        if row["mandatory_review"]:
            row["review_tier"] = "primary"
            row["review_reason"] = ";".join(diagnostic_review_reasons(diagnostics))
    assign_primary_budget(after, primary_budget)
    for old, row in zip(rows, after, strict=True):
        if row["selective_output"] != REVIEW_PRIMARY and old["selective_output"] == DIAGNOSTIC_SAMPLE:
            row["selective_output"] = DIAGNOSTIC_SAMPLE
    conflict_rows = [row for row in after
                     if any(d["budget_conflict"] for d in row["phase_coverage_diagnostics"])]
    mandatory = [row for row in after if row["mandatory_review"]]
    promotions = [row for row in after if row["selective_output"] == REVIEW_PRIMARY
                  and row["before_output"] != REVIEW_PRIMARY]
    n_members = sum(row["member_count"] for row in rows)
    n_keepers = sum(len(row["candidate_keeper_ids"]) for row in rows)
    after_keepers = sum(len(row["candidate_keeper_ids"]) for row in after)
    metrics = {
        "groups": len(rows), "members": n_members, "candidate_keepers_before": n_keepers,
        "candidate_keepers_after": after_keepers,
        "keeper_changed_groups": sum(a["candidate_keeper_ids"] != b["candidate_keeper_ids"]
                                     for a, b in zip(rows, after, strict=True)),
        "retention_before": n_keepers / n_members if n_members else 0,
        "retention_after": after_keepers / n_members if n_members else 0,
        "retention_delta_keepers": after_keepers - n_keepers,
        "conflict_groups": len(conflict_rows),
        "conflict_keys": [f'{r["dataset"]}:G{r["group_id"]}' for r in conflict_rows],
        "mandatory_review_groups": len(mandatory),
        "mandatory_review_members": sum(row["member_count"] for row in mandatory),
        "review_required_before": sum(bool(row["review_required"]) for row in rows),
        "review_required_after": sum(bool(row["review_required"]) for row in after),
        "selective_before": dict(Counter(row["selective_output"] for row in rows)),
        "selective_after": dict(Counter(row["selective_output"] for row in after)),
        "new_primary_groups": len(promotions),
        "new_primary_members": sum(row["member_count"] for row in promotions),
        "review_budget_overflow_groups": sum(row["review_budget_overflow"] for row in after),
        "primary_members_before": sum(row["member_count"] for row in rows if row["selective_output"] == REVIEW_PRIMARY),
        "primary_members_after": sum(row["member_count"] for row in after if row["selective_output"] == REVIEW_PRIMARY),
        "keep_all_conflict_upper_bound_extra_keepers": sum(
            row["member_count"] - len(row["candidate_keeper_ids"]) for row in conflict_rows),
        "population_risk": None, "population_risk_upper95": None,
    }
    return {"schema_version": 1, "policy": "A_BUDGET_UNCHANGED_REVIEW",
            "authority": "shadow_review_only", "auto_remove_authority": "BYTE_IDENTICAL_ONLY",
            "replay_scope": "frozen_selection_diagnostics_not_pipeline_rerun",
            "metrics": metrics, "groups": after,
            "frontier": frontier(after, [0, .03, .05, .10, .20])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.input] + ([args.requirements] if args.requirements else [])
    if args.output.resolve() in {p.resolve() for p in paths}:
        parser.error("output must not overwrite input evidence")
    data = {str(p): p.read_bytes() for p in paths}
    result = replay(json.loads(data[str(args.input)])["groups"],
                    json.loads(data[str(args.requirements)]) if args.requirements else None)
    result["input_sha256"] = {p: hashlib.sha256(b).hexdigest() for p, b in data.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
