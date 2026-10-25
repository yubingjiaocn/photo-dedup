"""Authorized 12-group stratified canary; continue only through explicit safety gate."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cloud_judge_batch as b

EXECUTION_POLICY = {
    "version": "review-only-continuation-v3",
    "single_group_quality_and_cost_are_advisory": True,
    "canary_thresholds": "diagnostic_only",
    "hard_stops": [
        "source_or_frozen_inventory_mutation",
        "source_identity_mismatch",
        "path_escape",
        "duplicate_dispatch",
        "ten_consecutive_groups_without_effective_response",
    ],
    "automatic_retries": False,
    "concurrency": 1,
}


def choose_canary(frozen, router):
    routed = {e["group"] for e in router["groups"] if e["status"] == "routed"}
    chosen = []
    for date in b.DATES:
        groups = [g for g in frozen["groups"] if g["date"] == date]
        known = sorted(
            [g for g in groups if g["group"] in routed], key=lambda g: g["group"]
        )
        small = (
            known[0]
            if known
            else min(groups, key=lambda g: (len(g["frames"]), g["group"]))
        )
        large = min(
            [g for g in groups if g["group"] != small["group"]],
            key=lambda g: (-len(g["frames"]), g["group"]),
        )
        chosen.extend([small["group"], large["group"]])
    b.verify(len(chosen) == len(set(chosen)) == 12, "exactly 12 canary groups required")
    return chosen


def gate(states, *, finished):
    reasons = []
    if any(s["decision_table_writes"] for s in states):
        reasons.append("decision_table_write_detected")
    for s in states:
        if s["external_calls"] and (
            s["estimated_cost_usd"] is None or s["estimated_cost_usd"] > 0.20
        ):
            reasons.append("unknown_or_over_0.20_estimated_cost:" + s["group"])
        if s["reserved"] and not s.get("response_sha256"):
            reasons.append("transport_or_uncertain_failure:" + s["group"])
    invalid = [s for s in states if s["state"] != "validated"]
    if len(invalid) > 2:
        reasons.append("more_than_two_invalid_canary_groups")
    # Repeated structural/provider failure family is a systemic anomaly, not a retry target.
    failures = collections.Counter(
        (s.get("reason") or "unknown").split(":", 1)[0] for s in invalid
    )
    if any(n >= 2 for n in failures.values()):
        reasons.append("repeated_failure_family")
    valid = sum(s["state"] == "validated" for s in states)
    if finished and (len(states) != 12 or valid < 10):
        reasons.append("canary_requires_at_least_10_validated_of_12")
    return {
        "passed": finished and not reasons,
        "early_stop": bool(reasons),
        "validated": valid,
        "examined": len(states),
        "reasons": reasons,
        "max_estimated_cost_usd": max(
            (s["estimated_cost_usd"] or 0 for s in states), default=0
        ),
    }


def continuation_gate(states, output):
    """Quality/cost failures are warnings; only source safety or ten empty responses stop."""
    reasons = []
    for state in states:
        if state["decision_table_writes"]:
            reasons.append("source_decision_write_detected:" + state["group"])
        if state["external_calls"] > 1:
            reasons.append("duplicate_dispatch_detected:" + state["group"])
        # Rejected model references are single-group semantic failures. Source/packet
        # identity failures before a provider receipt are infrastructure safety failures.
        if not state.get("response_sha256") and any(
            token in (state.get("reason") or "")
            for token in (
                "source SHA mismatch",
                "frozen membership mismatch",
                "path escape",
                "asset changed",
                "source/region/context mismatch",
                "source images mismatch",
            )
        ):
            reasons.append("source_identity_or_path_safety:" + state["group"])
    empty = 0
    for state in reversed(states):
        effective = state["state"] == "validated"
        if not effective and state.get("response_sha256"):
            try:
                receipt = b.read(
                    Path(output) / state["group"] / "provider-response.json"
                )
                effective = any(
                    isinstance(block.get("text"), str) and block["text"].strip()
                    for block in receipt.get("output", {})
                    .get("message", {})
                    .get("content", [])
                )
            except (OSError, ValueError, TypeError, AttributeError):
                effective = False
        if effective:
            break
        empty += 1
    if empty >= 10:
        reasons.append("ten_consecutive_groups_without_effective_response")
    return {
        "hard_stop": bool(reasons),
        "reasons": reasons,
        "consecutive_without_effective_response": empty,
        "quality_and_cost_failures_are_warnings": True,
    }


def checkpoint(output, frozen, report, phase):
    """Durable aggregate progress/usage plus one ledger row per consumed slot."""
    indexed = {g["group"]: g for g in frozen["groups"]}
    consumed = [s for s in report["groups"] if s["reserved"]]
    strata = {}
    for field in ("date", "group_size", "scene", "decision", "validation"):
        buckets = {}
        for state in consumed:
            group = indexed[state["group"]]
            value = {
                "date": group["date"],
                "group_size": str(len(group["frames"])),
                "scene": state["scene_route"],
                "decision": state["route"],
                "validation": state["state"],
            }[field]
            bucket = buckets.setdefault(
                value, {"groups": 0, "validated": 0, "estimated_cost_usd": 0}
            )
            bucket["groups"] += 1
            bucket["validated"] += state["state"] == "validated"
            bucket["estimated_cost_usd"] = round(
                bucket["estimated_cost_usd"] + (state["estimated_cost_usd"] or 0), 8
            )
        strata[field] = buckets
    progress = {
        "phase": phase,
        "consumed_groups": len(consumed),
        "total_groups": len(frozen["groups"]),
        "inherited_calls": sum(
            s["external_calls"] for s in consumed if s.get("imported_from")
        ),
        "new_calls": sum(
            s["external_calls"] for s in consumed if not s.get("imported_from")
        ),
        "external_calls_total": report["external_calls_total"],
        "usage": report["usage"],
        "estimated_cost_usd_known": report["estimated_cost_usd_known"],
        "cost_unknown_groups": report["cost_unknown_groups"],
        "strata": strata,
        "decision_table_writes": report["decision_table_writes"],
        "preference_accuracy": None,
        "preference_accuracy_reason": "No Willy human labels.",
    }
    b.save(output / "progress.json", progress)
    b.save(output / "call-ledger.json", consumed)
    b.save(output / "summary.json", progress)


def run(frozen_path, router_path, output, *, execute=False, fixture=None):
    b.verify(execute or fixture is not None, "explicit execute required")
    frozen, router = b.read(frozen_path), b.read(router_path)
    output = b.artifact_root(output)
    b.save(output / "execution-policy.json", EXECUTION_POLICY)
    canary = choose_canary(frozen, router)
    plan = {
        "snapshot_sha256": frozen["snapshot_sha256"],
        "canary": canary,
        "strata": "two per unique date: available local coarse scene (otherwise smallest group), plus largest other group",
        "minimum_validated": 10,
        "canary_size": 12,
        "max_estimated_group_cost_usd": 0.20,
        "concurrency": 1,
        "total_unique_groups": 194,
        "zero_decision_writes_required": True,
        "no_automatic_retry": True,
    }
    path = output / "authorization-plan.json"
    if path.exists():
        b.verify(b.read(path) == plan, "authorization plan changed; no replay")
    else:
        b.save(path, plan)
    stopped = output / "stopped.json"
    if stopped.exists():
        return b.read(stopped)
    states = []
    report = None
    for gid in canary:
        report = b.run(
            frozen_path,
            router_path,
            output,
            execute=execute,
            fixture=fixture,
            concurrency=1,
            selected_groups=[gid],
        )
        checkpoint(output, frozen, report, "canary")
        state = next(s for s in report["groups"] if s["group"] == gid)
        states.append(state)
        check = gate(states, finished=len(states) == 12)
        b.save(output / "canary-gate.json", check)
        print(
            json.dumps(
                {
                    "phase": "canary",
                    "group": gid,
                    "state": state["state"],
                    "cost": state["estimated_cost_usd"],
                    "validated": check["validated"],
                    "examined": len(states),
                    "quality_warning": check["early_stop"],
                }
            ),
            flush=True,
        )
        safety = continuation_gate(states, output)
        b.save(output / "continuation-gate.json", safety)
        if safety["hard_stop"]:
            result = {
                "status": "stopped_safety_or_ten_empty_responses",
                "gate": safety,
                "external_calls_total": report["external_calls_total"],
                "report": "report.json",
            }
            b.save(stopped, result)
            return result
    b.save(output / "canary-report.json", report)
    remaining = [g["group"] for g in frozen["groups"] if g["group"] not in canary]
    for gid in remaining:
        report = b.run(
            frozen_path,
            router_path,
            output,
            execute=execute,
            fixture=fixture,
            concurrency=1,
            selected_groups=[gid],
        )
        checkpoint(output, frozen, report, "remainder")
        state = next(s for s in report["groups"] if s["group"] == gid)
        print(
            json.dumps(
                {
                    "phase": "remainder",
                    "group": gid,
                    "state": state["state"],
                    "cost": state["estimated_cost_usd"],
                    "external_calls_total": report["external_calls_total"],
                }
            ),
            flush=True,
        )
        states.append(state)
        safety = continuation_gate(states, output)
        b.save(output / "continuation-gate.json", safety)
        if safety["hard_stop"]:
            result = {
                "status": "stopped_safety_or_ten_empty_responses",
                "gate": safety,
                "group": gid,
                "external_calls_total": report["external_calls_total"],
                "report": "report.json",
            }
            b.save(stopped, result)
            return result
    result = {
        "status": "complete",
        "external_calls_total": report["external_calls_total"],
        "groups": 194,
        "report": "report.json",
    }
    checkpoint(output, frozen, report, "complete")
    b.save(output / "completion.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frozen", type=Path, default=ROOT / "artifacts/cloud-judge-batch/frozen.json"
    )
    p.add_argument("--scene-router", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            run(args.frozen, args.scene_router, args.output, execute=args.execute),
            indent=2,
        )
    )
