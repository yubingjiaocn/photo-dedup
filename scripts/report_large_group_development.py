"""Development-only A/B quality and source-exposure Pareto report; no model calls."""

from __future__ import annotations

from scripts import run_large_group_development as a
from scripts.prepare_large_group_candidate import OUT, TUNING
from src import cloud_judge_batch as b
from src import large_group_projection as projection
from src import two_pass_review as tp


def metrics(records, route):
    true = false = missed = exact = 0
    for item in records:
        human, selected = set(item["human"]), set(item[route])
        true += len(human & selected)
        false += len(selected - human)
        missed += len(human - selected)
        exact += human == selected
    precision = true / (true + false) if true + false else 0
    recall = true / (true + missed) if true + missed else 0
    return {
        "groups": len(records),
        "exact": exact,
        "exact_rate": exact / len(records),
        "precision": precision,
        "recall": recall,
        "f1": 2 * true / (2 * true + false + missed),
        "missing": missed,
        "extra": false,
        "true_positive": true,
    }


def main():
    dev = a.allowed_groups()
    labels = b.read(TUNING / "development-labels.json")["groups"]
    plans = b.read(OUT / "exposure-b-plans.json")["plans"]["balanced"]
    plans = {g["group"]: g for g in plans}
    preflight = b.read(OUT / "exposure-b-evaluation.json")
    eligibility = next(x for x in preflight["routes"] if x["policy"] == "balanced")
    records, receipts = [], []
    all_a_sources, all_b_sources = set(), set()
    parts_a = pixels_a = parts_b = pixels_b = 0
    regressions = []
    for group in dev:
        state = b.read(a.RUN / group / "state.json")
        b.verify(
            state["status"] == "validated", "all 39 A groups must validate before lock"
        )
        request = b.read(a.RUN / group / "request.json")
        b.verify(
            tp.sha(tp.canonical(request)) == state["request_sha256"],
            "saved A request changed",
        )
        source_request = b.read(a.SOURCE / group / "request.json")
        source_rows = [x["frame"] for x in source_request["context"]["temporal"]]
        b.verify(
            projection.build_request(source_request, source_rows) == request,
            "final locked A builder differs from dispatched request",
        )
        answer = b.read(a.RUN / group / "validated.json")
        keep_a = {x["frame"]["file_id"] for x in answer["recommended"]}
        plan = plans[group]
        if len(plan["exposed"]) < plan["members"]:
            case_b = OUT / "development-run-b-r2" / group
            state_b = b.read(case_b / "state.json")
            b.verify(state_b["status"] == "validated", "B result required")
            answer_b = b.read(case_b / "validated.json")
            keep_b = {x["frame"]["file_id"] for x in answer_b["recommended"]}
            receipt_b = str((case_b / "validated.json").relative_to(b.ROOT))
        else:
            # Same packet and contract: reuse A, not a stochastic extra call.
            state_b, keep_b = state, keep_a
            receipt_b = "reuse_identical_A_input_and_result"
        human = set(labels[group]["human_selected"])
        if len(human - keep_b) > len(human - keep_a):
            regressions.append(group)
        records.append(
            {
                "group": group,
                "human": sorted(human),
                "baseline": labels[group]["baseline_ai"],
                "a": sorted(keep_a),
                "b": sorted(keep_b),
                "a_missing": sorted(human - keep_a),
                "b_missing": sorted(human - keep_b),
                "a_extra": sorted(keep_a - human),
                "b_extra": sorted(keep_b - human),
                "members": plan["members"],
                "b_exposed": len(plan["exposed"]),
                "b_receipt": receipt_b,
            }
        )
        all_a_sources.update(state["exposure"]["source_frames"])
        all_b_sources.update(state_b["exposure"]["source_frames"])
        parts_a += state["exposure"]["image_parts"]
        pixels_a += state["exposure"]["pixels"]
        parts_b += state_b["exposure"]["image_parts"]
        pixels_b += state_b["exposure"]["pixels"]
    quality = {route: metrics(records, route) for route in ("baseline", "a", "b")}
    capacity = {
        "a": {
            "vlm_groups": len(dev),
            "members": sum(x["members"] for x in records),
            "unique_source_photos_exposed": len(all_a_sources),
            "high_resolution_candidates": len(all_a_sources),
            "image_parts": parts_a,
            "pixels": pixels_a,
            "human_selected_exposure_recall": 1.0,
        },
        "b": {
            "vlm_groups": len(dev),
            "members": sum(x["members"] for x in records),
            "unique_source_photos_exposed": len(all_b_sources),
            "high_resolution_candidates": len(all_b_sources),
            "image_parts": parts_b,
            "pixels": pixels_b,
            "human_selected_exposure_recall": eligibility[
                "human_selected_exposure_recall"
            ],
        },
    }
    for root in (OUT / "development-run-a", a.RUN, OUT / "development-run-b-r2"):
        for group in dev:
            state_path = root / group / "state.json"
            if state_path.exists():
                state = b.read(state_path)
                receipts.append(
                    {
                        "path": str(state_path.relative_to(b.ROOT)),
                        "status": state["status"],
                        "external_calls": state["external_calls"],
                        "exposed": state["exposure"]["unique_source_photos"],
                        "usage": state.get("usage", {}),
                    }
                )
    reject_b = (
        bool(regressions)
        or eligibility["human_selected_exposure_recall"] < 1.0
        or quality["b"]["missing"] > quality["a"]["missing"]
    )
    result = {
        "contract": projection.CONTRACT,
        "scope": "39 development groups only; assisted labels, no holdout evaluation",
        "quality": quality,
        "capacity": capacity,
        "groups": records,
        "b_decision": "rejected_quality_regression"
        if reject_b
        else "eligible_development_pareto_candidate",
        "b_missing_regression_groups": regressions,
        "actual_experiment": {
            "calls": sum(r["external_calls"] for r in receipts),
            "photo_submissions": sum(r["exposed"] for r in receipts),
            "input_tokens": sum(r["usage"].get("inputTokens", 0) for r in receipts),
            "output_tokens": sum(r["usage"].get("outputTokens", 0) for r in receipts),
            "receipts": receipts,
        },
        "interpretation": "Exact and P/R/F1 are development set agreement, not blinded accuracy. B unchanged groups reuse A deterministically. Contact sheets count every source, not one exposure. No development dollar cap.",
        "lock": {
            "candidate": projection.CONTRACT,
            "semantic_or_prompt_revisions_remaining": 0,
            "holdout_access": False,
            "live_runner_enabled": False,
            "files": {
                path: tp.sha((b.ROOT / path).read_bytes())
                for path in [
                    "src/large_group_review.py",
                    "src/large_group_projection.py",
                    "src/photo_exposure.py",
                    "scripts/run_large_group_development.py",
                    "schemas/large-group-judge-v4-candidate-2.schema.json",
                ]
            },
        },
    }
    b.save(OUT / "development-comparison.json", result)
    print(
        {"quality": quality, "capacity": capacity, "b_decision": result["b_decision"]}
    )


if __name__ == "__main__":
    main()
