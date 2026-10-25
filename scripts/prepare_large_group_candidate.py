"""Development-only OFFLINE preparation. No SDK, live state or inventory access.

Reads only allowlisted saved artifacts; writes only candidate/. Does not execute
canaries. Evaluation labels never enter the runtime request builder.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from src import cloud_judge_batch as b
from src import large_group_review as v4
from src import two_pass_review as tp

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "artifacts/large-group-tuning"
SOURCE = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3"
OUT = TUNING / "candidate"
# Offline sampling metadata only; never imported by runtime code.
CANARIES = [
    "2026-04-04-G041",
    "2026-04-04-G035",
    "2026-04-04-G029",
    "2026-04-04-G057",
    "2026-07-25-G018",
    "2026-07-25-G030",
    "2026-04-04-G056",
    "2026-07-31-G028",
]


def prepare_group(group, development):
    b.verify(group in development, "group not in development allowlist")
    case = SOURCE / group
    request = b.read(case / "request.json")
    rows = [x["frame"] for x in request["context"]["temporal"]]
    candidate = v4.build_request(request, rows)
    destination = OUT / "requests" / group
    hashes = {}
    for asset in candidate["images"]:
        src = (case / asset["path"]).resolve()
        b.verify(src.is_relative_to(case.resolve()), "source asset path escape")
        data = src.read_bytes()
        b.verify(
            tp.sha(data) == asset["sha256"] and len(data) == asset["bytes"],
            "source asset changed",
        )
        dst = (destination / asset["path"]).resolve()
        b.verify(
            dst.is_relative_to(destination.resolve()), "candidate asset path escape"
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        hashes[asset["path"]] = tp.sha(data)
    b.save(destination / "request.json", candidate)
    b.save(
        destination / "bindings.json",
        {
            "frames": rows,
            "source_request_sha256": tp.sha((case / "request.json").read_bytes()),
            "assets": hashes,
        },
    )
    state = b.read(case / "state.json")
    # Conservative native-image + UTF-8-byte text accounting, not a provider quote.
    # 20 image tiles * 10k tokens plus <=48k text bytes bounds this packet policy.
    reservation = (248000 * 3 + v4.MAX_OUTPUT_TOKENS * 15) / 1e6
    return {
        "group": group,
        "member_count": len(rows),
        "request": str((destination / "request.json").relative_to(ROOT)),
        "request_sha256": tp.sha(tp.canonical(candidate)),
        "complete_pages": len(rows),
        "baseline_usage": state["usage"],
        "baseline_estimated_cost_usd": state["estimated_cost_usd"],
        "reservation_usd": reservation,
    }


def main():
    split = b.read(TUNING / "split.json")
    labels = b.read(TUNING / "development-labels.json")
    dev = set(split["development_groups"])
    b.verify(
        set(labels["groups"]) == dev and len(dev) == 39, "development split mismatch"
    )
    b.verify(not dev.intersection(split["holdout_groups"]), "split overlap")
    metrics, records = Counter(), []
    for group, label in labels["groups"].items():
        case = SOURCE / group
        answer = b.read(case / "validated.json")
        request = b.read(case / "request.json")
        state = b.read(case / "state.json")
        rows = [x["frame"] for x in request["context"]["temporal"]]
        by_id = {r["file_id"]: r for r in rows}
        ai = {r["frame"]["file_id"] for r in answer["recommended"]}
        human = set(label["human_selected"])
        b.verify(
            ai == set(label["baseline_ai"]) and human <= set(by_id),
            "baseline/label mismatch",
        )
        b.verify(len(rows) == label["member_count"], "member count mismatch")
        for rec in answer["recommended"]:
            b.verify(
                rec["frame"] == by_id[rec["frame"]["file_id"]], "baseline SHA mismatch"
            )
        category = (
            "exact"
            if ai == human
            else "missing_only"
            if ai < human
            else "extra_only"
            if human < ai
            else "replacement_or_mixed"
        )
        metrics[category] += 1
        metrics["missing_frames"] += len(human - ai)
        metrics["extra_frames"] += len(ai - human)
        records.append(
            {
                "group": group,
                "member_count": len(rows),
                "category": category,
                "missing": [by_id[i] for i in sorted(human - ai)],
                "extra": [by_id[i] for i in sorted(ai - human)],
                "human": [by_id[i] for i in sorted(human)],
                "baseline": answer["recommended"],
                "request_sha256": tp.sha((case / "request.json").read_bytes()),
                "validated_sha256": tp.sha((case / "validated.json").read_bytes()),
                "visual_warnings": state.get("visual_warnings", []),
                "complete_pages": sum(
                    a["kind"] == "full_group_page" for a in request["images"]
                ),
                "temporal_equal_values": len(
                    {t["capture_time"] for t in request["context"]["temporal"]}
                )
                < len(rows),
                "source": str(case.relative_to(ROOT)),
            }
        )
    b.save(
        OUT / "development-baseline.json",
        {
            "contract": v4.CONTRACT,
            "label_source_sha256": tp.sha(
                (TUNING / "development-labels.json").read_bytes()
            ),
            "counts": dict(metrics),
            "groups": records,
            "interpretation": "Set disagreement is observable; semantic root causes require visual review, not automatic inference from labels.",
        },
    )
    canaries = [prepare_group(g, dev) for g in CANARIES]
    b.save(
        OUT / "canary-plan.json",
        {
            "contract": v4.CONTRACT,
            "status": "awaiting_parent_authorization",
            "external_calls": 0,
            "max_calls": 8,
            "retry": False,
            "model": b.MODEL,
            "rates_assumption_usd_per_million": {"input": 3, "output": 15},
            "input_reservation_tokens_per_group": 248000,
            "output_limit_tokens": v4.MAX_OUTPUT_TOKENS,
            "reservation_total_usd": round(
                sum(c["reservation_usd"] for c in canaries), 6
            ),
            "groups": canaries,
            "transport_requirement": "Use candidate privacy_instruction and max_output_tokens; do not invoke legacy SonnetJudge (English-only suffix and 4096-token limit). No paid transport supplied in this offline phase.",
            "stop": [
                "Any original/DB/v3 artifact or decision/keeper/delete write",
                "Any holdout access or non-allowlisted group",
                "Any identity/hash/complete-frame mismatch",
                "Any truncated, non-Chinese, invalid census or unbound output: stop, retain raw, no retry",
                "Counted input tokens exceed reservation or rates differ: stop for re-authorization",
                "Any exact control regression or added near-duplicate in G029: stop expansion",
                "After 8 calls maximum: independent parent review; no automatic tuning/replay",
            ],
            "quality_gate": "Existing-exact control G056 and set-correct portions must remain covered; G029 must not gain extras; missing-frame count must fall without increasing total extras across these canaries. This is development selection, never holdout accuracy.",
        },
    )
    print(
        json.dumps(
            {
                "groups": len(records),
                "counts": dict(metrics),
                "prepared_canaries": len(canaries),
                "external_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
