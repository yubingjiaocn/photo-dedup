"""Pure local revalidation of immutable real A/B receipts. Never calls a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jsonschema

from src import two_pass_review as tp


def run(pilot, output):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.is_relative_to(pilot):
        raise tp.ReviewError("new artifacts directory outside immutable pilot required")
    ledger = tp.strict_json((pilot / "call-ledger.json").read_text())
    report = {"external_calls_this_run": 0, "decision_table_writes": 0, "groups": []}
    schema = json.loads(
        (ROOT / "schemas/two-pass-deep-judge-v1.schema.json").read_text()
    )
    for group in ("2026-07-25-G004", "2026-04-05-G007"):
        hashes, packets = {}, {}
        for phase, directory in (
            ("A", pilot / group),
            ("B", pilot / "revalidate-v2" / group),
        ):
            entries = [e for e in ledger if e["group"] == group and e["pass"] == phase]
            if len(entries) != 1:
                raise tp.ReviewError("one receipt per group/pass required")
            entry = entries[0]
            for suffix, field in (
                ("wire-request.json", "request_sha256"),
                ("request.json", "payload_sha256"),
                ("provider-response.json", "response_sha256"),
                ("raw.txt", "response_text_sha256"),
            ):
                p = directory / f"pass-{phase}-{suffix}"
                actual = tp.sha(p.read_bytes())
                if actual != entry[field]:
                    raise tp.ReviewError(f"immutable receipt hash mismatch: {p}")
                hashes[str(p.relative_to(pilot))] = actual
            response = tp.strict_json(
                (directory / f"pass-{phase}-provider-response.json").read_text()
            )
            text = (directory / f"pass-{phase}-raw.txt").read_text()
            if (
                response["stopReason"] != "end_turn"
                or response["usage"] != entry["usage"]
                or text
                != "\n".join(
                    b["text"] for b in response["output"]["message"]["content"]
                )
            ):
                raise tp.ReviewError("provider/raw receipt mismatch")
            packets[phase] = tp.strict_json(text)
            tp.privacy_check(packets[phase])
        a, b = packets["A"], packets["B"]
        rows = [c["frame"] for c in a["coverage"]]
        a_request = tp.strict_json((pilot / group / "pass-A-request.json").read_text())
        if a_request["context"]["frames"] != [tp.identity(r) for r in rows]:
            raise tp.ReviewError("raw A coverage/source binding mismatch")
        jsonschema.validate(
            a,
            json.loads(
                (ROOT / "schemas/two-pass-subject-map-v1.schema.json").read_text()
            ),
        )
        validation = tp.validate_map(a, rows)
        request = tp.strict_json(
            (pilot / "revalidate-v2" / group / "pass-B-request.json").read_text()
        )
        if request["context"]["scene_map"] != a:
            raise tp.ReviewError("A/B binding mismatch")
        jsonschema.validate(b, schema)
        validated = tp.validate_judgment(
            b, rows, a, validation, request["context"]["evidence"]
        )
        if validated != b or validated["decision"] != "attention_only":
            raise tp.ReviewError("expected unchanged real attention-only baseline")
        case = output / group
        case.mkdir(parents=True, exist_ok=True)
        for phase in ("A", "B"):
            source = pilot / group if phase == "A" else pilot / "revalidate-v2" / group
            (case / f"pass-{phase}-raw.txt").write_bytes(
                (source / f"pass-{phase}-raw.txt").read_bytes()
            )
            (case / f"pass-{phase}-raw.json").write_bytes(tp.canonical(packets[phase]))
        (case / "pass-A-validated.json").write_bytes(
            tp.canonical({"map": a, "validation": validation})
        )
        (case / "pass-B-validated.json").write_bytes(tp.canonical(validated))
        bindings = {
            e["region"]: validation["lookup"][e["region"]]["frame"]
            for d in b["discriminators"]
            for e in d["evidence"]
        }
        item = {
            "group": group,
            "decision": b["decision"],
            "raw_unchanged": True,
            "hashes_verified": hashes,
            "evidence_frame_bindings": bindings,
            "text_lengths": [len(d["text"]) for d in b["discriminators"]],
        }
        (case / "provenance.json").write_bytes(tp.canonical(item))
        report["groups"].append(item)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_bytes(tp.canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot", type=Path, default=ROOT / "artifacts/two-pass-real-pilot"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/cloud-judge-batch/real-baseline",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.pilot, args.output), indent=2))
