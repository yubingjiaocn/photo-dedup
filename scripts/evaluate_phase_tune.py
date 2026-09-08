#!/usr/bin/env python3
"""Evaluate baseline/v2/v3 reports against anonymous tune annotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.offline_evaluation import evaluate_predictions


def _reports(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for dataset in ("disney", "jx3"):
        value = json.loads((root / f"{dataset}.json").read_text(encoding="utf-8"))
        records.update({f"{dataset}:G{row['group_id']}": row for row in value["groups"]})
    return records


def _predictions(
    annotations: list[dict[str, Any]], records: dict[str, dict[str, Any]], *, baseline: bool,
) -> dict[str, dict[str, Any]]:
    predictions = {}
    for annotation in annotations:
        row = records[annotation["group_id"]]
        if baseline:
            phases = {"0": [str(value) for value in row["member_ids"]]}
            keepers = row["baseline_keeper_ids"]
            review = []
        else:
            phases = {
                str(phase["phase_id"]): [str(value) for value in phase["member_ids"]]
                for phase in row["logical_phases"]
            }
            keepers = row["candidate_keeper_ids"]
            review = row["member_ids"] if row.get("review_required") else []
        predictions[annotation["group_id"]] = {
            "keepers": [str(value) for value in keepers],
            "phases": phases,
            "review_member_ids": [str(value) for value in review],
        }
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation_document = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations = annotation_document["annotations"]
    v2_records, v3_records = _reports(args.v2), _reports(args.v3)
    events = {"disney", "jx3"}
    result = {
        "schema_version": 1,
        "annotation_status": annotation_document["annotation_status"],
        "annotation_group_count": len(annotations),
        "baseline": evaluate_predictions(
            annotations, _predictions(annotations, v2_records, baseline=True),
            expected_events=events,
        ),
        "v2": evaluate_predictions(
            annotations, _predictions(annotations, v2_records, baseline=False),
            expected_events=events,
        ),
        "v3": evaluate_predictions(
            annotations, _predictions(annotations, v3_records, baseline=False),
            expected_events=events,
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
