"""Measure frozen text layers only; never plans ROI, renders images, or calls models."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_vlm_rebuild import digest
from src import db, vlm_review


def run(eval_root, output):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("writes must stay inside repository")
    frozen = sorted(eval_root.glob("*/**/vlm-review/groups/*.json"))
    if len(frozen) != 22:
        raise ValueError("requires frozen 22-sidecar set")
    reports = []
    for sidecar in frozen:
        inventory = sidecar.parents[2] / "inventory.sqlite"
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            before = digest(conn)
            report = vlm_review.context_report(db.group_page_by_id(conn, int(sidecar.stem)))
            report.update(dataset=sidecar.parents[2].name, decision_tables_before=before,
                          decision_tables_after=digest(conn))
            reports.append(report)
    result = vlm_review.aggregate_context_reports(reports)
    result["roi_planning_attempts"] = 0
    result["decision_tables_unchanged"] = all(r["decision_tables_before"] == r["decision_tables_after"] for r in reports)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fable-roi-narrow/context-layers-22.json")
    args = parser.parse_args()
    run(args.eval_root, args.output)
