"""Exactly G004/G007, Sonnet only, no retry or frozen-tree writes."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_vlm_rebuild import digest
from src import db
from src import two_pass_review as tp


def run(eval_root, output, execute=False):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("repo output only")
    output.mkdir(parents=True, exist_ok=True)
    rates = {
        "input_per_million": 3.0,
        "output_per_million": 15.0,
        "basis": "Illustrative Sonnet-family planning assumption only; Sonnet 5.5 Bedrock tariff not independently verified. Not invoice cost.",
    }
    adapter = tp.BedrockAdapter(
        output, model_id="global.anthropic.claude-sonnet-5-5", rates=rates
    )
    old = ROOT / "artifacts/multi-subject-narrow"
    old_hashes = {p.name: tp.sha(p.read_bytes()) for p in old.glob("*.json")}
    groups = []
    for date, gid, expected in (
        ("2026-07-25", 4, {470, 481}),
        ("2026-04-05", 7, {35, 131}),
    ):
        inventory = (
            eval_root
            / "astra-regime-validation-20260910/development"
            / date
            / "inventory.sqlite"
        )
        database_sha = tp.sha(inventory.read_bytes())
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            before = digest(conn)
            rows = [dict(r) for r in db.group_page_by_id(conn, gid)]
            assert {r["file_id"] for r in rows} == expected
            sources = {
                r["file_id"]: Path(
                    conn.execute(
                        "SELECT path FROM files WHERE id=?", (r["file_id"],)
                    ).fetchone()[0]
                )
                for r in rows
            }
            assert all(
                p.resolve().is_relative_to(eval_root.resolve())
                for p in sources.values()
            )
            group = f"{date}-G{gid:03d}"
            if execute:
                result = tp.run_real(
                    rows, sources, output / group, adapter=adapter, group=group
                )
            else:
                # Input checks only; zero transport calls, no fake output.
                result = {"group": group, "mode": "preflight_only", "frames": []}
                for r in rows:
                    assert (
                        tp.sha(sources[r["file_id"]].read_bytes())
                        == r["content_sha256"]
                    )
                    result["frames"].append(
                        {
                            k: r[k]
                            for k in ("file_id", "content_sha256", "width", "height")
                        }
                    )
            result["decision_tables_before"] = before
            result["decision_tables_after"] = digest(conn)
            assert before == result["decision_tables_after"]
        assert database_sha == tp.sha(inventory.read_bytes())
        result["frozen_database_sha256_unchanged"] = database_sha
        groups.append(result)
    assert old_hashes == {p.name: tp.sha(p.read_bytes()) for p in old.glob("*.json")}
    report = {
        "mode": "real_bedrock_review_only" if execute else "preflight_only",
        "groups": groups,
        "external_calls": len(adapter.calls),
        "maximum_calls": 4,
        "old_real_json_hashes_unchanged": old_hashes,
        "production_enabled": False,
        "preference_accuracy": "not_computed_no_Willy_labels",
        "estimated_cost_usd": round(
            sum(c.get("estimated_cost_usd") or 0 for c in adapter.calls), 8
        ),
        "cost_basis": rates,
    }
    (output / ("report.json" if execute else "preflight.json")).write_bytes(
        tp.canonical(report)
    )
    print(
        json.dumps(
            {
                "groups": [
                    {
                        "group": g["group"],
                        "A": g.get("pass_A_status"),
                        "B": g.get("pass_B_status"),
                        "blockers": g.get("blockers"),
                    }
                    for g in groups
                ],
                "external_calls": len(adapter.calls),
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/two-pass-real-pilot"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize at most four Sonnet Bedrock calls",
    )
    args = parser.parse_args()
    run(args.eval_root, args.output, args.execute)
