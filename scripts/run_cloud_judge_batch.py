"""Prepare or explicitly execute the frozen single-cloud-judge batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cloud_judge_batch as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="write a read-only frozen inventory snapshot, no models",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/home/ubuntu/photo-dedup-eval/astra-regime-validation-20260910"),
    )
    parser.add_argument(
        "--frozen", type=Path, default=ROOT / "artifacts/cloud-judge-batch/frozen.json"
    )
    parser.add_argument("--scene-router", "--handoff", dest="handoff", type=Path)
    parser.add_argument(
        "--confirmed-visuals",
        type=Path,
        help="separate operator-confirmed strip boxes; never local model output",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/cloud-judge-batch/prepared"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicit one Sonnet call per eligible group; requires parent/user authorization",
    )
    parser.add_argument("--concurrency", type=int, default=1, choices=range(1, 5))
    parser.add_argument(
        "--fixture-responses",
        type=Path,
        help="offline responses mapping; never uses network",
    )
    parser.add_argument("--input-rate", type=float, default=3.0)
    parser.add_argument("--output-rate", type=float, default=15.0)
    parser.add_argument(
        "--rate-basis",
        default="Illustrative planning assumption; unverified tariff, not invoice cost.",
    )
    args = parser.parse_args()
    if args.freeze:
        if args.execute or args.handoff or args.fixture_responses:
            parser.error("freeze is a standalone zero-model action")
        batch.artifact_root(args.frozen.parent)
        data = batch.snapshot(args.eval_root)
        if args.frozen.exists() and batch.read(args.frozen) != data:
            parser.error("existing frozen snapshot differs; refusing overwrite")
        batch.save(args.frozen, data)
        print(json.dumps(data["totals"]))
        return
    if args.handoff is None:
        parser.error(
            "--handoff required; scene-router manifest required; no implicit local inference"
        )
    fixture = (
        batch.FixtureJudge(batch.read(args.fixture_responses))
        if args.fixture_responses
        else None
    )
    report = batch.run(
        args.frozen,
        args.handoff,
        args.output,
        execute=args.execute,
        concurrency=args.concurrency,
        rates={
            "input_per_million": args.input_rate,
            "output_per_million": args.output_rate,
            "basis": args.rate_basis,
        },
        fixture=fixture,
        confirmed_visuals_path=args.confirmed_visuals,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
