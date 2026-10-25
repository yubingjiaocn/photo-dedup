"""Local file exchange CLI. Never uploads, trains, calls a model or executes choices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.learning_bundle import (
    export_bundle,
    export_original_supplement,
    validate_bundle,
)
from src.learning_io import read_json
from src.learning_shadow import (
    fixture_artifact,
    import_artifact,
    rollback,
    shadow_context,
)
from src.review_events import EventJournal, HoldoutPolicy, backfill


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen", type=Path, default=ROOT / "artifacts/cloud-judge-batch/frozen.json"
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3",
    )
    parser.add_argument("--registry-trust", type=Path, help="explicit trusted fingerprints for a different dataset")
    parser.add_argument("--split-registry", type=Path, action="append", help="explicit split registry for a different dataset")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("backfill")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--journal", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int)
    p = sub.add_parser("validate")
    p.add_argument("--bundle", type=Path, required=True)
    p = sub.add_parser("fixture-artifact")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("import")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("rollback")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--none", action="store_true")
    p = sub.add_parser("shadow")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--group", required=True)
    p = sub.add_parser("export-original-supplement")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument(
        "--confirm",
        action="store_true",
        help="copy ONLY the <=32 requested originals; omit for count/bytes preview",
    )
    args = parser.parse_args(argv)
    frozen = read_json(args.frozen)
    policy = HoldoutPolicy(frozen, args.split_registry, args.registry_trust)
    if args.command == "backfill":
        result = backfill(
            args.snapshot, args.frozen, args.batch, EventJournal(args.journal), policy
        )
    elif args.command == "export":
        print(
            "WARNING: private review images/features; inspect privacy and storage/transfer costs before manual upload",
            file=sys.stderr,
        )
        result = export_bundle(
            EventJournal(args.journal).export_events(),
            frozen,
            args.batch,
            args.destination,
            policy,
            args.start,
            args.end,
        )
    elif args.command == "validate":
        result = validate_bundle(args.bundle, policy)
    elif args.command == "fixture-artifact":
        value = fixture_artifact(args.bundle, args.destination, policy)
        result = {
            "fixture": True,
            "dataset_bundle_sha256": value["dataset_bundle_sha256"],
        }
    elif args.command == "import":
        result = import_artifact(args.artifact, args.bundle, args.output, policy)
    elif args.command == "rollback":
        result = rollback(args.output, args.none)
    elif args.command == "shadow":
        result = shadow_context(args.bundle, args.output, args.group, policy)
    else:
        result = export_original_supplement(
            args.request, args.bundle, frozen, args.destination, policy, args.confirm
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
