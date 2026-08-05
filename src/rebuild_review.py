"""Rebuild review workbench: re-cluster and regenerate review pages.

For Windows users with separate config: ensures Stage2+3 use the custom config
that points to the correct inventory.sqlite and output directory without
re-running Stage0/1 (which would rescan/recompute features).

Usage:
    python -m src.rebuild_review --config run-config.yaml --root E:\\Photos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import stage2_cluster
from . import stage3_report


def rebuild(config_path: str, root_override: Optional[str] = None) -> int:
    """Run stage2 then stage3 serially, stopping on first failure.

    MUST fail closed if the config-resolved DB path:
    - doesn't exist (not created by Stage 0/1)
    - lacks schema_version/stage1_done_at in meta table
    - has no valid features rows

    This prevents typo in paths.db from creating an empty DB or overwriting
    existing review.html/delete_local.txt with an invalid rebuild.
    """
    from pathlib import Path
    from . import db
    from .config import load_config

    print(f"[rebuild_review] starting with config={config_path}, root={root_override}")

    # Load config to resolve the DB path
    cfg = load_config(config_path)
    db_path = Path(cfg.db_path)

    # Fail closed: DB must exist before we run Stage 2
    if not db_path.is_file():
        raise FileNotFoundError(
            f"[rebuild_review] FATAL: inventory DB not found: {db_path}\n"
            f"Stage 2+3 must not create a new database. Run Stage 0+1 first to build the feature cache."
        )

    # Verify DB structure before opening with open_db (which would create schema)
    import sqlite3
    conn_ro = sqlite3.connect(str(db_path), uri=True)
    try:
        tables = {r[0] for r in conn_ro.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "files" not in tables or "features" not in tables or "meta" not in tables:
            raise ValueError(
                f"[rebuild_review] FATAL: {db_path} missing essential tables (files/features/meta).\n"
                f"This is not a valid photo-dedup inventory. Run Stage 0+1 first."
            )
    finally:
        conn_ro.close()

    # Now open normally to verify Stage 1 completion
    conn_check = db.open_db(db_path)
    try:
        stage1_done = db.get_meta(conn_check, "stage1_done_at")
        if not stage1_done:
            raise ValueError(
                f"[rebuild_review] FATAL: {db_path} has no stage1_done_at in meta table.\n"
                f"Stage 1 has not completed. Run it first to populate the features table."
            )
        # Verify there are actual feature records
        feature_count = conn_check.execute(
            "SELECT COUNT(*) AS n FROM features WHERE status = 'done'"
        ).fetchone()["n"]
        if feature_count == 0:
            raise ValueError(
                f"[rebuild_review] FATAL: {db_path} has 0 features rows with status='done'.\n"
                f"Stage 1 produced no usable features. Run Stage 0+1 first on a valid photo library."
            )
        print(f"[rebuild_review] verified: DB exists, schema OK, stage1 done, {feature_count} features")
    finally:
        conn_check.close()

    # Stage2: clustering
    print("[rebuild_review] running stage2_cluster...")
    stage2_cluster.run(config_path=config_path, root_override=root_override)

    # Stage3: report generation
    print("[rebuild_review] running stage3_report...")
    stage3_report.run(config_path=config_path, root_override=root_override)

    print("[rebuild_review] complete")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild review workbench (Stage2+3) with custom config"
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to run-config.yaml (must specify paths.db and paths.output_dir)",
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Photo root directory (e.g., E:\\Photos on Windows)",
    )
    args = ap.parse_args(argv)

    if not Path(args.config).exists():
        print(f"[rebuild_review] config not found: {args.config}")
        return 1

    return rebuild(args.config, args.root)


if __name__ == "__main__":
    sys.exit(main())
