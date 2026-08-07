r"""Rebuild the review workbench: re-run Stage 2 + Stage 3 only.

Same command shape as :mod:`src.run_pipeline`, on purpose::

    python -m src.rebuild_review --root E:\Photos --output F:\photo-review

``--root``/``--output`` are the two things the user already knows; the paths the
stages need (``paths.root``, ``paths.db``, ``paths.output_dir``) are derived from
them by :mod:`src.runtime_config`, exactly as ``run_pipeline`` derives them. There
is no hand-maintained ``run-config.yaml``, and no way for a rebuild to cluster a
database other than ``OUTPUT/inventory.sqlite``.

``--config`` is optional and means *tunables only* (cluster thresholds, quality
weights); any ``paths`` it declares are ignored. Stage 1 knobs (``backend``,
thumbnail size) are deliberately not settable here: the feature cache already
exists and a rebuild must not restate how it was computed.

Stage 0/1 are never run, so this never re-reads the photo library. The run fails
closed unless the resolved database exists, has the schema, and records a
finished Stage 1 with usable feature rows.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from . import db, root_scope, runtime_config, stage2_cluster, stage3_report


def verify_inventory(db_path: Path, root: Path) -> int:
    """Fail closed unless ``db_path`` is a finished Stage 0/1 inventory for ``root``.

    Returns the number of usable feature rows. Refuses when the database:

    * doesn't exist (Stage 2 must never *create* one -- a path typo would
      otherwise produce an empty DB and overwrite a good review.html);
    * lacks the ``files``/``features``/``meta`` tables;
    * is bound to a different photo root than ``--root``
      (:class:`src.root_scope.ParameterError`);
    * has no ``stage1_done_at``, or no ``status='done'`` feature rows.

    The checks are ordered cheapest-and-most-read-only first: the root binding is
    verified before :func:`src.db.open_db`, which would migrate the schema, so a
    rejected rebuild leaves the output directory byte-for-byte as it found it.
    """
    if not db_path.is_file():
        raise FileNotFoundError(
            f"[rebuild_review] FATAL: inventory DB not found: {db_path}\n"
            f"Stage 2+3 must not create a new database. "
            f"Run Stage 0+1 first (python -m src.run_pipeline) to build the feature cache."
        )

    # Structure check before db.open_db(), which would create the schema.
    conn_ro = sqlite3.connect(str(db_path), uri=True)
    try:
        tables = {r[0] for r in conn_ro.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "files" not in tables or "features" not in tables or "meta" not in tables:
            raise ValueError(
                f"[rebuild_review] FATAL: {db_path} missing essential tables "
                f"(files/features/meta).\n"
                f"This is not a valid photo-dedup inventory. Run Stage 0+1 first."
            )
    finally:
        conn_ro.close()

    # Same gate stage 2/3 apply, but before open_db(): a rebuild must never
    # cluster an inventory belonging to a different photo root.
    root_scope.preflight(db_path, root)

    conn_check = db.open_db(db_path)
    try:
        if not db.get_meta(conn_check, "stage1_done_at"):
            raise ValueError(
                f"[rebuild_review] FATAL: {db_path} has no stage1_done_at in meta table.\n"
                f"Stage 1 has not completed. Run it first to populate the features table."
            )
        feature_count = int(conn_check.execute(
            "SELECT COUNT(*) AS n FROM features WHERE status = 'done'"
        ).fetchone()["n"])
        if feature_count == 0:
            raise ValueError(
                f"[rebuild_review] FATAL: {db_path} has 0 features rows with status='done'.\n"
                f"Stage 1 produced no usable features. "
                f"Run Stage 0+1 first on a valid photo library."
            )
    finally:
        conn_check.close()
    print(f"[rebuild_review] verified: DB exists, schema OK, stage1 done, {feature_count} features")
    return feature_count


def rebuild(root: str, output: str, config_path: Optional[str] = None) -> int:
    """Rebuild Stage 2 + Stage 3 for one ``--root``/``--output`` pair.

    The inventory must already be ``OUTPUT/inventory.sqlite``. That is asserted
    twice -- by :func:`verify_inventory` here and by
    :func:`src.runtime_config.written`, which re-resolves the generated config the
    way the stages will -- so a rebuild can only touch the database belonging to
    the given output directory, bound to the given root.

    Raises before running any stage: :class:`FileNotFoundError` (no inventory),
    :class:`ValueError` (not a finished Stage 1 inventory), or
    :class:`src.root_scope.ParameterError` (bound to a different root).
    """
    root_path, output_path, db_path = runtime_config.resolve_paths(root, output)
    # Explicit tunables file must exist; load_config would otherwise fall back to
    # the packaged defaults without saying so.
    runtime_config.base_config_path(config_path)
    print(
        f"[rebuild_review] starting with root={root_path}, output={output_path}, "
        f"tunables={config_path or 'default config.yaml'}"
    )
    verify_inventory(db_path, root_path)
    # No backend/thumb_px: a rebuild must not restate Stage 1 parameters.
    with runtime_config.written(
        root_path, output_path, base_config=config_path
    ) as runtime_path:
        print("[rebuild_review] running stage2_cluster...")
        stage2_cluster.run(config_path=str(runtime_path), root_override=str(root_path))
        print("[rebuild_review] running stage3_report...")
        stage3_report.run(config_path=str(runtime_path), root_override=str(root_path))
    print("[rebuild_review] complete")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild the review workbench (Stage 2+3 only) for one --root/--output pair; "
            "never re-reads photos, never deletes anything."
        )
    )
    ap.add_argument(
        "--root", required=True,
        help="photo directory the output directory belongs to (e.g. E:\\Photos)",
    )
    ap.add_argument(
        "--output", required=True,
        help=("directory holding inventory.sqlite, thumbnails and the review "
              "(the same value you passed to run_pipeline --output)"),
    )
    ap.add_argument(
        "--config", default=None,
        help=("optional tunables base (cluster thresholds, quality weights); its paths are "
              "ignored, --root/--output decide them (default: the repository config.yaml)"),
    )
    args = ap.parse_args(argv)

    if args.config is not None and not Path(args.config).expanduser().is_file():
        print(f"[rebuild_review][ERROR] config not found: {args.config}", file=sys.stderr)
        return 1

    # Fail closed with the reason on stderr and an exit status, not a traceback.
    try:
        return rebuild(args.root, args.output, config_path=args.config)
    except root_scope.ParameterError as exc:
        print(f"[rebuild_review][ERROR] {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"[rebuild_review][ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
