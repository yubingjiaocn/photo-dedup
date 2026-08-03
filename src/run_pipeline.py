"""One-command, review-only entry point for the four-stage pipeline."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml

from .config import load_config
from . import stage0_inventory, stage1_features, stage2_cluster, stage3_report


def _runtime_config(root: Path, output: Path, backend: str) -> dict[str, Any]:
    """Build an in-memory override without changing the user's config.yaml."""
    data = load_config().as_dict()
    data["paths"] = dict(data["paths"])
    data["paths"].update(
        root=str(root),
        db=str(output / "inventory.sqlite"),
        output_dir=str(output),
    )
    data["features"] = dict(data["features"])
    data["features"]["backend"] = backend
    return data


def run(root: str, output: str, backend: str = "torch", limit: int | None = None) -> dict[str, Any]:
    """Run inventory -> features -> cluster -> report. Never executes deletion."""
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"photo root is not a directory: {root_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    output_path.mkdir(parents=True, exist_ok=True)

    config = _runtime_config(root_path, output_path, backend)
    with tempfile.TemporaryDirectory(prefix="photo-dedup-") as temp_dir:
        config_path = Path(temp_dir) / "runtime-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"[pipeline] stage 0/4: inventory (limit={limit or 'all'})")
        inventory = stage0_inventory.run(config_path=str(config_path), limit=limit)
        print(f"[pipeline] stage 1/4: features (backend={backend}, limit={limit or 'all'})")
        features = stage1_features.run(
            config_path=str(config_path), backend_override=backend, limit=limit
        )
        print("[pipeline] stage 2/4: cluster")
        clusters = stage2_cluster.run(config_path=str(config_path))
        print("[pipeline] stage 3/4: build review")
        report = stage3_report.run(config_path=str(config_path))

    review = output_path / "review.html"
    result = {
        "inventory": inventory,
        "features": features,
        "clusters": clusters,
        "report": report,
        "review_html": str(review),
    }
    print("\n[pipeline] complete — no photos were moved or deleted")
    print(f"[pipeline] review HTML: {review}")
    print(
        "[pipeline] stats: "
        f"files={inventory.get('files', 0)}, "
        f"processed={features.get('processed', 0)}, "
        f"groups={report.get('groups', 0)}, "
        f"review={report.get('maybe', 0) + report.get('unknown', 0)}, "
        f"proposed_auto_remove={report.get('delete_files', 0)}"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run stages 0-3 and create a review report; never move or delete photos."
    )
    parser.add_argument("--root", required=True, help="photo directory to scan")
    parser.add_argument("--output", required=True, help="directory for DB and review files")
    parser.add_argument("--backend", choices=("torch", "stub"), default="torch")
    parser.add_argument("--limit", type=int, default=None, help="scan/process at most N files")
    args = parser.parse_args(argv)
    try:
        run(args.root, args.output, backend=args.backend, limit=args.limit)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[pipeline][ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
