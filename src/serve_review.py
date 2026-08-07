"""Serve an existing review output without re-running any pipeline stage.

Usage::

    python -m src.serve_review --output "C:\\photo-review"

The command refreshes only the committed front-end build (``review.html`` and
hashed ``assets/*``) inside the output directory, then starts the loopback review
server against the existing ``inventory.sqlite``, thumbnails and
``review_state.json``. It never runs Stage 0-3 and never modifies an original.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import review_assets
from .run_pipeline import serve_review


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open an existing photo-dedup review without re-running the pipeline."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="existing output directory containing inventory.sqlite and thumbnails",
    )
    parser.add_argument("--port", type=int, default=0,
                        help="local review port (default: automatic)")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the browser automatically")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    db_path = output / "inventory.sqlite"
    if not output.is_dir():
        print(f"[serve_review][ERROR] output directory not found: {output}", file=sys.stderr)
        return 1
    if not db_path.is_file():
        print(f"[serve_review][ERROR] inventory DB not found: {db_path}", file=sys.stderr)
        return 1

    try:
        manifest = review_assets.install_into(output)
        print(f"[serve_review] refreshed review UI ({len(manifest['assets'])} assets)")
        serve_review(output, port=args.port, open_browser=not args.no_open)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[serve_review][ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
