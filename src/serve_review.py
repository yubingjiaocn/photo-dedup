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

from . import review_assets
from .run_pipeline import serve_review


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--cloud-batch", type=Path, help="explicit read-only cloud batch directory")
    parser.add_argument("--cloud-manifest", type=Path, help="frozen.json binding the batch to source inventories")
    parser.add_argument("--manual-holdout-review", action="store_true", help="Willy manual review only; holdout saves stay sealed")
    args = parser.parse_args(argv)

    if args.cloud_batch:
        if not args.cloud_manifest:
            parser.error("--cloud-batch requires --cloud-manifest")
        try:
            serve_cloud_review(Path(args.output), args.cloud_batch, args.cloud_manifest,
                               args.port, not args.no_open, allow_holdout=args.manual_holdout_review)
            return 0
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"[serve_review][ERROR] {exc}", file=sys.stderr)
            return 1

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


def serve_cloud_review(output, batch, manifest, port=0, open_browser=True, *, allow_holdout=False):
    """Serve bundled UI without installing files into source or output trees."""
    import http.server
    import webbrowser

    from .cloud_review import CloudReviewData
    from .review_server import create_handler

    data = CloudReviewData(output, batch, manifest, allow_holdout=allow_holdout)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), create_handler(data))
    url = f"http://127.0.0.1:{server.server_port}/review.html?mode=cloud"
    print(f"[review] private cloud review: {url}", flush=True)
    print(f"[review] {data.status()['total']} groups; only output/review-state is writable", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        data.close()


if __name__ == "__main__":
    sys.exit(main())
