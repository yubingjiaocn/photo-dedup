"""One-command, review-only entry point for the four-stage pipeline.

Boundaries this module keeps (deliberately, and tested):

* It never imports or calls ``execute_local``; no photo is moved or deleted.
* It writes a temporary runtime config, so the user's ``config.yaml`` is intact.
* The review server starts only **after** every stage finished, because Stage 1
  owns the single sequential HDD pass and browsing mid-run would fight it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import http.server
import platform
import sys
import tempfile
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Sequence

import yaml

from . import (
    db,
    pipeline_report,
    review_server,
    stage0_inventory,
    stage1_features,
    stage2_cluster,
    stage3_report,
    thumbnails,
)
from .config import load_config


class _Tee:
    """Mirror console output to a UTF-8 diagnostic log."""

    def __init__(self, console: Any, log_file: Any) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()


def _print_runtime_diagnostics(args: argparse.Namespace, log_path: Path) -> None:
    """Print enough local runtime context to diagnose a Windows failure."""
    print("\n" + "=" * 72)
    print(f"[diagnostics] run started UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    print(f"[diagnostics] log file: {log_path}")
    print(f"[diagnostics] Python: {sys.version.replace(chr(10), ' ')}")
    print(f"[diagnostics] platform: {platform.platform()}")
    print(f"[diagnostics] executable: {sys.executable}")
    print(
        "[diagnostics] options: "
        f"root={args.root!r}, output={args.output!r}, backend={args.backend!r}, "
        f"limit={args.limit!r}, review_limit={args.review_limit}, "
        f"thumb_px={args.thumb_px}, serve={args.serve}, port={args.port}"
    )
    try:
        import torch

        print(
            "[diagnostics] torch: "
            f"version={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
            f"cuda_version={torch.version.cuda}"
        )
        if torch.cuda.is_available():
            print(f"[diagnostics] GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[diagnostics] torch probe failed: {type(exc).__name__}: {exc}")


def _elapsed(call: Any, /, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call(*args, **kwargs)
    return result, time.perf_counter() - started


def start_review_server(output: Path, port: int = 0) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Bind the local-only paged review server rooted at the output directory."""
    return review_server.start_server(Path(output), port)


def serve_review(output: Path, port: int = 0, open_browser: bool = True) -> None:
    """Serve until Ctrl+C, optionally opening the default browser."""
    server, url = start_review_server(output, port)
    print(f"[pipeline] local review URL: {url}")
    print("[pipeline] the review UI is read-only and reads thumbnails only from the SSD cache")
    if open_browser:
        opened = webbrowser.open(url)
        if not opened:
            print("[pipeline] browser did not open automatically; paste the URL above into a browser")
    print("[pipeline] serving review locally; press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[pipeline] review server stopped")
    finally:
        server.server_close()
        data = getattr(server, "review_data", None)
        if data is not None:
            data.close()


def _runtime_config(root: Path, output: Path, backend: str, thumb_px: int) -> dict[str, Any]:
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
    thumb_cfg = dict(data["features"].get("thumbnails") or {})
    thumb_cfg["enabled"] = True
    thumb_cfg["max_px"] = int(thumb_px)
    data["features"]["thumbnails"] = thumb_cfg
    return data


def _still_image_count(db_path: Path) -> int:
    """Still images in the inventory (thumbnail + ETA denominator)."""
    if not db_path.is_file():
        return 0
    conn = db.open_db(db_path)
    try:
        return db.count_still_images(conn)
    finally:
        conn.close()


def _performance_panel(performance: dict[str, Any], disk: dict[str, Any] | None) -> str:
    """Compact HTML panel embedded in review.html (same numbers as the file)."""
    import html as html_mod

    lines = pipeline_report.render_lines(performance, disk)
    body = "<br>".join(html_mod.escape(line) for line in lines if line)
    return f"<strong>Observed performance and thumbnail disk usage</strong><br>{body}"


def run(
    root: str,
    output: str,
    backend: str = "torch",
    limit: int | None = None,
    review_limit: int = stage3_report.DEFAULT_REVIEW_LIMIT,
    thumb_px: int = thumbnails.DEFAULT_MAX_PX,
) -> dict[str, Any]:
    """Run inventory -> features -> cluster -> report. Never executes deletion."""
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"photo root is not a directory: {root_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if review_limit < 1:
        raise ValueError("review_limit must be at least 1")
    if thumb_px < 1:
        raise ValueError("thumb_px must be at least 1")
    output_path.mkdir(parents=True, exist_ok=True)

    db_path = output_path / "inventory.sqlite"
    pipeline_report.print_thumbnail_plan(output_path, _still_image_count(db_path))

    config = _runtime_config(root_path, output_path, backend, thumb_px)
    with tempfile.TemporaryDirectory(prefix="photo-dedup-") as temp_dir:
        config_path = Path(temp_dir) / "runtime-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"[pipeline] stage 0/4: inventory (limit={limit or 'all'})")
        inventory, stage0_seconds = _elapsed(
            stage0_inventory.run, config_path=str(config_path), limit=limit
        )
        print(f"[pipeline] stage 1/4: features + SSD thumbnails "
              f"(backend={backend}, limit={limit or 'all'})")
        features, stage1_seconds = _elapsed(
            stage1_features.run,
            config_path=str(config_path), backend_override=backend, limit=limit,
        )
        print("[pipeline] stage 2/4: cluster")
        clusters, stage2_seconds = _elapsed(stage2_cluster.run, config_path=str(config_path))

        timings = {
            "stage0": stage0_seconds, "stage1": stage1_seconds,
            "stage2": stage2_seconds, "stage3": 0.0,
        }
        still_images = int(
            inventory.get("library_still_images") or _still_image_count(db_path)
        )
        thumb_stats = dict(features.get("thumbnails") or {})
        disk = pipeline_report.thumbnail_disk_report(output_path, still_images, thumb_stats)
        performance = pipeline_report.build_performance(
            timings, inventory, features, library_still_images=still_images
        )

        print("[pipeline] stage 3/4: build paged review")
        report, stage3_seconds = _elapsed(
            stage3_report.run,
            config_path=str(config_path), review_limit=review_limit,
            performance_panel=_performance_panel(performance, disk),
        )

    # Stage 3's own wall time is measured, then folded into the written report.
    timings["stage3"] = stage3_seconds
    performance = pipeline_report.build_performance(
        timings, inventory, features, library_still_images=still_images
    )
    disk = pipeline_report.thumbnail_disk_report(
        output_path, still_images, dict(report.get("thumbnails") or thumb_stats)
    )
    pipeline_report.write_performance_file(output_path, performance, disk)
    stage3_report.rewrite_performance_panel(
        output_path / "review.html", _performance_panel(performance, disk)
    )

    review = output_path / "review.html"
    result = {
        "inventory": inventory,
        "features": features,
        "clusters": clusters,
        "report": report,
        "performance": performance,
        "thumbnail_disk": disk,
        "review_html": str(review),
    }
    print("\n[pipeline] complete — no photos were moved or deleted")
    print(f"[pipeline] review HTML: {review}")
    print(
        "[pipeline] stats: "
        f"files={inventory.get('files', 0)}, "
        f"processed={features.get('processed', 0)}, "
        f"groups={report.get('groups', 0)}, "
        f"all_timeline={report.get('all_items', 0)}, "
        f"review={report.get('maybe', 0) + report.get('unknown', 0)}, "
        f"proposed_auto_remove={report.get('delete_files', 0)}"
    )
    pipeline_report.print_report(performance, disk)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run stages 0-3 and create a paged review; never move or delete photos."
    )
    parser.add_argument("--root", required=True, help="photo directory to scan")
    parser.add_argument("--output", required=True, help="directory for DB, thumbnails, review")
    parser.add_argument("--backend", choices=("torch", "stub"), default="torch")
    parser.add_argument("--limit", type=int, default=None, help="scan/process at most N files")
    parser.add_argument(
        "--review-limit", type=int, default=stage3_report.DEFAULT_REVIEW_LIMIT,
        help=("static file:// fallback tile count; paged server views are always complete "
              f"(default: {stage3_report.DEFAULT_REVIEW_LIMIT})"),
    )
    parser.add_argument(
        "--thumb-px", type=int, default=thumbnails.DEFAULT_MAX_PX,
        help=f"thumbnail long edge in pixels (default: {thumbnails.DEFAULT_MAX_PX})",
    )
    parser.add_argument("--port", type=int, default=0, help="local review port (default: automatic)")
    parser.add_argument(
        "--serve", action=argparse.BooleanOptionalAction, default=True,
        help="serve the review on 127.0.0.1 after the pipeline (default: enabled)",
    )
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        log_path = output_path / "photo-dedup.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)), \
                    contextlib.redirect_stderr(_Tee(sys.stderr, log_file)):
                _print_runtime_diagnostics(args, log_path)
                try:
                    result = run(
                        args.root, args.output, backend=args.backend, limit=args.limit,
                        review_limit=args.review_limit, thumb_px=args.thumb_px,
                    )
                    if args.serve:
                        serve_review(
                            Path(result["report"]["output_dir"]), args.port, not args.no_open
                        )
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"[pipeline][ERROR] {exc}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    return 1
                except Exception as exc:  # unexpected failures must still be diagnosable
                    print(
                        f"[pipeline][UNEXPECTED ERROR] {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    return 1
        return 0
    except OSError as exc:
        print(f"[pipeline][ERROR] cannot create diagnostic log in {output_path}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
