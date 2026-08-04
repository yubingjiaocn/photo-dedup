"""Stage 1 setup and finalisation: everything either side of the batch loop.

Split out of :mod:`src.stage1_features` so that file is about the loop and this
one is about the bookkeeping around it — admission, the pending-work query, the
optional detectors, and the stats/meta written at the end. Each step keeps its
own telemetry phase, so a slow SSD listing or a slow admission pass shows up as
itself rather than as "unaccounted".

Nothing here runs off the main thread: every function touches the SQLite
connection, the thumbnail cache, or model construction.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from . import db, eye_detection, feature_admission, root_scope, thumbnails


def admit(conn: Any, cfg: Config, scope: Any, telemetry: Any) -> Tuple[List[Any], Dict[str, int]]:
    """Record oversize/extreme-aspect exclusions before any decode happens."""
    max_process_mp = float(cfg.features.get("max_process_megapixels", 64))
    if max_process_mp <= 0:
        raise ValueError("features.max_process_megapixels must be greater than zero")
    max_aspect_ratio = float(cfg.features.get("max_process_aspect_ratio", 3.0))
    if max_aspect_ratio < 1:
        raise ValueError("features.max_process_aspect_ratio must be at least 1")
    with telemetry.phase("setup_admission"):
        skipped = feature_admission.mark_unprocessable_skipped(
            conn, int(max_process_mp * 1_000_000), max_aspect_ratio, scope=scope
        )
        conn.commit()
    counts = {
        reason: sum(row["skip_reason"] == reason for row in skipped)
        for reason in ("PIXEL_LIMIT", "ASPECT_RATIO")
    }
    for row in skipped:
        print(f"[stage1][SKIP {row['skip_reason']}] {row['path']} — "
              f"{row['width']}x{row['height']}, {row['megapixels']:.2f} MP, "
              f"ratio={row['aspect_ratio']:.3f}")
    return skipped, counts


def pending_work(conn: Any, thumbnailer: Any, scope: Any, telemetry: Any,
                 limit: Optional[int]) -> List[Any]:
    """Rows still needing Stage 1, in inventory order.

    With thumbnails enabled, one cheap SSD listing decides which cached JPEGs
    actually exist, so a deleted thumbnail is regenerated and a stale one is
    never reused.
    """
    if thumbnailer is None:
        with telemetry.phase("setup_pending_query"):
            pending = list(db.iter_files_for_features(conn, scope=scope))
    else:
        with telemetry.phase("setup_thumb_listing"):
            db.register_thumb_presence(conn, thumbnails.ids_on_disk(thumbnailer.directory))
        with telemetry.phase("setup_pending_query"):
            pending = list(db.iter_files_for_features(
                conn, thumb_max_px=thumbnailer.max_px, scope=scope))
    return pending[:limit] if limit else pending


def build_thumbnailer(cfg: Config, telemetry: Any = None) -> Optional[thumbnails.Thumbnailer]:
    """Construct the SSD thumbnail cache writer unless it is disabled."""
    thumb_cfg = cfg.features.get("thumbnails", {})
    if not isinstance(thumb_cfg, dict):
        thumb_cfg = {}
    if thumb_cfg.get("enabled", True) is False:
        return None
    return thumbnails.Thumbnailer(
        thumbnails.thumbs_dir(cfg.output_dir),
        max_px=int(thumb_cfg.get("max_px", thumbnails.DEFAULT_MAX_PX)),
        jpeg_quality=int(thumb_cfg.get("jpeg_quality", thumbnails.DEFAULT_JPEG_QUALITY)),
        telemetry=telemetry,
    )


def build_eye_detector(cfg: Config, has_work: bool) -> Any:
    """The opt-in eye detector, or None. Never constructed with nothing to do."""
    eye_cfg = cfg.features.get("eye_detection", {})
    if not has_work or not eye_cfg.get("enabled", False):
        return None
    return eye_detection.MediaPipeEyeDetector(eye_cfg, cfg.models_dir)


def finalize(conn: Any, cfg: Config, scope: Any, telemetry: Any, *, backend_name: str,
             skipped: List[Any], skip_counts: Dict[str, int], thumbnailer: Any,
             settings: Any, done: int, failed: int, total: int,
             loop_started: float) -> Dict[str, Any]:
    """Write meta, print the run summary, and assemble the stats dict."""
    db.set_meta(conn, "stage1_backend", backend_name)
    db.set_meta(conn, "stage1_skipped_oversize", str(len(skipped)))
    db.set_meta(conn, "stage1_skipped_pixel_limit", str(skip_counts["PIXEL_LIMIT"]))
    db.set_meta(conn, "stage1_skipped_aspect_ratio", str(skip_counts["ASPECT_RATIO"]))
    db.set_meta(conn, "stage1_done_at", str(int(time.time())))
    elapsed = time.time() - loop_started
    rate = done / elapsed if elapsed > 0 else 0.0
    print(f"[stage1] processed {done} images ({failed} unreadable), skipped={len(skipped)} "
          f"(PIXEL_LIMIT={skip_counts['PIXEL_LIMIT']}, "
          f"ASPECT_RATIO={skip_counts['ASPECT_RATIO']}) "
          f"in {elapsed:.1f}s ({rate:.1f}/s)")
    with telemetry.phase("finalize_stats"):
        scope_summary = root_scope.summary(conn, scope)
        thumb_recorded = (None if thumbnailer is None
                          else db.thumbnail_stats(conn, scope=scope))
    stats: Dict[str, Any] = {
        "processed": done, "failed": failed, "succeeded": done - failed,
        "total": total, "skipped_oversize": len(skipped),
        "skipped_pixel_limit": skip_counts["PIXEL_LIMIT"],
        "skipped_aspect_ratio": skip_counts["ASPECT_RATIO"],
        "max_process_megapixels": float(cfg.features.get("max_process_megapixels", 64)),
        "max_process_aspect_ratio": float(cfg.features.get("max_process_aspect_ratio", 3.0)),
        "scope": scope_summary,
        # The knobs the run actually used, so a benchmark or a bug report never
        # has to guess whether prefetch was on.
        "loop_settings": {
            "batch_size": settings.batch_size,
            "cpu_workers": settings.cpu_workers,
            "prefetch_batches": settings.prefetch_batches,
            "iqa_batch_size": settings.iqa_batch_size,
            "prefetch_enabled": settings.prefetch_enabled,
            "inflight_batches": settings.inflight_batches,
            "memory_note": settings.memory_note(),
        },
    }
    if thumbnailer is None:
        stats["thumbnails"] = thumbnails.disabled_stats()
    else:
        thumb_stats = thumbnailer.stats()
        thumb_stats.update(thumb_recorded or {})
        db.set_meta(conn, "stage1_thumb_max_px", str(thumbnailer.max_px))
        stats["thumbnails"] = thumb_stats
        print(
            f"[stage1] thumbnails: created={thumb_stats['created']} "
            f"reused={thumb_stats['reused']} failed={thumb_stats['failed']} "
            f"cache={thumb_stats['cache_files']} files / "
            f"{thumb_stats['cache_bytes'] / (1024 ** 3):.2f} GiB"
        )
        if thumb_stats["failed"]:
            print(
                f"[stage1][WARN] {thumb_stats['failed']} thumbnail(s) failed this run; "
                "those items show an explicit 'thumbnail unavailable' tile in the review UI"
            )
    return stats
