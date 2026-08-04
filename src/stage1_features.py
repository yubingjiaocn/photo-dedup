"""Stage 1 -- feature extraction (the GPU-heavy, HDD-bound stage).

For every still image (``file_kind`` in jpg / jpg_motion) this computes:

* ``phash``            -- 64-bit perceptual hash (exact-dup detection)
* ``dinov2_embedding`` -- 768-d semantic vector (burst / scene similarity)
* ``quality_score``    -- MUSIQ 0..100 (which frame to keep)
* ``quality_meta``     -- JSON: sharpness, CLIP-IQA, face_quality, backend
* ``faces_json``       -- YuNet bbox + landmarks + score per face

Results stream into the ``features`` table, committing every
``scan.commit_every`` images, so a Ctrl+C (or crash) resumes exactly where it
stopped -- only rows without a ``status='done'`` feature row are reprocessed.

Two interchangeable backends (see :mod:`src.stage1_backends`)
-----------------------------------------------------------
* **torch** -- real DINOv2 + pyiqa (MUSIQ/CLIP-IQA) + OpenCV YuNet. Needs the
  GPU stack from requirements.txt.
* **stub**  -- deterministic numpy-only features. Used by the test suite and
  for no-GPU dry runs. Same output schema, so stages 2/3 don't care which ran.

``features.backend: auto`` uses torch if importable, else stub.

One read, one decode, many outputs
----------------------------------
Each photo is read from the HDD exactly once (hashing that same sequential
read) and decoded exactly once. The embedding, IQA, faces, exposure, scene
routing **and** the SSD review thumbnail are all produced from that one decoded
``PIL.Image``, which is what makes a 100k-photo library on a single mechanical
disk practical.

Two lanes, bounded
------------------
:mod:`src.stage1_pipeline` reads and prepares the *next* batches while this loop
runs the current one's models. The reader stays single-threaded (one mechanical
disk, one sequential pass), the per-image CPU preparation goes to a small worker
pool, and everything stateful — YuNet, the thumbnail cache, every SQLite
statement — stays on this thread. Output order is the inventory order in every
configuration, so the DB writes, the error rows and resume behaviour are
unchanged. ``features.cpu_workers: 0`` restores the original single-threaded
pipeline exactly.

Scope + telemetry
-----------------
Work is restricted to the photo root this output directory is bound to
(:mod:`src.root_scope`), so a mismatched ``--root`` fails closed instead of
silently processing another library's rows, and an unbound output directory either
adopts the supplied root or the stage refuses to run.

:mod:`src.stage1_telemetry` accumulates one ``perf_counter`` pair per
instrumented call (per image, per batch, or per one-off setup step -- the report
states which). Setup and model load are timed separately from the batch loop, and
both are reconciled against Stage 1's outer wall time, so nothing hides.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config, load_config
from . import db, root_scope
from . import quality as Q
from . import exposure
from . import decision
from . import eye_detection
from . import feature_admission
from . import thumbnails
from . import stage1_telemetry as telemetry_mod
from . import stage1_pipeline
from .scene_router import DecodeState, RoutingInput, SceneRouter
from .stage1_backends import (  # noqa: F401 (re-exported for existing callers/tests)
    DEFAULT_IQA_BATCH_SIZE,
    EMBED_DIM,
    StubBackend,
    TorchBackend,
    _load_yunet,
    resolve_backend,
)
from .stage1_settings import resolve_loop_settings

try:  # progress bar is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x


# ===========================================================================
# Main loop
# ===========================================================================

def _open_image_and_sha(path: Path, recorder: Any = None) -> Tuple[Image.Image, str]:
    """Read compressed bytes once, hashing that same sequential read before decode.

    ``recorder`` is anything exposing ``phase``/``add``: the ``Telemetry`` object
    when this runs on the main thread, or a producer recorder when it runs on the
    reader thread. Passing the recorder rather than the collector is what keeps a
    reader thread out of the main thread's accumulators.
    """
    def phase(key: str) -> Any:
        return recorder.phase(key) if recorder is not None else telemetry_mod.NULL_SPAN

    digest = hashlib.sha256()
    data = io.BytesIO()
    hash_seconds = 0.0
    with phase("source_open_read"):
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                started = time.perf_counter()
                digest.update(chunk)
                hash_seconds += time.perf_counter() - started
                data.write(chunk)
        data.seek(0)
    if recorder is not None:
        # Hashing rides along the same sequential read, so its cost is already
        # inside the read span. Bill it to its own phase and subtract it from the
        # read span: the columns stay additive instead of double counting. The
        # call is counted even when it rounds to zero, so every per-image phase of
        # a successful read reports the same number of calls.
        recorder.add("hash_sha256", hash_seconds)
        recorder.add("source_open_read", -hash_seconds, count=0)
    with phase("decode"):
        img = Image.open(data)
        img = img.convert("RGB")
    return img, digest.hexdigest()


def _process_batch(
    backend: Any,
    rows: Sequence[Any],
    cfg: Config,
    eye_detector: Any = None,
    scene_router: Any = None,
    thumbnailer: Any = None,
    telemetry: Any = None,
) -> List[Dict[str, Any]]:
    """Compute features for one batch of file rows. Returns feature dicts.

    Reads, decodes and prepares the batch inline (no producer threads) and then
    runs :func:`_consume_batch`. This is the single-batch entry point used by the
    tests and by any caller that already has rows in hand; :func:`run` uses the
    bounded producer instead and calls :func:`_consume_batch` directly.
    """
    source = stage1_pipeline.build_source([rows], backend, cfg, _open_image_and_sha,
                                          workers=0, prefetch_batches=0, telemetry=telemetry)
    out_rows: List[Dict[str, Any]] = []
    for prepared in source:
        out_rows.extend(_consume_batch(
            backend, prepared, cfg, eye_detector=eye_detector, scene_router=scene_router,
            thumbnailer=thumbnailer, telemetry=telemetry,
        ))
    return out_rows


def _consume_batch(
    backend: Any,
    prepared: Any,
    cfg: Config,
    eye_detector: Any = None,
    scene_router: Any = None,
    thumbnailer: Any = None,
    telemetry: Any = None,
) -> List[Dict[str, Any]]:
    """Main-thread half of one batch: models, stateful detectors, feature rows.

    The router is constructed at batch scope when the caller did not retain a
    run-scoped instance.  It receives the same decoded RGB ``Image`` used by
    embeddings and technical detectors; it never opens the source path.

    The optional ``thumbnailer`` gets that *same* decoded object too, so the
    SSD thumbnail cache costs no additional HDD read and no second decode.

    Everything touched here is main-thread-only by design: YuNet holds a mutable
    input size, the thumbnail writer accumulates rows for one DB statement, and
    the eye detector and scene provider are single-instance models.
    """
    router = scene_router if scene_router is not None else _build_scene_router(cfg)
    out_rows: List[Dict[str, Any]] = []

    def phase(key: str, count: int = 1) -> Any:
        return telemetry.phase(key, count) if telemetry is not None else telemetry_mod.NULL_SPAN

    valid: List[Any] = []
    for item in prepared.items:
        if item.ok:
            valid.append(item)
            continue
        # The failed attempt still cost real time (a partial read, a decode
        # that raised, the failure bookkeeping). Instrument it instead of
        # letting it disappear into unaccounted time.
        with phase("error_handling"):
            row = item.row
            print(f"[stage1][WARN] cannot read {row['path']}: {item.error}")
            routing = router.route(RoutingInput(
                metadata=_routing_metadata(row), decode_state=DecodeState.FAILED
            ))
            if thumbnailer is not None:
                thumbnailer.record_source_failure(
                    row, f"{thumbnails.SOURCE_DECODE_FAILED}: "
                         f"{type(item.error).__name__}: {item.error}"
                )
            out_rows.append(_error_feature_row(int(row["id"]), routing=routing))

    if not valid:
        return out_rows

    # One embedding call and one IQA call per distinct bounded shape for the whole
    # batch, instead of two model calls plus two host syncs per image.
    embeddings = backend.embed_prepared([item.backend for item in valid])
    qualities = backend.quality_prepared([item.backend for item in valid])
    for item, emb, (score, meta) in zip(valid, embeddings, qualities):
        row, image = item.row, item.image
        faces = backend.faces_prepared(item.backend, image)
        with phase("face_quality"):
            img_rgb = np.asarray(image)
            meta["face_quality"] = Q.compute_face_quality(img_rgb, faces)
        meta["schema_version"] = 2
        with phase("exposure"):
            # The calibrated 512 px resize already happened during preparation;
            # the metrics themselves need the face boxes, so they run here.
            meta["exposure"] = exposure.extract_prepared(
                item.exposure_rgb, image.size, faces,
                int(cfg.features.get("exposure_long_edge", 512)),
            )
        meta["detectors"] = decision.detector_extensions()
        if eye_detector is not None:
            with phase("eye_detection"):
                try:
                    meta["eye_detection"] = eye_detector.analyze(
                        image, expected_face_count=len(faces)
                    )
                except Exception:
                    meta["eye_detection"] = eye_detection.unavailable_result(
                        "DETECTOR_INFERENCE_FAILED"
                    )
        # Shadow-only, additive metadata.  The provider is intentionally None
        # until SigLIP is implemented; SceneRouter then emits MODEL_UNAVAILABLE
        # without model loading, downloads, or network access.
        with phase("scene_routing"):
            meta["routing"] = router.route(RoutingInput(
                metadata=_routing_metadata(row), image=image
            ))
        # Same decoded object, no reopen: the SSD thumbnail is a by-product of
        # the decode the extractors above already paid for.
        if thumbnailer is not None:
            thumbnailer.capture(row, image)
        with phase("other_cpu"):
            out_rows.append(
                {
                    "file_id": int(row["id"]),
                    "phash": item.phash,
                    "content_sha256": item.sha256,
                    "dinov2_embedding": Q.embedding_to_blob(emb),
                    "quality_score": float(score),
                    "quality_meta": json.dumps(meta),
                    "face_count": len(faces),
                    "faces_json": json.dumps(faces),
                    "status": "done",
                }
            )
    return out_rows


def _build_scene_router(cfg: Config, provider: Any = None) -> SceneRouter:
    """Construct the optional offline provider only when explicitly enabled."""
    scene_cfg = cfg.features.get("scene_routing", {})
    if provider is None and isinstance(scene_cfg, dict) and scene_cfg.get("enabled") is True:
        try:
            from .siglip_runtime import build_siglip_provider

            provider = build_siglip_provider(cfg)
        except Exception:
            # Missing dependencies, artifacts, hashes, or CUDA all fail closed.
            provider = None
    return SceneRouter(cfg, provider=provider)


def _routing_metadata(row: Any) -> Dict[str, Any]:
    """Extract only present inventory fields needed by the metadata gate.

    ``sqlite3.Row`` supports subscription and ``keys()`` but not ``get``;
    tests and callers may use ordinary mappings.  Keeping this adapter narrow
    avoids adding invented values to routing metadata or changing DB schema.
    """
    fields = (
        "file_kind", "motion_partner_id", "motion_photo", "is_motion_photo",
        "motion_partner_path", "paired_asset_id",
    )
    if isinstance(row, dict):
        return {field: row[field] for field in fields if field in row}
    keys = row.keys() if hasattr(row, "keys") else ()
    return {field: row[field] for field in fields if field in keys}


def _error_feature_row(file_id: int, routing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    quality_meta: Dict[str, Any] = {"error": True}
    if routing is not None:
        quality_meta["routing"] = routing
    return {
        "file_id": file_id,
        "phash": None,
        "content_sha256": None,
        "dinov2_embedding": None,
        "quality_score": None,
        "quality_meta": json.dumps(quality_meta),
        "face_count": 0,
        "faces_json": "[]",
        "status": "done_error",
    }


# ===========================================================================
# Setup and finalization helpers
# ===========================================================================


def _admit(conn: Any, cfg: Config, scope: Any, telemetry: Any) -> Tuple[List[Any], Dict[str, int]]:
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


def _pending_work(conn: Any, thumbnailer: Any, scope: Any, telemetry: Any,
                  limit: Optional[int]) -> List[Any]:
    """Rows still needing Stage 1, in inventory order."""
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


def _build_thumbnailer(cfg: Config, telemetry: Any = None) -> Optional[thumbnails.Thumbnailer]:
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


def _build_eye_detector(cfg: Config, has_work: bool) -> Any:
    """The opt-in eye detector, or None. Never constructed with nothing to do."""
    eye_cfg = cfg.features.get("eye_detection", {})
    if not has_work or not eye_cfg.get("enabled", False):
        return None
    return eye_detection.MediaPipeEyeDetector(eye_cfg, cfg.models_dir)


def _finalize(conn: Any, cfg: Config, scope: Any, telemetry: Any, *, backend_name: str,
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


# ===========================================================================
# Main entry point
# ===========================================================================


def run(config_path: Optional[str] = None, backend_override: Optional[str] = None,
        limit: Optional[int] = None, root_override: Optional[str] = None,
        cpu_workers: Optional[int] = None) -> Dict[str, Any]:
    """Run stage 1. Returns a small stats dict."""
    outer_start = time.perf_counter()
    cfg = load_config(config_path)
    # Fail closed before opening/creating anything in the output directory: a
    # wrong --root, and equally an unbound directory with no root to adopt, must
    # not even create inventory.sqlite or upgrade its schema.
    root_scope.preflight_stage(cfg.db_path, root_override, cfg.declared_root)
    # Every loop knob is validated before any work starts, so a bad value fails
    # with a message naming the key instead of being silently clamped.
    settings = resolve_loop_settings(cfg, workers_override=cpu_workers)
    conn = db.open_db(cfg.db_path)
    telemetry = telemetry_mod.build(cfg.features)
    try:
        stats = _run_with_db(conn, cfg, telemetry, settings, backend_override,
                             limit, root_override)
    finally:
        # A Ctrl+C or a mid-loop failure must not leave the SQLite file locked by
        # an abandoned connection: the next run would fail with "database is
        # locked" instead of resuming. Rows committed before the interruption stay
        # durable; the partial batch since the last commit is rolled back, which is
        # exactly the work resume expects to recompute.
        conn.close()
    # Outer wall time closes the books: batch total + timed setup/finalize +
    # whatever is left (imports, config load, prints) must add up to it.
    telemetry.set_outer_seconds(time.perf_counter() - outer_start)
    stats["phase_telemetry"] = telemetry.snapshot()
    telemetry_mod.print_summary(stats["phase_telemetry"])
    return stats


def _run_with_db(conn: Any, cfg: Config, telemetry: Any, settings: Any,
                 backend_override: Optional[str], limit: Optional[int],
                 root_override: Optional[str]) -> Dict[str, Any]:
    """The body of :func:`run`, with the connection's lifetime owned by caller."""
    # Never run unscoped, and never bind to a root nobody named: an unbound output
    # directory adopts --root or a *declared* paths.root, else the stage refuses
    # (see root_scope.adopt_or_resolve).
    scope = root_scope.adopt_or_resolve(
        conn, root_override, db_path=str(cfg.db_path), declared_root=cfg.declared_root
    )
    skipped, skip_counts = _admit(conn, cfg, scope, telemetry)
    thumbnailer = _build_thumbnailer(cfg, telemetry)
    pending = _pending_work(conn, thumbnailer, scope, telemetry, limit)
    total = len(pending)
    # Model load is a large, one-off cost on Windows (weights + CUDA context).
    # It is timed and surfaced, not folded into the first batch or hidden.
    with telemetry.phase("setup_model_load"):
        backend = resolve_backend(cfg, backend_override) if pending else None
    if backend is not None:
        backend.telemetry = telemetry
    with telemetry.phase("setup_detectors"):
        eye_detector = _build_eye_detector(cfg, bool(pending))
        # No detector/provider/model is constructed when all inventory rows were
        # skipped or already terminal.
        scene_router = _build_scene_router(cfg) if pending else None
    thumb_note = (
        "off" if thumbnailer is None
        else f"{thumbnailer.max_px}px -> {thumbnailer.directory}"
    )
    backend_name = backend.name if backend is not None else "not-loaded"
    print(f"[stage1] backend={backend_name} pending={total} batch_size={settings.batch_size} "
          f"thumbs={thumb_note}")
    print(f"[stage1] {settings.note()}")
    print(f"[stage1] {settings.memory_note()}")
    print(f"[stage1] {root_scope.scope_note(root_scope.summary(conn, scope))}")

    done = 0
    failed = 0
    since_commit = 0
    t0 = time.time()
    batches = _guarded_batches(pending, settings.batch_size, settings.max_inflight_pixels)
    # The producer reads and prepares ahead; ``close()`` in the finally is what
    # makes Ctrl+C, an exception or an early break cancel the queue and join the
    # reader instead of leaking threads or skipping the final commit.
    source = stage1_pipeline.build_source(
        batches, backend, cfg, _open_image_and_sha,
        workers=settings.cpu_workers, prefetch_batches=settings.prefetch_batches,
        telemetry=telemetry,
    )
    try:
        for prepared in tqdm(source, desc="features", unit="batch"):
            with telemetry.batch(len(prepared)) as batch_span:
                feature_rows = _consume_batch(
                    backend, prepared, cfg, eye_detector=eye_detector,
                    scene_router=scene_router, thumbnailer=thumbnailer, telemetry=telemetry,
                )
                batch_failed = sum(1 for row in feature_rows if row["status"] == "done_error")
                batch_span.counted(len(feature_rows) - batch_failed, batch_failed)
                # db_write is a batch-unit phase: one executemany per batch, on
                # this thread. No worker ever touches the SQLite connection.
                with telemetry.phase("db_write"):
                    db.batch_insert_features(conn, feature_rows)
                    if thumbnailer is not None:
                        db.batch_upsert_thumbnails(conn, thumbnailer.drain())
                done += len(feature_rows)
                failed += batch_failed
                since_commit += len(feature_rows)
                if since_commit >= settings.commit_every:
                    with telemetry.phase("db_commit"):
                        conn.commit()
                    since_commit = 0
    finally:
        source.close()
    # Every commit is timed, including this final one.
    with telemetry.phase("db_commit_final"):
        conn.commit()
    stats = _finalize(
        conn, cfg, scope, telemetry, backend_name=backend_name, skipped=skipped,
        skip_counts=skip_counts, thumbnailer=thumbnailer, settings=settings,
        done=done, failed=failed, total=total, loop_started=t0,
    )
    conn.commit()
    return stats


def _guarded_batches(rows: Sequence[Any], batch_size: int, max_pixels: int) -> List[List[Any]]:
    """Bound full-resolution decoded images held concurrently."""
    batches: List[List[Any]] = []
    current: List[Any] = []
    pixels = 0
    for row in rows:
        item_pixels = max(1, int(row["width"] or 0) * int(row["height"] or 0))
        if current and (len(current) >= batch_size or pixels + item_pixels > max_pixels):
            batches.append(current)
            current, pixels = [], 0
        current.append(row)
        pixels += item_pixels
    if current:
        batches.append(current)
    return batches


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: feature extraction")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--backend", default=None, choices=["auto", "torch", "stub"])
    ap.add_argument("--limit", type=int, default=None, help="process at most N images")
    ap.add_argument("--root", default=None,
                    help="verify the output directory belongs to this photo root")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="threads preparing images (0 = serial, pre-change pipeline)")
    args = ap.parse_args(argv)
    run(config_path=args.config, backend_override=args.backend, limit=args.limit,
        root_override=args.root, cpu_workers=args.cpu_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
