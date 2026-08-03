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
from . import db
from . import quality as Q
from . import exposure
from . import decision
from . import eye_detection
from . import thumbnails
from .scene_router import DecodeState, RoutingInput, SceneRouter
from .stage1_backends import (  # noqa: F401 (re-exported for existing callers/tests)
    EMBED_DIM,
    StubBackend,
    TorchBackend,
    _load_yunet,
    resolve_backend,
)

try:  # progress bar is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x


# ===========================================================================
# Main loop
# ===========================================================================

def _open_image_and_sha(path: Path) -> Tuple[Image.Image, str]:
    """Read compressed bytes once, hashing that same sequential read before decode."""
    digest = hashlib.sha256()
    data = io.BytesIO()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
            data.write(chunk)
    data.seek(0)
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
) -> List[Dict[str, Any]]:
    """Compute features for one batch of file rows. Returns feature dicts.

    The router is constructed at batch scope when the caller did not retain a
    run-scoped instance.  It receives the same decoded RGB ``Image`` used by
    embeddings and technical detectors; it never opens the source path.

    The optional ``thumbnailer`` gets that *same* decoded object too, so the
    SSD thumbnail cache costs no additional HDD read and no second decode.
    """
    router = scene_router if scene_router is not None else _build_scene_router(cfg)
    images: List[Image.Image] = []
    hashes: List[str] = []
    valid_rows: List[Any] = []
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        try:
            image, sha256 = _open_image_and_sha(Path(row["path"]))
            images.append(image)
            hashes.append(sha256)
            valid_rows.append(row)
        except Exception as exc:  # unreadable / corrupt file
            print(f"[stage1][WARN] cannot read {row['path']}: {exc}")
            routing = router.route(RoutingInput(
                metadata=_routing_metadata(row), decode_state=DecodeState.FAILED
            ))
            if thumbnailer is not None:
                thumbnailer.record_source_failure(
                    row, f"{thumbnails.SOURCE_DECODE_FAILED}: {type(exc).__name__}: {exc}"
                )
            out_rows.append(_error_feature_row(int(row["id"]), routing=routing))

    if not valid_rows:
        return out_rows

    embeddings = backend.embed_batch(images)
    for row, image, sha256, emb in zip(valid_rows, images, hashes, embeddings):
        score, meta = backend.quality(image)
        faces = backend.faces(image)
        img_rgb = np.asarray(image)
        fq = Q.compute_face_quality(img_rgb, faces)
        meta["face_quality"] = fq
        meta["schema_version"] = 2
        meta["exposure"] = exposure.extract(
            image, faces, int(cfg.features.get("exposure_long_edge", 512))
        )
        meta["detectors"] = decision.detector_extensions()
        if eye_detector is not None:
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
        meta["routing"] = router.route(RoutingInput(
            metadata=_routing_metadata(row), image=image
        ))
        # Same decoded object, no reopen: the SSD thumbnail is a by-product of
        # the decode the extractors above already paid for.
        if thumbnailer is not None:
            thumbnailer.capture(row, image)
        out_rows.append(
            {
                "file_id": int(row["id"]),
                "phash": Q.phash_bytes(image),
                "content_sha256": sha256,
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


def _build_thumbnailer(cfg: Config) -> Optional[thumbnails.Thumbnailer]:
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
    )


def run(config_path: Optional[str] = None, backend_override: Optional[str] = None,
        limit: Optional[int] = None) -> Dict[str, Any]:
    """Run stage 1. Returns a small stats dict."""
    cfg = load_config(config_path)
    conn = db.open_db(cfg.db_path)
    backend = resolve_backend(cfg, backend_override)
    eye_cfg = cfg.features.get("eye_detection", {})
    eye_detector = (
        eye_detection.MediaPipeEyeDetector(eye_cfg, cfg.models_dir)
        if eye_cfg.get("enabled", False)
        else None
    )
    # Construct once per Stage 1 run. Local runtime failures remain a
    # MODEL_UNAVAILABLE shadow record and never interrupt existing features.
    scene_router = _build_scene_router(cfg)
    thumbnailer = _build_thumbnailer(cfg)
    batch_size = max(1, int(cfg.features.get("batch_size", 4)))
    max_pixels = int(float(cfg.features.get("max_inflight_megapixels", 80)) * 1_000_000)
    commit_every = int(cfg.scan.get("commit_every", 100))

    if thumbnailer is None:
        pending = list(db.iter_files_for_features(conn))
    else:
        # One cheap SSD listing decides which cached JPEGs actually exist, so a
        # deleted thumbnail is regenerated and a stale one is never reused.
        db.register_thumb_presence(conn, thumbnails.ids_on_disk(thumbnailer.directory))
        pending = list(db.iter_files_for_features(conn, thumb_max_px=thumbnailer.max_px))
    if limit:
        pending = pending[:limit]
    total = len(pending)
    thumb_note = (
        "off" if thumbnailer is None
        else f"{thumbnailer.max_px}px -> {thumbnailer.directory}"
    )
    print(f"[stage1] backend={backend.name} pending={total} batch_size={batch_size} "
          f"thumbs={thumb_note}")

    done = 0
    since_commit = 0
    t0 = time.time()
    batches = _guarded_batches(pending, batch_size, max_pixels)
    for batch in tqdm(batches, desc="features", unit="batch"):
        feature_rows = _process_batch(
            backend, batch, cfg, eye_detector=eye_detector, scene_router=scene_router,
            thumbnailer=thumbnailer,
        )
        db.batch_insert_features(conn, feature_rows)
        if thumbnailer is not None:
            db.batch_upsert_thumbnails(conn, thumbnailer.drain())
        done += len(feature_rows)
        since_commit += len(feature_rows)
        if since_commit >= commit_every:
            conn.commit()
            since_commit = 0
    conn.commit()

    db.set_meta(conn, "stage1_backend", backend.name)
    db.set_meta(conn, "stage1_done_at", str(int(time.time())))
    dt = time.time() - t0
    rate = done / dt if dt > 0 else 0.0
    print(f"[stage1] processed {done} images in {dt:.1f}s ({rate:.1f}/s)")
    stats: Dict[str, Any] = {"processed": done, "total": total}
    if thumbnailer is None:
        stats["thumbnails"] = thumbnails.disabled_stats()
    else:
        thumb_stats = thumbnailer.stats()
        thumb_stats.update(db.thumbnail_stats(conn))
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
    conn.commit()
    conn.close()
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
    args = ap.parse_args(argv)
    run(config_path=args.config, backend_override=args.backend, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
