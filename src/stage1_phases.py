"""Stage 1 phase vocabulary: what can be measured, and in which units.

Split out of :mod:`src.stage1_telemetry` so the collector file stays about
measuring and this one is the single place that declares *what* a phase is. Both
:mod:`src.stage1_telemetry` and :mod:`src.telemetry_report` import from here, and
the collector re-exports every name so existing callers keep working.

The three attributes are the honesty contract:

``scope``
    ``SETUP``  -- one-off, outside the per-batch loop.
    ``LOOP``   -- measured on the main thread inside a batch, so it is part of
                  the batch total and reconciles against it.
    ``WAIT``   -- measured on the main thread *between* batches. The producer's
                  next batch is claimed before the batch span opens, so this time
                  is genuinely main-thread time but is **not** inside any batch
                  total. It reconciles against Stage 1's outer wall time instead,
                  and it is the honest answer to "what did prefetch actually cost
                  the main thread".
    ``WORKER`` -- measured on a producer thread (reader or CPU worker). It runs
                  *concurrently* with the loop, so it must never be added to the
                  loop's accounting; it is reported in its own section.

``unit``
    Whether one instrumented call covered one image, one batch, or one step.
    A phase whose work was batched must say ``batch``, so a per-model-call
    measurement can never be misread as a per-image one.

``kind``
    ``host`` -- host wall clock. ``gpu_event`` -- a lane that can additionally be
    sampled with CUDA events.

A phase key may legitimately be recorded in ``LOOP`` *or* ``WORKER`` depending on
the run: with ``features.cpu_workers: 0`` the decode happens on the main thread
inside the batch, and with workers it happens off it. ``scope`` records where the
phase lives by default; the collector reports where it was actually measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

LOOP = "loop"
SETUP = "setup"
WORKER = "worker"
WAIT = "wait"

# A "call" is what one instrumented span covers: one image, one batch, or one
# one-off setup step. The report prints this so counts cannot be misread.
UNIT_IMAGE = "img"
UNIT_BATCH = "batch"
UNIT_STEP = "step"


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    kind: str                 # 'host' | 'gpu_event' (gpu_event = samplable lane)
    scope: str                # LOOP | SETUP | WORKER
    unit: str                 # UNIT_IMAGE | UNIT_BATCH | UNIT_STEP


PHASES: Tuple[Phase, ...] = (
    # --- one-off, outside the per-batch loop ------------------------------
    Phase("setup_admission", "setup: oversize admission", "host", SETUP, UNIT_STEP),
    Phase("setup_pending_query", "setup: pending-work query", "host", SETUP, UNIT_STEP),
    Phase("setup_thumb_listing", "setup: thumbnail cache listing", "host", SETUP, UNIT_STEP),
    Phase("setup_model_load", "setup: model load (backend)", "host", SETUP, UNIT_STEP),
    Phase("setup_detectors", "setup: optional detectors", "host", SETUP, UNIT_STEP),
    # --- producer work: sequential source read + parallel CPU preparation --
    # These run on the main thread when cpu_workers=0 and on producer threads
    # otherwise. The collector records which, so neither reading is invented.
    Phase("source_open_read", "source open + read (HDD)", "host", WORKER, UNIT_IMAGE),
    Phase("decode", "JPEG decode", "host", WORKER, UNIT_IMAGE),
    Phase("hash_sha256", "SHA-256 of source bytes", "host", WORKER, UNIT_IMAGE),
    Phase("phash", "perceptual hash", "host", WORKER, UNIT_IMAGE),
    Phase("embed_preprocess", "embedding preprocess", "host", WORKER, UNIT_IMAGE),
    Phase("quality_preprocess", "IQA preprocess (bounded resize)", "host", WORKER, UNIT_IMAGE),
    Phase("quality_sharpness", "sharpness (CPU)", "host", WORKER, UNIT_IMAGE),
    Phase("exposure_preprocess", "exposure resize (CPU)", "host", WORKER, UNIT_IMAGE),
    Phase("faces_preprocess", "face detect input prep (CPU)", "host", WORKER, UNIT_IMAGE),
    # --- per batch / per image, on the main thread -------------------------
    # Claimed before the batch span opens, hence WAIT rather than LOOP.
    Phase("prefetch_wait", "waiting for prepared batch", "host", WAIT, UNIT_BATCH),
    Phase("embed_inference", "embedding inference", "gpu_event", LOOP, UNIT_BATCH),
    Phase("iqa_stack_upload", "IQA stack + upload", "host", LOOP, UNIT_BATCH),
    Phase("quality_musiq", "IQA MUSIQ", "gpu_event", LOOP, UNIT_BATCH),
    Phase("quality_clipiqa", "IQA CLIP-IQA", "gpu_event", LOOP, UNIT_BATCH),
    Phase("faces_yunet", "face detect YuNet", "host", LOOP, UNIT_IMAGE),
    Phase("face_quality", "face quality (CPU)", "host", LOOP, UNIT_IMAGE),
    Phase("exposure", "exposure metrics (CPU)", "host", LOOP, UNIT_IMAGE),
    Phase("eye_detection", "eye detection", "host", LOOP, UNIT_IMAGE),
    Phase("scene_routing", "scene routing (shadow)", "host", LOOP, UNIT_IMAGE),
    Phase("thumbnail_resize", "thumbnail resize", "host", LOOP, UNIT_IMAGE),
    Phase("thumbnail_encode_write", "thumbnail encode + write (SSD)", "host", LOOP, UNIT_IMAGE),
    Phase("error_handling", "unreadable-source handling", "host", LOOP, UNIT_IMAGE),
    Phase("db_write", "DB write (executemany)", "host", LOOP, UNIT_BATCH),
    Phase("db_commit", "DB commit (in loop)", "host", LOOP, UNIT_STEP),
    Phase("other_cpu", "other CPU work", "host", LOOP, UNIT_IMAGE),
    # --- after the loop ---------------------------------------------------
    Phase("db_commit_final", "DB commit (final)", "host", SETUP, UNIT_STEP),
    Phase("finalize_stats", "finalize: meta + stats queries", "host", SETUP, UNIT_STEP),
)
BY_KEY: Dict[str, Phase] = {phase.key: phase for phase in PHASES}
PHASE_KEYS: Tuple[str, ...] = tuple(phase.key for phase in PHASES)

# Counters are plain integers that describe *what* the loop did rather than how
# long it took: how many model calls covered how many images. They are what
# proves the IQA lane is batched instead of per image.
COUNTER_LABELS: Dict[str, str] = {
    "iqa_musiq_calls": "MUSIQ model calls",
    "iqa_musiq_images": "images scored by MUSIQ",
    "iqa_clipiqa_calls": "CLIP-IQA model calls",
    "iqa_clipiqa_images": "images scored by CLIP-IQA",
    "iqa_shape_groups": "IQA shape groups (distinct bounded sizes)",
    "embed_calls": "embedding model calls",
    "embed_images": "images embedded",
    "prefetch_batches_delivered": "prefetched batches delivered",
    "prefetch_images_prepared": "images prepared by producer threads",
}
