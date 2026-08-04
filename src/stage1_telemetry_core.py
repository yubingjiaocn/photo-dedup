"""Stage 1 telemetry core: phase vocabulary and span measurement objects.

Phase vocabulary (PHASES, BY_KEY) defines what can be measured and in which units.
Span objects (_Span, _WorkerSpan, etc.) are the actual measuring devices used by
:mod:`src.stage1_telemetry`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Phase vocabulary: what can be measured -------------------------------------

LOOP = "loop"
SETUP = "setup"
WORKER = "worker"
WAIT = "wait"

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

# --- Span objects: the actual measuring devices ---------------------------------


class _Span:
    """Times one instrumented call on the main thread."""

    __slots__ = ("_sink", "_key", "_count", "_start")

    def __init__(self, sink: Any, key: str, count: int) -> None:
        self._sink = sink
        self._key = key
        self._count = count
        self._start = 0.0

    def __enter__(self) -> "_Span":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self._sink.add(self._key, time.perf_counter() - self._start, self._count)
        return False


class _WorkerSpan:
    """Times one call made on a producer thread (accumulated under a lock)."""

    __slots__ = ("_sink", "_key", "_count", "_start")

    def __init__(self, sink: Any, key: str, count: int) -> None:
        self._sink = sink
        self._key = key
        self._count = count
        self._start = 0.0

    def __enter__(self) -> "_WorkerSpan":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self._sink.add_worker(self._key, time.perf_counter() - self._start, self._count)
        return False


class _NullSpan:
    """Zero-cost stand-in used when telemetry (or a sample) is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


NULL_SPAN = _NullSpan()


class _BatchSpan:
    """Times one batch; latches sampling on entry, resolves events on exit."""

    __slots__ = ("_sink", "_attempted", "_succeeded", "_failed", "_start")

    def __init__(self, sink: Any, attempted: int) -> None:
        self._sink = sink
        self._attempted = int(attempted)
        self._succeeded: Optional[int] = None
        self._failed = 0
        self._start = 0.0

    def counted(self, succeeded: int, failed: int) -> None:
        """Report what the batch actually achieved (attempted stays as given)."""
        self._succeeded = int(succeeded)
        self._failed = int(failed)

    def __enter__(self) -> "_BatchSpan":
        self._sink._begin_batch()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        elapsed = time.perf_counter() - self._start
        self._sink.record_batch(elapsed, self._attempted, self._succeeded, self._failed)
        self._sink._resolve_gpu_events()
        return False


class _GpuEventSpan:
    """Records (does not synchronise) CUDA events around one sampled call."""

    __slots__ = ("_sink", "_key", "_start", "_end", "_ok")

    def __init__(self, sink: Any, key: str) -> None:
        self._sink = sink
        self._key = key
        self._ok = False
        self._start: Any = None
        self._end: Any = None

    def __enter__(self) -> "_GpuEventSpan":
        torch = self._sink._torch
        try:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
            self._ok = True
        except Exception:
            self._ok = False
            self._sink.note("CUDA event creation failed; host-wall numbers only")
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if not self._ok:
            return False
        try:
            self._end.record()
            self._sink._queue_events(self._key, self._start, self._end)
        except Exception:
            self._sink.note("CUDA event timing failed; host-wall numbers only")
        return False


PendingEvents = List[Tuple[str, Any, Any]]
