"""Stage 1 phase telemetry: where the wall clock actually goes, cheaply.

Why this exists
---------------
A Windows run showed 1000 images in 461s of feature loop with the GPU pulsing
between 0% and 81% and only 4.8 GB VRAM in use: the GPU was starved, but the
log could not say by what. This module makes the loop self-explaining without
turning it into a profiler.

Cost and honesty rules
----------------------
* **One ``perf_counter`` pair per instrumented call.** Accumulation is a float
  add into a dict. Each phase records how many calls it measured and whether a
  call is one image, one batch, or one setup step, so a per-image measurement is
  never reported as if it were per batch.
* **Host wall time is labelled host-wall.** With an async CUDA backend a "GPU"
  phase measured on the host is really *submit + whatever the driver made us
  wait for*, i.e. an upper bound on kernel time.
* **CUDA-event timing is opt-in (default off) and never synchronises inside a
  measured phase.** When enabled, the sampling decision is latched once at batch
  entry, at most one sample is taken per model lane per sampled batch, and the
  events are only *recorded* (asynchronous, sub-microsecond) during the phase.
  All of them are resolved with a single ``synchronize`` after the batch's wall
  clock has already been taken -- so a sampled batch costs one synchronisation
  in total, outside every host-wall phase and outside the batch total.
* **Nothing is hidden.** In-loop phases reconcile against the measured batch
  total (``loop_unaccounted``), and the whole stage reconciles against its outer
  wall time (``outer_unaccounted``), which is where setup and model load live.
* **Counts are attempted/succeeded/failed.** A batch with an unreadable file
  still spent time, so the error path is instrumented and counted rather than
  quietly inflating the per-image averages.
* **A phase's ``calls`` counts attempts**, including one that raised: the time
  was really spent. Outcomes are reported separately as
  ``images_succeeded``/``images_failed``, so a failing library is visible in both
  dimensions instead of being averaged away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOOP = "loop"
SETUP = "setup"

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
    scope: str                # LOOP | SETUP
    unit: str                 # UNIT_IMAGE | UNIT_BATCH | UNIT_STEP


PHASES: Tuple[Phase, ...] = (
    # --- one-off, outside the per-batch loop ------------------------------
    Phase("setup_admission", "setup: oversize admission", "host", SETUP, UNIT_STEP),
    Phase("setup_pending_query", "setup: pending-work query", "host", SETUP, UNIT_STEP),
    Phase("setup_thumb_listing", "setup: thumbnail cache listing", "host", SETUP, UNIT_STEP),
    Phase("setup_model_load", "setup: model load (backend)", "host", SETUP, UNIT_STEP),
    Phase("setup_detectors", "setup: optional detectors", "host", SETUP, UNIT_STEP),
    # --- per batch / per image -------------------------------------------
    Phase("source_open_read", "source open + read (HDD)", "host", LOOP, UNIT_IMAGE),
    Phase("decode", "JPEG decode", "host", LOOP, UNIT_IMAGE),
    Phase("hash_sha256", "SHA-256 of source bytes", "host", LOOP, UNIT_IMAGE),
    Phase("phash", "perceptual hash", "host", LOOP, UNIT_IMAGE),
    Phase("embed_preprocess", "embedding preprocess", "host", LOOP, UNIT_BATCH),
    Phase("embed_inference", "embedding inference", "gpu_event", LOOP, UNIT_BATCH),
    Phase("quality_preprocess", "IQA preprocess (bounded resize)", "host", LOOP, UNIT_IMAGE),
    Phase("quality_musiq", "IQA MUSIQ", "gpu_event", LOOP, UNIT_IMAGE),
    Phase("quality_clipiqa", "IQA CLIP-IQA", "gpu_event", LOOP, UNIT_IMAGE),
    Phase("quality_sharpness", "sharpness (CPU)", "host", LOOP, UNIT_IMAGE),
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

# CUDA-event sampling is opt-in: the default run must have zero observer effect.
DEFAULT_GPU_EVENT_EVERY = 0


class _Span:
    """Times one instrumented call."""

    __slots__ = ("_sink", "_key", "_count", "_start")

    def __init__(self, sink: "Telemetry", key: str, count: int) -> None:
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


class _NullSpan:
    """Zero-cost stand-in used when telemetry (or a sample) is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


NULL_SPAN = _NullSpan()


class Telemetry:
    """Cumulative per-phase seconds/counts for one Stage 1 run."""

    def __init__(self, enabled: bool = True, gpu_event_every: int = DEFAULT_GPU_EVENT_EVERY,
                 torch_module: Any = None) -> None:
        self.enabled = bool(enabled)
        self.gpu_event_every = max(0, int(gpu_event_every))
        self.seconds: Dict[str, float] = {key: 0.0 for key in PHASE_KEYS}
        self.counts: Dict[str, int] = {key: 0 for key in PHASE_KEYS}
        self.gpu_event_seconds: Dict[str, float] = {}
        self.gpu_event_samples: Dict[str, int] = {}
        self.batches = 0
        self.images_attempted = 0
        self.images_succeeded = 0
        self.images_failed = 0
        self.batch_seconds: List[float] = []
        self.loop_seconds = 0.0
        self.outer_seconds: Optional[float] = None
        self.sampled_batches = 0
        self.gpu_syncs = 0
        self.notes: List[str] = []
        self._torch = torch_module
        self._cuda_ready: Optional[bool] = None
        self._sampling = False          # latched once per batch, never mid-batch
        self._sampled_keys: set[str] = set()
        self._pending: List[Tuple[str, Any, Any]] = []

    # -- recording ----------------------------------------------------------
    def phase(self, key: str, count: int = 1) -> Any:
        """Time one call of ``key``. Unknown keys fold into ``other_cpu``."""
        if not self.enabled:
            return NULL_SPAN
        return _Span(self, key if key in self.seconds else "other_cpu", count)

    def add(self, key: str, seconds: float, count: int = 1) -> None:
        if not self.enabled:
            return
        if key not in self.seconds:
            key = "other_cpu"
        self.seconds[key] += float(seconds)
        self.counts[key] += int(count)

    def note(self, text: str) -> None:
        if self.enabled and text and text not in self.notes:
            self.notes.append(text)

    def set_outer_seconds(self, seconds: float) -> None:
        """Total Stage 1 wall time, so setup/loop can be reconciled against it."""
        self.outer_seconds = float(seconds)

    def batch(self, attempted: int) -> "_BatchSpan":
        """Time one batch and latch the CUDA-event sampling decision."""
        return _BatchSpan(self, attempted)

    def record_batch(self, seconds: float, attempted: int,
                     succeeded: Optional[int] = None, failed: int = 0) -> None:
        if not self.enabled:
            return
        attempted = int(attempted)
        failed = int(failed)
        succeeded = attempted - failed if succeeded is None else int(succeeded)
        self.batches += 1
        self.images_attempted += attempted
        self.images_succeeded += succeeded
        self.images_failed += failed
        self.loop_seconds += float(seconds)
        self.batch_seconds.append(float(seconds))

    # -- optional CUDA-event sampling --------------------------------------
    def _begin_batch(self) -> None:
        """Latch sampling for this batch. Called before any phase of the batch."""
        self._sampled_keys.clear()
        self._pending.clear()
        self._sampling = (
            self.enabled and self.gpu_event_every > 0
            and self.batches % self.gpu_event_every == 0
            and self._cuda_available()
        )
        if self._sampling:
            self.sampled_batches += 1

    def sampling_active(self) -> bool:
        """Whether this batch is being sampled (latched at batch entry)."""
        return self._sampling

    def _cuda_available(self) -> bool:
        if self._cuda_ready is None:
            torch = self._torch
            try:
                if torch is None:
                    import torch as torch_module  # noqa: WPS433 (optional dependency)

                    torch = torch_module
                    self._torch = torch
                self._cuda_ready = bool(torch.cuda.is_available())
            except Exception:
                self._cuda_ready = False
            if not self._cuda_ready:
                self.note("CUDA event timing unavailable (no CUDA); every number is host-wall")
        return bool(self._cuda_ready)

    def gpu_event_phase(self, key: str) -> Any:
        """Record CUDA events around one call; resolved once at batch end.

        Returns a no-op span unless this batch is sampled *and* this lane has not
        been sampled yet in this batch, so a batch of N images still yields at
        most one sample per lane -- never N synchronisations.
        """
        if not self._sampling or key in self._sampled_keys:
            return NULL_SPAN
        self._sampled_keys.add(key)
        return _GpuEventSpan(self, key)

    def _queue_events(self, key: str, start: Any, end: Any) -> None:
        self._pending.append((key, start, end))

    def _resolve_gpu_events(self) -> None:
        """One synchronise per sampled batch, after the batch clock was taken."""
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        try:
            pending[-1][2].synchronize()
            self.gpu_syncs += 1
            for key, start, end in pending:
                seconds = start.elapsed_time(end) / 1000.0
                self.gpu_event_seconds[key] = (
                    self.gpu_event_seconds.get(key, 0.0) + seconds
                )
                self.gpu_event_samples[key] = self.gpu_event_samples.get(key, 0) + 1
        except Exception:
            self.note("CUDA event timing failed; host-wall numbers only")

    def add_gpu_event(self, key: str, seconds: float, samples: int = 1) -> None:
        """Record an already-resolved GPU-event measurement (used by tests)."""
        self.gpu_event_seconds[key] = self.gpu_event_seconds.get(key, 0.0) + float(seconds)
        self.gpu_event_samples[key] = self.gpu_event_samples.get(key, 0) + int(samples)

    # -- derived ------------------------------------------------------------
    def scoped_seconds(self, scope: str) -> float:
        return sum(value for key, value in self.seconds.items()
                   if BY_KEY[key].scope == scope)

    def loop_unaccounted(self) -> float:
        return max(0.0, self.loop_seconds - self.scoped_seconds(LOOP))

    def outer_unaccounted(self) -> Optional[float]:
        """Stage-1 wall time neither inside a batch nor in a timed setup step."""
        if self.outer_seconds is None:
            return None
        return max(0.0, self.outer_seconds - self.loop_seconds - self.scoped_seconds(SETUP))

    def percentiles(self) -> Dict[str, Optional[float]]:
        return {"p50": _percentile(self.batch_seconds, 0.50),
                "p95": _percentile(self.batch_seconds, 0.95)}

    def snapshot(self) -> Dict[str, Any]:
        """Plain dict for reports/JSON. Only phases with time or calls appear."""
        images = self.images_attempted
        phases = [self._phase_payload(BY_KEY[key], images) for key in PHASE_KEYS
                  if self.seconds[key] > 0.0 or self.counts[key] > 0]
        outer_unaccounted = self.outer_unaccounted()
        return {
            "enabled": self.enabled,
            "images_attempted": images,
            "images_succeeded": self.images_succeeded,
            "images_failed": self.images_failed,
            "batches": self.batches,
            "loop_seconds": round(self.loop_seconds, 4),
            "outer_seconds": (round(self.outer_seconds, 4)
                              if self.outer_seconds is not None else None),
            "loop_accounted_seconds": round(self.scoped_seconds(LOOP), 4),
            "setup_accounted_seconds": round(self.scoped_seconds(SETUP), 4),
            "loop_unaccounted_seconds": round(self.loop_unaccounted(), 4),
            "loop_unaccounted_percent": ((100.0 * self.loop_unaccounted() / self.loop_seconds)
                                         if self.loop_seconds else None),
            "outer_unaccounted_seconds": (round(outer_unaccounted, 4)
                                          if outer_unaccounted is not None else None),
            "batch_percentiles": self.percentiles(),
            "gpu_event_sampling_every": self.gpu_event_every,
            "gpu_event_sampled_batches": self.sampled_batches,
            "gpu_event_synchronisations": self.gpu_syncs,
            "phases": phases,
            "notes": list(self.notes),
        }

    def _phase_payload(self, phase: Phase, images: int) -> Dict[str, Any]:
        seconds = self.seconds[phase.key]
        denominator = self.loop_seconds if phase.scope == LOOP else self.outer_seconds
        return {
            "key": phase.key,
            "label": phase.label,
            "kind": phase.kind,
            "scope": phase.scope,
            "unit": phase.unit,
            "seconds": round(seconds, 4),
            "calls": self.counts[phase.key],
            "ms_per_image": (1000.0 * seconds / images) if images else None,
            "percent": (100.0 * seconds / denominator) if denominator else None,
            "gpu_event_seconds": (round(self.gpu_event_seconds[phase.key], 4)
                                  if phase.key in self.gpu_event_seconds else None),
            "gpu_event_samples": self.gpu_event_samples.get(phase.key),
        }


class _BatchSpan:
    """Times one batch; latches sampling on entry, resolves events on exit."""

    __slots__ = ("_sink", "_attempted", "_succeeded", "_failed", "_start")

    def __init__(self, sink: Telemetry, attempted: int) -> None:
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
        # Batch wall time is taken *before* resolving CUDA events, so the single
        # synchronisation of a sampled batch cannot inflate any reported number.
        self._sink.record_batch(elapsed, self._attempted, self._succeeded, self._failed)
        self._sink._resolve_gpu_events()
        return False


class _GpuEventSpan:
    """Records (does not synchronise) CUDA events around one sampled call."""

    __slots__ = ("_sink", "_key", "_start", "_end", "_ok")

    def __init__(self, sink: Telemetry, key: str) -> None:
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
            self._start.record()        # asynchronous; no host/device sync here
            self._ok = True
        except Exception:
            self._ok = False
            self._sink.note("CUDA event creation failed; host-wall numbers only")
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if not self._ok:
            return False
        try:
            self._end.record()          # still asynchronous
            self._sink._queue_events(self._key, self._start, self._end)
        except Exception:
            self._sink.note("CUDA event timing failed; host-wall numbers only")
        return False


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 4)


def build(cfg_features: Any, torch_module: Any = None) -> Telemetry:
    """Construct telemetry from the ``features.telemetry`` config section."""
    getter: Callable[[str, Any], Any]
    if hasattr(cfg_features, "get"):
        getter = cfg_features.get
    else:  # pragma: no cover - defensive
        def getter(_name: str, default: Any = None) -> Any:
            return default
    section = getter("telemetry", {}) or {}
    if not isinstance(section, dict):
        section = {}
    return Telemetry(
        enabled=bool(section.get("enabled", True)),
        gpu_event_every=int(section.get("gpu_event_every", DEFAULT_GPU_EVENT_EVERY)),
        torch_module=torch_module,
    )


# Rendering lives in :mod:`src.telemetry_report`, which imports this module for
# the phase vocabulary. These thin wrappers keep ``telemetry.render_lines(...)``
# working for existing callers without an import cycle at module load.
def render_lines(snapshot: Dict[str, Any]) -> List[str]:
    from . import telemetry_report

    return telemetry_report.render_lines(snapshot)


def print_summary(snapshot: Dict[str, Any], prefix: str = "[stage1]") -> None:
    from . import telemetry_report

    telemetry_report.print_summary(snapshot, prefix)
