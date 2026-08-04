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
* **Concurrent producer work is never added to the loop's books.** With
  ``features.cpu_workers > 0`` the source read, decode and per-image CPU
  preparation happen on producer threads *while* the main thread runs the GPU
  and detector lanes. That time is real but it is not main-thread time, so it
  accumulates in a separate ``WORKER`` scope (under a lock) and is reported in
  its own section. What the main thread actually paid for it is the
  ``prefetch_wait`` phase, which *is* in the loop's books. ``overlap_seconds``
  is then simply producer CPU seconds minus that wait: work that was genuinely
  hidden behind the GPU rather than a speedup that was assumed.
* **Counts are attempted/succeeded/failed.** A batch with an unreadable file
  still spent time, so the error path is instrumented and counted rather than
  quietly inflating the per-image averages.
* **A phase's ``calls`` counts attempts**, including one that raised: the time
  was really spent. Outcomes are reported separately as
  ``images_succeeded``/``images_failed``, so a failing library is visible in both
  dimensions instead of being averaged away.
* **Counters state batching in the open.** ``iqa_musiq_calls`` next to
  ``iqa_musiq_images`` is what makes "one model call covered four images"
  checkable instead of a claim in a commit message.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# The phase vocabulary lives in its own module; re-exported so every existing
# caller of ``stage1_telemetry.PHASES`` / ``.LOOP`` keeps working unchanged.
from .stage1_phases import (  # noqa: F401 (intentional re-export)
    BY_KEY,
    COUNTER_LABELS,
    LOOP,
    PHASE_KEYS,
    PHASES,
    SETUP,
    UNIT_BATCH,
    UNIT_IMAGE,
    UNIT_STEP,
    WAIT,
    WORKER,
    Phase,
)
# The measuring devices themselves live in :mod:`src.stage1_spans`; re-exported
# so ``stage1_telemetry.NULL_SPAN`` keeps working for every existing caller.
from .stage1_spans import (  # noqa: F401 (intentional re-export)
    NULL_SPAN,
    _BatchSpan,
    _GpuEventSpan,
    _NullSpan,
    _Span,
    _WorkerSpan,
)

# CUDA-event sampling is opt-in: the default run must have zero observer effect.
DEFAULT_GPU_EVENT_EVERY = 0


class Telemetry:
    """Cumulative per-phase seconds/counts for one Stage 1 run."""

    def __init__(self, enabled: bool = True, gpu_event_every: int = DEFAULT_GPU_EVENT_EVERY,
                 torch_module: Any = None) -> None:
        self.enabled = bool(enabled)
        self.gpu_event_every = max(0, int(gpu_event_every))
        self.seconds: Dict[str, float] = {key: 0.0 for key in PHASE_KEYS}
        self.counts: Dict[str, int] = {key: 0 for key in PHASE_KEYS}
        # Producer-thread accounting is kept apart from main-thread accounting:
        # the two run concurrently, so adding them would invent time.
        self.worker_seconds: Dict[str, float] = {}
        self.worker_counts: Dict[str, int] = {}
        self.counters: Dict[str, int] = {}
        self._lock = threading.Lock()
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

    def worker_phase(self, key: str, count: int = 1) -> Any:
        """Time one call made off the main thread (producer read/decode/prep)."""
        if not self.enabled:
            return NULL_SPAN
        return _WorkerSpan(self, key if key in BY_KEY else "other_cpu", count)

    def add_worker(self, key: str, seconds: float, count: int = 1) -> None:
        """Accumulate producer-thread seconds. Safe from any thread."""
        if not self.enabled:
            return
        if key not in BY_KEY:
            key = "other_cpu"
        with self._lock:
            self.worker_seconds[key] = self.worker_seconds.get(key, 0.0) + float(seconds)
            self.worker_counts[key] = self.worker_counts.get(key, 0) + int(count)

    def count(self, key: str, value: int = 1) -> None:
        """Increment a plain counter (model calls, images per call, ...)."""
        if not self.enabled:
            return
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + int(value)

    def note(self, text: str) -> None:
        if not self.enabled or not text:
            return
        with self._lock:
            if text not in self.notes:
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
        if scope == WORKER:
            return sum(self.worker_seconds.values())
        return sum(value for key, value in self.seconds.items()
                   if BY_KEY[key].scope == scope)

    def loop_unaccounted(self) -> float:
        """Batch wall time not covered by a main-thread phase.

        Only main-thread measurements count here. A phase key declared ``WORKER``
        but measured on the main thread (``cpu_workers: 0``) is in ``self.seconds``
        and therefore *is* accounted; the same key measured on a producer thread
        lives in ``self.worker_seconds`` and is not, because it did not consume
        main-thread time.

        ``WAIT`` phases are excluded as well: the producer's next batch is claimed
        *before* the batch span opens, so that time is outside every batch total
        and is reconciled against the outer wall time instead.
        """
        return max(0.0, self.loop_seconds - self.loop_accounted_seconds())

    def loop_accounted_seconds(self) -> float:
        return sum(value for key, value in self.seconds.items()
                   if BY_KEY[key].scope in (LOOP, WORKER))

    def overlap_seconds(self) -> Optional[float]:
        """Producer CPU seconds the main thread did *not* wait for.

        Producer work minus the measured ``prefetch_wait``. Positive means that
        much read/decode/prepare really did happen behind the main thread's GPU
        and detector work. It is a measurement, not a modelled speedup, and it is
        ``None`` when no producer thread ran.
        """
        produced = self.scoped_seconds(WORKER)
        if produced <= 0.0 and not self.worker_counts:
            return None
        return produced - self.seconds.get("prefetch_wait", 0.0)

    def outer_unaccounted(self) -> Optional[float]:
        """Stage-1 wall time in no batch, no timed setup step and no wait."""
        if self.outer_seconds is None:
            return None
        return max(0.0, self.outer_seconds - self.loop_seconds
                   - self.scoped_seconds(SETUP) - self.scoped_seconds(WAIT))

    def percentiles(self) -> Dict[str, Optional[float]]:
        return {"p50": _percentile(self.batch_seconds, 0.50),
                "p95": _percentile(self.batch_seconds, 0.95)}

    def snapshot(self) -> Dict[str, Any]:
        """Plain dict for reports/JSON. Only phases with time or calls appear."""
        images = self.images_attempted
        phases = [self._phase_payload(BY_KEY[key], images) for key in PHASE_KEYS
                  if self.seconds[key] > 0.0 or self.counts[key] > 0]
        worker_phases = [self._worker_payload(BY_KEY[key], images)
                         for key in PHASE_KEYS if self.worker_seconds.get(key)
                         or self.worker_counts.get(key)]
        outer_unaccounted = self.outer_unaccounted()
        overlap = self.overlap_seconds()
        return {
            "enabled": self.enabled,
            "images_attempted": images,
            "images_succeeded": self.images_succeeded,
            "images_failed": self.images_failed,
            "batches": self.batches,
            "loop_seconds": round(self.loop_seconds, 4),
            "outer_seconds": (round(self.outer_seconds, 4)
                              if self.outer_seconds is not None else None),
            "loop_accounted_seconds": round(self.loop_accounted_seconds(), 4),
            "setup_accounted_seconds": round(self.scoped_seconds(SETUP), 4),
            "wait_accounted_seconds": round(self.scoped_seconds(WAIT), 4),
            "loop_unaccounted_seconds": round(self.loop_unaccounted(), 4),
            "loop_unaccounted_percent": ((100.0 * self.loop_unaccounted() / self.loop_seconds)
                                         if self.loop_seconds else None),
            "outer_unaccounted_seconds": (round(outer_unaccounted, 4)
                                          if outer_unaccounted is not None else None),
            "worker_seconds": round(self.scoped_seconds(WORKER), 4),
            "worker_phases": worker_phases,
            "prefetch_wait_seconds": round(self.seconds.get("prefetch_wait", 0.0), 4),
            "overlap_seconds": (round(overlap, 4) if overlap is not None else None),
            "counters": dict(self.counters),
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
        if phase.scope == WORKER:
            # Measured on the main thread despite being a producer-scope phase
            # (cpu_workers=0), so the batch total is the right denominator.
            denominator = self.loop_seconds
        return {
            "key": phase.key,
            "label": phase.label,
            "kind": phase.kind,
            "scope": LOOP if phase.scope == WORKER else phase.scope,
            "unit": phase.unit,
            "seconds": round(seconds, 4),
            "calls": self.counts[phase.key],
            "ms_per_image": (1000.0 * seconds / images) if images else None,
            "percent": (100.0 * seconds / denominator) if denominator else None,
            "gpu_event_seconds": (round(self.gpu_event_seconds[phase.key], 4)
                                  if phase.key in self.gpu_event_seconds else None),
            "gpu_event_samples": self.gpu_event_samples.get(phase.key),
        }

    def _worker_payload(self, phase: Phase, images: int) -> Dict[str, Any]:
        """One producer-thread phase. Percent is of total producer CPU seconds.

        Deliberately *not* a percentage of the loop: producer seconds are
        concurrent with the loop and can legitimately exceed it, so presenting
        them on the loop's scale would read as >100% of a wall clock.
        """
        seconds = self.worker_seconds.get(phase.key, 0.0)
        total = self.scoped_seconds(WORKER)
        return {
            "key": phase.key,
            "label": phase.label,
            "kind": phase.kind,
            "scope": WORKER,
            "unit": phase.unit,
            "seconds": round(seconds, 4),
            "calls": self.worker_counts.get(phase.key, 0),
            "ms_per_image": (1000.0 * seconds / images) if images else None,
            "percent": (100.0 * seconds / total) if total else None,
            "gpu_event_seconds": None,
            "gpu_event_samples": None,
        }


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
