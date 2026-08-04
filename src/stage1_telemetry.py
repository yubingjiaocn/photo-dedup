"""Stage 1 phase telemetry: where the wall clock actually goes, cheaply.

Why this exists
---------------
A Windows run showed 1000 images in 461s of feature loop with the GPU pulsing
between 0% and 81% and only 4.8 GB VRAM in use: the GPU was starved, but the
log could not say by what. This module makes the loop self-explaining without
turning it into a profiler.

Cost and honesty rules (both matter)
------------------------------------
* **One ``perf_counter`` pair per phase per batch.** Accumulation is a float add
  into a dict; there is no per-micro-op CUDA synchronisation, because
  ``torch.cuda.synchronize()`` inside a hot loop would serialise the very
  pipelining we are trying to measure and would itself change the timings.
* **Host wall time is labelled as host wall time.** With an async CUDA backend,
  a "GPU" phase measured on the host is really *submit + whatever the driver
  made us wait for*. Every phase carries a ``kind`` (``host`` or ``gpu_event``)
  and the report prints that label, so nobody reads a queueing artefact as
  kernel time.
* **Optional GPU-event timing, off the hot path.** When CUDA is available the
  embedding/quality phases can additionally be measured with CUDA events on a
  sampled subset of batches (default every 16th). That is the only place where a
  synchronise happens, it is per batch (not per op), and it is skipped entirely
  on CPU or when the sample rate is 0.
* **Unaccounted time is reported, never hidden.** ``batch_total`` minus the sum
  of the phases is printed as ``unaccounted``, so an under-instrumented step
  shows up instead of silently inflating a neighbour.

The vocabulary is fixed in :data:`PHASES` so the summary is stable enough to
diff between runs, and so a missing optional component (no CLIP-IQA, no YuNet,
no MediaPipe) simply reports 0 counts instead of breaking the format.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# (key, human label, timing kind). ``gpu_event`` phases are the ones that can
# additionally be sampled with CUDA events.
PHASES: Tuple[Tuple[str, str, str], ...] = (
    ("source_open_read", "source open + read (HDD)", "host"),
    ("decode", "JPEG decode", "host"),
    ("exif_meta", "EXIF/metadata from decode", "host"),
    ("hash_sha256", "SHA-256 of source bytes", "host"),
    ("phash", "perceptual hash", "host"),
    ("embed_preprocess", "embedding preprocess (resize/normalise)", "host"),
    ("embed_inference", "embedding inference", "gpu_event"),
    ("quality_preprocess", "IQA preprocess (bounded resize)", "host"),
    ("quality_musiq", "IQA MUSIQ", "gpu_event"),
    ("quality_clipiqa", "IQA CLIP-IQA", "gpu_event"),
    ("quality_sharpness", "sharpness (CPU)", "host"),
    ("faces_yunet", "face detect YuNet", "host"),
    ("face_quality", "face quality (CPU)", "host"),
    ("exposure", "exposure metrics (CPU)", "host"),
    ("eye_detection", "eye detection", "host"),
    ("scene_routing", "scene routing (shadow)", "host"),
    ("thumbnail_resize", "thumbnail resize", "host"),
    ("thumbnail_encode_write", "thumbnail encode + write (SSD)", "host"),
    ("db_write", "DB write (executemany)", "host"),
    ("db_commit", "DB commit", "host"),
    ("other_cpu", "other CPU work", "host"),
)
PHASE_KEYS: Tuple[str, ...] = tuple(key for key, _label, _kind in PHASES)
_LABELS: Dict[str, str] = {key: label for key, label, _kind in PHASES}
_KINDS: Dict[str, str] = {key: kind for key, _label, kind in PHASES}

DEFAULT_GPU_EVENT_EVERY = 16
_KIND_NOTE = {"host": "host-wall", "gpu_event": "host-wall"}


class _Span:
    """Context manager returned by :meth:`Telemetry.phase`."""

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
    """Zero-cost stand-in used when telemetry is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


_NULL_SPAN = _NullSpan()
# Public no-op span so callers can instrument unconditionally without a branch.
NULL_SPAN = _NULL_SPAN


class Telemetry:
    """Cumulative per-phase seconds/counts for one Stage 1 run."""

    def __init__(self, enabled: bool = True, gpu_event_every: int = DEFAULT_GPU_EVENT_EVERY,
                 torch_module: Any = None) -> None:
        self.enabled = bool(enabled)
        self.gpu_event_every = max(0, int(gpu_event_every))
        self.seconds: Dict[str, float] = {key: 0.0 for key in PHASE_KEYS}
        self.counts: Dict[str, int] = {key: 0 for key in PHASE_KEYS}
        self.gpu_event_seconds: Dict[str, float] = {}
        self.gpu_event_counts: Dict[str, int] = {}
        self.batches = 0
        self.images = 0
        self.batch_seconds: List[float] = []
        self.total_seconds = 0.0
        self.notes: List[str] = []
        self._torch = torch_module
        self._cuda_ready: Optional[bool] = None

    # -- recording ----------------------------------------------------------
    def phase(self, key: str, count: int = 1) -> Any:
        """Time one phase. Unknown keys fold into ``other_cpu`` (never crash)."""
        if not self.enabled:
            return _NULL_SPAN
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

    def batch(self, images: int) -> "_BatchSpan":
        return _BatchSpan(self, images)

    def record_batch(self, seconds: float, images: int) -> None:
        if not self.enabled:
            return
        self.batches += 1
        self.images += int(images)
        self.total_seconds += float(seconds)
        self.batch_seconds.append(float(seconds))

    # -- optional CUDA-event sampling --------------------------------------
    def gpu_sampling_now(self) -> bool:
        """True when this batch should also be measured with CUDA events."""
        if not self.enabled or self.gpu_event_every <= 0 or not self._cuda_available():
            return False
        return self.batches % self.gpu_event_every == 0

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
                self.note("GPU event timing unavailable (no CUDA); all phases are host-wall")
        return bool(self._cuda_ready)

    def gpu_event_phase(self, key: str) -> Any:
        """Measure one phase with CUDA events (synchronises once, per batch)."""
        if not self.gpu_sampling_now():
            return _NULL_SPAN
        return _GpuEventSpan(self, key)

    def add_gpu_event(self, key: str, seconds: float) -> None:
        self.gpu_event_seconds[key] = self.gpu_event_seconds.get(key, 0.0) + float(seconds)
        self.gpu_event_counts[key] = self.gpu_event_counts.get(key, 0) + 1

    # -- derived ------------------------------------------------------------
    def accounted_seconds(self) -> float:
        return sum(self.seconds.values())

    def unaccounted_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.accounted_seconds())

    def percentiles(self) -> Dict[str, Optional[float]]:
        return {"p50": _percentile(self.batch_seconds, 0.50),
                "p95": _percentile(self.batch_seconds, 0.95)}

    def snapshot(self) -> Dict[str, Any]:
        """Plain dict for reports/JSON. Only phases with time or counts appear."""
        phases = []
        for key in PHASE_KEYS:
            seconds = self.seconds[key]
            count = self.counts[key]
            if seconds <= 0.0 and count == 0:
                continue
            phases.append({
                "key": key,
                "label": _LABELS[key],
                "kind": _KINDS[key],
                "seconds": round(seconds, 4),
                "count": count,
                "ms_per_image": (1000.0 * seconds / self.images) if self.images else None,
                "percent": (100.0 * seconds / self.total_seconds) if self.total_seconds else None,
                "gpu_event_seconds": (round(self.gpu_event_seconds[key], 4)
                                      if key in self.gpu_event_seconds else None),
                "gpu_event_batches": self.gpu_event_counts.get(key),
            })
        unaccounted = self.unaccounted_seconds()
        return {
            "enabled": self.enabled,
            "images": self.images,
            "batches": self.batches,
            "total_seconds": round(self.total_seconds, 4),
            "accounted_seconds": round(self.accounted_seconds(), 4),
            "unaccounted_seconds": round(unaccounted, 4),
            "unaccounted_percent": ((100.0 * unaccounted / self.total_seconds)
                                    if self.total_seconds else None),
            "batch_percentiles": self.percentiles(),
            "gpu_event_sampling_every": self.gpu_event_every,
            "phases": phases,
            "notes": list(self.notes),
        }


class _BatchSpan:
    """Times a whole batch so ``unaccounted`` can be computed honestly."""

    __slots__ = ("_sink", "_images", "_start")

    def __init__(self, sink: Telemetry, images: int) -> None:
        self._sink = sink
        self._images = int(images)
        self._start = 0.0

    def __enter__(self) -> "_BatchSpan":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self._sink.record_batch(time.perf_counter() - self._start, self._images)
        return False


class _GpuEventSpan:
    """CUDA-event timing for one phase of one sampled batch."""

    __slots__ = ("_sink", "_key", "_start", "_end", "_ok")

    def __init__(self, sink: Telemetry, key: str) -> None:
        self._sink = sink
        self._key = key
        self._ok = False
        self._start = None
        self._end = None

    def __enter__(self) -> "_GpuEventSpan":
        torch = self._sink._torch
        try:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if not self._ok:
            return False
        try:
            self._end.record()
            self._end.synchronize()
            self._sink.add_gpu_event(self._key, self._start.elapsed_time(self._end) / 1000.0)
        except Exception:
            self._sink.note("GPU event timing failed; host-wall numbers only")
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


# --- formatting ------------------------------------------------------------

def render_lines(snapshot: Dict[str, Any], max_phases: int = 0) -> List[str]:
    """Fixed-width, greppable summary shared by the log and performance.txt."""
    if not snapshot or not snapshot.get("enabled"):
        return ["Stage 1 phase breakdown: disabled"]
    images = int(snapshot.get("images") or 0)
    total = float(snapshot.get("total_seconds") or 0.0)
    lines = [
        "Stage 1 phase breakdown (cumulative, host-wall unless marked)",
        f"images={images} batches={snapshot.get('batches', 0)} "
        f"batch_total={total:.2f}s"
        + (f" ({1000.0 * total / images:.1f} ms/image)" if images else ""),
        f"{'phase':<34}{'count':>8}{'seconds':>10}{'ms/img':>9}{'%':>7}  timing",
    ]
    phases = list(snapshot.get("phases") or [])
    phases.sort(key=lambda item: float(item.get("seconds") or 0.0), reverse=True)
    if max_phases > 0:
        phases = phases[:max_phases]
    for phase in phases:
        lines.append(_phase_line(phase))
    lines.append(_unaccounted_line(snapshot))
    percentiles = snapshot.get("batch_percentiles") or {}
    lines.append(
        f"batch wall time p50={_fmt_seconds(percentiles.get('p50'))} "
        f"p95={_fmt_seconds(percentiles.get('p95'))}"
    )
    gpu_lines = _gpu_event_lines(snapshot)
    lines.extend(gpu_lines)
    for note in snapshot.get("notes") or []:
        lines.append(f"note: {note}")
    lines.append(
        "host-wall means time measured on the CPU thread; with an async CUDA backend it "
        "includes submit + wait, so it is an upper bound on kernel time."
    )
    return lines


def _phase_line(phase: Dict[str, Any]) -> str:
    ms = phase.get("ms_per_image")
    percent = phase.get("percent")
    return (
        f"{str(phase.get('label'))[:33]:<34}"
        f"{int(phase.get('count') or 0):>8}"
        f"{float(phase.get('seconds') or 0.0):>10.2f}"
        f"{(f'{ms:.2f}' if ms is not None else '-'):>9}"
        f"{(f'{percent:.1f}' if percent is not None else '-'):>7}"
        f"  {_KIND_NOTE.get(str(phase.get('kind')), 'host-wall')}"
    )


def _unaccounted_line(snapshot: Dict[str, Any]) -> str:
    seconds = float(snapshot.get("unaccounted_seconds") or 0.0)
    percent = snapshot.get("unaccounted_percent")
    images = int(snapshot.get("images") or 0)
    ms = (1000.0 * seconds / images) if images else None
    return (
        f"{'unaccounted (loop overhead)':<34}{'-':>8}{seconds:>10.2f}"
        f"{(f'{ms:.2f}' if ms is not None else '-'):>9}"
        f"{(f'{percent:.1f}' if percent is not None else '-'):>7}  host-wall"
    )


def _gpu_event_lines(snapshot: Dict[str, Any]) -> List[str]:
    sampled = [phase for phase in (snapshot.get("phases") or [])
               if phase.get("gpu_event_seconds") is not None]
    if not sampled:
        return ["GPU event timing: not sampled (CPU run, disabled, or no sampled batch yet)"]
    every = snapshot.get("gpu_event_sampling_every")
    lines = [f"GPU event timing (CUDA events, sampled every {every} batch(es))"]
    for phase in sampled:
        batches = int(phase.get("gpu_event_batches") or 0)
        seconds = float(phase.get("gpu_event_seconds") or 0.0)
        per_batch = (seconds / batches) if batches else None
        lines.append(
            f"  {str(phase.get('label'))[:33]:<33} batches={batches:<5} "
            f"gpu={seconds:.3f}s"
            + (f" ({per_batch * 1000.0:.1f} ms/batch)" if per_batch is not None else "")
        )
    return lines


def _fmt_seconds(value: Optional[float]) -> str:
    return f"{value:.3f}s" if value is not None else "unknown"


def print_summary(snapshot: Dict[str, Any], prefix: str = "[stage1]") -> None:
    for line in render_lines(snapshot):
        print(f"{prefix} {line}")


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
