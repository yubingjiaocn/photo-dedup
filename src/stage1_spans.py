"""Span objects for :mod:`src.stage1_telemetry`: the actual measuring devices.

Split out so the collector file stays about accounting and this one is only
about *how* a measurement is taken. Every span is a context manager that costs
one ``perf_counter`` pair (plus, for the CUDA lane, two asynchronous event
records) and reports into a sink implementing ``add``/``add_worker``.

Four kinds exist, and the difference between them is the whole honesty story:

``_Span``
    Main-thread work. Accumulates into the sink's loop/setup books.
``_WorkerSpan``
    Producer-thread work. Accumulates into a separate, lock-protected set of
    counters, because it runs concurrently with the main thread and must never
    be added to the loop total.
``_BatchSpan``
    One batch. Latches the CUDA-event sampling decision on entry and resolves
    the queued events on exit -- *after* the batch's wall clock has been read,
    so the single synchronisation of a sampled batch cannot inflate any number.
``_GpuEventSpan``
    Records (never synchronises) CUDA events around one sampled call.

``NULL_SPAN`` is the zero-cost stand-in returned whenever telemetry, or this
particular sample, is disabled.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple


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
        # Batch wall time is taken *before* resolving CUDA events, so the single
        # synchronisation of a sampled batch cannot inflate any reported number.
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


PendingEvents = List[Tuple[str, Any, Any]]
