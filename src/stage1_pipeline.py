"""Stage 1 producer: bounded, ordered prefetch of prepared images.

Why this exists
---------------
The Windows baseline spent 504.80 s of a 510.57 s Stage 1 inside the batch loop
for 1000 images, with the GPU at 20-25%. The breakdown said the loop was CPU
bound, not GPU bound: IQA preprocess 82.82 s, embedding preprocess 51.50 s,
sharpness 49.39 s, JPEG decode 40.18 s, exposure 31.14 s and the HDD read
24.99 s all happened on the same thread that then waited for MUSIQ and CLIP-IQA.
Measured on this box (``docs/STAGE1_THROUGHPUT.md``), that per-image CPU work
scales 3.18x on four threads and is essentially free while a GPU lane is busy
(3.87x when overlapped), because Pillow, numpy and the resize all release the
GIL.

The three lanes
---------------
1. **One reader thread.** It calls the *same* read+decode function the serial
   path uses, one file at a time, in inventory order. A single reader is
   deliberate: the library lives on one mechanical disk, so N threads opening N
   different files would turn a sequential pass into a seek storm. Each file is
   still read exactly once and decoded exactly once, and the read is still the
   read that feeds the SHA-256.
2. **A small worker pool.** Per-image, purely functional CPU preparation of an
   already-decoded frame: perceptual hash, the bounded IQA array + sharpness,
   the embedding preprocess, the exposure resize and the face-detector input.
   Nothing here touches SQLite, the YuNet detector, the thumbnail writer or any
   other stateful component.
3. **The main thread.** Everything stateful or GPU-bound: the batched model
   calls, YuNet (``FaceDetectorYN`` carries an input size, so it is not
   thread-safe), the thumbnail cache, and every SQLite statement.

Bounds and order
----------------
* The queue holds at most ``prefetch_batches`` batches of futures, and the
  reader can be assembling one more while blocked on ``put``, so **at most
  ``prefetch_batches + 1`` batches of decoded frames exist at once**. Each batch
  is already bounded by ``features.max_inflight_megapixels``, which is what makes
  the memory ceiling a number rather than a hope:
  ``(prefetch_batches + 1) x max_inflight_megapixels`` of decoded pixels, plus
  each frame's bounded IQA array. With the defaults (2 batches, 80 MP) that is
  ~240 MP of RGB (~720 MiB) plus ~400 MiB of IQA arrays.
* Compressed bytes are handed to the decoder and dropped immediately; they are
  never accumulated.
* Output order is the submission order, always. Batches leave the queue in the
  order they were read and each batch's items keep their row order, because the
  consumer resolves that batch's futures in order. Concurrency never reorders a
  result, so the DB write order, the error rows and resume behaviour are
  identical to the serial path.
* A per-file failure is data, not an exception: it becomes a failed
  :class:`PreparedImage` exactly where the serial path would have produced an
  error row. A failure of the machinery itself (not of one file) propagates to
  the consumer on the next iteration and is not silently dropped.
* :meth:`_Prefetcher.close` is idempotent and always runs (``finally``), so
  Ctrl+C, a mid-loop exception, or a ``break`` cancels queued work, drains the
  queue and joins the reader instead of leaking threads.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from . import exposure
from . import quality as Q
from . import stage1_telemetry as telemetry_mod

# Sentinel put on the queue when the reader has finished normally.
_DONE = object()


@dataclass
class PreparedImage:
    """One image's CPU-side inputs, or the failure that stopped it.

    ``image`` is the single decoded RGB frame every consumer shares: the batched
    model calls, YuNet, face quality, scene routing and the thumbnail cache all
    receive this same object, so Stage 1 keeps its one-read/one-decode guarantee.
    """

    row: Any
    image: Any = None
    sha256: Optional[str] = None
    phash: Optional[bytes] = None
    exposure_rgb: Optional[np.ndarray] = None
    backend: Dict[str, Any] = field(default_factory=dict)
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.image is not None


@dataclass
class PreparedBatch:
    """One guarded batch, prepared and in submission order."""

    items: List[PreparedImage]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def rows(self) -> List[Any]:
        return [item.row for item in self.items]


def prepare_cpu(item: PreparedImage, backend: Any, cfg: Any, recorder: Any) -> PreparedImage:
    """Per-image CPU preparation of an already-decoded frame.

    Pure with respect to shared state: it reads ``item.image`` and writes only
    into the returned :class:`PreparedImage`. Safe to run on a worker thread and
    equally correct on the main thread, which is what makes ``cpu_workers: 0``
    a real serial fallback rather than a different pipeline.
    """
    if not item.ok:
        return item
    try:
        with recorder.phase("phash"):
            item.phash = Q.phash_bytes(item.image)
        with recorder.phase("exposure_preprocess"):
            item.exposure_rgb = exposure.prepare(
                item.image, int(cfg.features.get("exposure_long_edge", 512))
            )
        item.backend = backend.prepare_cpu(item.image, recorder)
    except Exception as exc:  # same tolerance the serial path had per file
        item.error = exc
        item.image = None
    return item


class _Recorder:
    """Routes producer-thread work to the lock-protected producer books.

    Exposes the same ``phase``/``add`` surface as :class:`src.stage1_telemetry.Telemetry`
    so it can be handed straight to the read+decode function, which is what keeps
    a reader thread from mutating the main thread's dictionaries.
    """

    __slots__ = ("_telemetry",)

    def __init__(self, telemetry: Any) -> None:
        self._telemetry = telemetry

    def phase(self, key: str, count: int = 1) -> Any:
        sink = self._telemetry
        return sink.worker_phase(key, count) if sink is not None else telemetry_mod.NULL_SPAN

    def add(self, key: str, seconds: float, count: int = 1) -> None:
        sink = self._telemetry
        if sink is not None:
            sink.add_worker(key, seconds, count)


class _MainRecorder:
    """Routes preparation phases to the main-thread (in-loop) books."""

    __slots__ = ("_telemetry",)

    def __init__(self, telemetry: Any) -> None:
        self._telemetry = telemetry

    def phase(self, key: str, count: int = 1) -> Any:
        sink = self._telemetry
        return sink.phase(key, count) if sink is not None else telemetry_mod.NULL_SPAN

    def add(self, key: str, seconds: float, count: int = 1) -> None:
        sink = self._telemetry
        if sink is not None:
            sink.add(key, seconds, count)


def main_recorder(telemetry: Any) -> _MainRecorder:
    """Recorder for preparation that really did happen on the main thread."""
    return _MainRecorder(telemetry)


def read_and_decode(row: Any, read_fn: Callable[..., Any], recorder: Any) -> PreparedImage:
    """Read + decode one row, turning a per-file failure into data.

    ``recorder`` (not the raw telemetry object) is handed to ``read_fn`` so the
    read/decode/hash spans land in the books of the thread that ran them.
    """
    try:
        image, sha256 = read_fn(Path(row["path"]), recorder)
    except Exception as exc:
        return PreparedImage(row=row, error=exc)
    return PreparedImage(row=row, image=image, sha256=sha256)


def serial_batches(
    batches: Iterable[Sequence[Any]], backend: Any, cfg: Any,
    read_fn: Callable[..., Any], telemetry: Any = None,
) -> Iterator[PreparedBatch]:
    """Prepare batches inline on the calling thread (``cpu_workers: 0``).

    Byte-for-byte the same work in the same order as the threaded path; only the
    thread differs. Timing is billed to the loop, because that is where it ran.
    """
    recorder = main_recorder(telemetry)
    for batch in batches:
        items = [prepare_cpu(read_and_decode(row, read_fn, recorder), backend, cfg, recorder)
                 for row in batch]
        yield PreparedBatch(items=items)


class _Prefetcher:
    """Reader thread + bounded worker pool feeding an ordered queue."""

    def __init__(
        self, batches: Iterable[Sequence[Any]], backend: Any, cfg: Any,
        read_fn: Callable[..., Any], workers: int, prefetch_batches: int,
        telemetry: Any = None,
    ) -> None:
        self._batches = batches
        self._backend = backend
        self._cfg = cfg
        self._read_fn = read_fn
        self._telemetry = telemetry
        self._recorder = _Recorder(telemetry)
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(prefetch_batches)))
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                        thread_name_prefix="stage1-prep")
        self._reader = threading.Thread(target=self._read_loop, name="stage1-reader",
                                        daemon=True)
        self._closed = False
        self._started = False

    # -- producer side ------------------------------------------------------
    def _read_loop(self) -> None:
        """Sequential reads, in order; per-image CPU work goes to the pool."""
        try:
            for batch in self._batches:
                if self._stop.is_set():
                    break
                futures: List[Future] = []
                for row in batch:
                    if self._stop.is_set():
                        break
                    # The read+decode is deliberately serial: one mechanical disk.
                    item = read_and_decode(row, self._read_fn, self._recorder)
                    futures.append(self._pool.submit(
                        prepare_cpu, item, self._backend, self._cfg, self._recorder))
                self._put(futures)
        except BaseException as exc:  # machinery failure, not a per-file failure
            self._put(exc, force=True)
        finally:
            self._put(_DONE, force=True)

    def _put(self, payload: Any, force: bool = False) -> None:
        """Block for queue space, but stay responsive to ``close()``."""
        while not self._stop.is_set() or force:
            try:
                self._queue.put(payload, timeout=0.1)
                return
            except queue.Full:
                if force and self._stop.is_set():
                    return          # consumer is gone; drop instead of blocking
                continue

    # -- consumer side ------------------------------------------------------
    def __iter__(self) -> Iterator[PreparedBatch]:
        if not self._started:
            self._started = True
            self._reader.start()
        while True:
            payload = self._queue.get()
            if payload is _DONE:
                return
            if isinstance(payload, BaseException):
                raise payload
            if self._telemetry is not None:
                self._telemetry.count("prefetch_batches_delivered")
                self._telemetry.count("prefetch_images_prepared", len(payload))
            # Resolving in order is what makes the output deterministic; the wait
            # is measured, so "the GPU lane starved" stays visible in the report.
            span = (self._telemetry.phase("prefetch_wait")
                    if self._telemetry is not None else telemetry_mod.NULL_SPAN)
            with span:
                items = [future.result() for future in payload]
            yield PreparedBatch(items=items)

    def close(self) -> None:
        """Cancel queued work, drain, and join. Idempotent; safe on Ctrl+C.

        Order matters: ``_stop`` is set *before* the queue is drained, so a reader
        blocked in :meth:`_put` observes the flag and returns instead of refilling
        the space that was just freed. Draining is what unblocks it; joining then
        confirms it is gone rather than assuming so.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        while True:                                  # unblock a reader in put()
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._started:
            self._reader.join(timeout=30.0)
            if self._reader.is_alive() and self._telemetry is not None:
                # Never claim a clean shutdown that did not happen. The thread is a
                # daemon so it cannot keep the process alive, but a reader that
                # would not stop is worth seeing in the report.
                self._telemetry.note(
                    "Stage 1 reader thread did not stop within 30s of cancellation")
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "_Prefetcher":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.close()
        return False


class _SerialSource:
    """Serial producer with the same context-manager contract as the threaded one."""

    def __init__(self, batches: Iterable[Sequence[Any]], backend: Any, cfg: Any,
                 read_fn: Callable[..., Any], telemetry: Any = None) -> None:
        self._iterator = serial_batches(batches, backend, cfg, read_fn, telemetry)

    def __iter__(self) -> Iterator[PreparedBatch]:
        return self._iterator

    def close(self) -> None:
        return None

    def __enter__(self) -> "_SerialSource":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def build_source(
    batches: Iterable[Sequence[Any]], backend: Any, cfg: Any,
    read_fn: Callable[..., Any], workers: int, prefetch_batches: int,
    telemetry: Any = None,
) -> Any:
    """Threaded producer, or the serial one when either knob is 0.

    ``workers=0`` or ``prefetch_batches=0`` reproduces the pre-change pipeline
    exactly (one thread, no queue), which is what the benchmark compares against
    and what a user can fall back to without changing anything else.
    """
    if int(workers) <= 0 or int(prefetch_batches) <= 0:
        return _SerialSource(batches, backend, cfg, read_fn, telemetry)
    return _Prefetcher(batches, backend, cfg, read_fn, workers, prefetch_batches, telemetry)
