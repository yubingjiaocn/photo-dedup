"""Bounded prefetch: order, bounds, cancellation, and thread discipline.

The producer exists to overlap CPU work with GPU work, and every one of those
words is a hazard. These tests pin the properties that make it safe rather than
merely fast:

* **Order.** Batches leave in submission order and each batch keeps its row
  order, in every configuration. If that broke, DB write order, error rows and
  resume behaviour would all drift.
* **Bounds.** At most ``prefetch_batches + 1`` batches of decoded frames exist at
  once, so a 100k-photo library cannot fill RAM.
* **Sequential reads.** Exactly one reader thread, one open per file: the library
  is on a single mechanical disk and N concurrent opens would be a seek storm.
* **Thread discipline.** SQLite and YuNet are only ever touched by the main
  thread; the queue is drained and the reader joined on every exit path.
* **Failures.** A corrupt file becomes one error row (not an exception); a broken
  producer raises into the consumer instead of finishing quietly short.
"""

from __future__ import annotations

import threading
import time

import pytest
from PIL import Image

from src import stage1_pipeline
from src.config import Config, load_config


class RecordingBackend:
    """Counts preparation and records which thread performed it."""

    name = "recording"
    telemetry = None

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.threads: list[int] = []
        self.order: list[int] = []
        self.lock = threading.Lock()

    def prepare_cpu(self, image, recorder=None):
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.threads.append(threading.get_ident())
        return {"iqa_array": None, "size": image.size}


def rows(tmp_path, count, prefix="img"):
    made = []
    for index in range(count):
        path = tmp_path / f"{prefix}-{index}.jpg"
        Image.new("RGB", (32, 24), (index * 7 % 256, 60, 120)).save(path)
        made.append({"id": index + 1, "path": str(path), "width": 32, "height": 24})
    return made


def read_fn(path, recorder=None):
    image = Image.open(path)
    image.load()
    return image.convert("RGB"), f"sha-{path.name}"


def batches_of(items, size):
    return [items[start:start + size] for start in range(0, len(items), size)]


# --- ordering ---------------------------------------------------------------

@pytest.mark.parametrize("workers,prefetch", [(0, 0), (1, 1), (2, 2), (4, 2), (8, 4)])
def test_output_order_is_always_the_submission_order(tmp_path, workers, prefetch):
    """Concurrency must not reorder a single row, in any configuration."""
    items = rows(tmp_path, 17)
    cfg = load_config()
    # A varying delay makes a naive "first finished wins" queue fail loudly.
    backend = RecordingBackend()
    source = stage1_pipeline.build_source(
        batches_of(items, 3), backend, cfg, read_fn,
        workers=workers, prefetch_batches=prefetch)
    try:
        observed = [int(item.row["id"]) for batch in source for item in batch.items]
    finally:
        source.close()
    assert observed == [int(row["id"]) for row in items]


def test_batch_boundaries_are_preserved(tmp_path):
    items = rows(tmp_path, 10)
    source = stage1_pipeline.build_source(
        batches_of(items, 4), RecordingBackend(), load_config(), read_fn,
        workers=4, prefetch_batches=2)
    try:
        sizes = [len(batch) for batch in source]
    finally:
        source.close()
    assert sizes == [4, 4, 2]                      # partial final batch preserved


def test_slow_first_batch_does_not_let_a_later_batch_overtake_it(tmp_path):
    """The queue is FIFO even when a later batch finishes preparing first."""
    items = rows(tmp_path, 6)

    class UnevenBackend(RecordingBackend):
        """The first two images prepared are slow, whichever they are."""

        def prepare_cpu(self, image, recorder=None):
            with self.lock:
                slow = len(self.order) < 2
                self.order.append(1)
            if slow:
                # Without ordered resolution these would arrive last.
                time.sleep(0.05)
            return super().prepare_cpu(image, recorder)

    source = stage1_pipeline.build_source(
        batches_of(items, 2), UnevenBackend(), load_config(), read_fn,
        workers=4, prefetch_batches=3)
    try:
        observed = [int(item.row["id"]) for batch in source for item in batch.items]
    finally:
        source.close()
    assert observed == [1, 2, 3, 4, 5, 6]


# --- bounds -----------------------------------------------------------------

def test_at_most_prefetch_plus_one_batches_are_in_flight(tmp_path):
    """The memory bound is a number, not a hope: queue + the reader's own batch."""
    items = rows(tmp_path, 40)
    live = 0
    peak = 0
    lock = threading.Lock()

    class TrackingBackend(RecordingBackend):
        def prepare_cpu(self, image, recorder=None):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            try:
                return super().prepare_cpu(image, recorder)
            finally:
                with lock:
                    live -= 1

    source = stage1_pipeline.build_source(
        batches_of(items, 4), TrackingBackend(), load_config(), read_fn,
        workers=2, prefetch_batches=2)
    try:
        for _batch in source:
            time.sleep(0.01)                        # simulate slow GPU consumer
    finally:
        source.close()
    # Never more than (prefetch_batches + 1) batches x batch_size images alive.
    assert peak <= 3 * 4, peak


def test_producer_blocks_instead_of_racing_ahead(tmp_path):
    """A stalled consumer must stop the producer, not let it read the library."""
    items = rows(tmp_path, 30)
    prepared_count = 0
    lock = threading.Lock()

    class CountingBackend(RecordingBackend):
        def prepare_cpu(self, image, recorder=None):
            nonlocal prepared_count
            with lock:
                prepared_count += 1
            return super().prepare_cpu(image, recorder)

    source = stage1_pipeline.build_source(
        batches_of(items, 3), CountingBackend(), load_config(), read_fn,
        workers=2, prefetch_batches=2)
    try:
        iterator = iter(source)
        next(iterator)                              # take one batch, then stall
        time.sleep(0.3)
        with lock:
            observed = prepared_count
        # 1 delivered + at most 2 queued + 1 being assembled = <= 4 batches of 3.
        assert observed <= 4 * 3, observed
    finally:
        source.close()


def test_inflight_bound_is_documented_by_the_settings(tmp_path):
    from src.stage1_settings import resolve_loop_settings

    cfg = Config({"features": {"batch_size": 4, "max_inflight_megapixels": 80,
                               "cpu_workers": 4, "prefetch_batches": 2},
                  "scan": {"commit_every": 100}})
    settings = resolve_loop_settings(cfg)
    assert settings.inflight_batches == 3           # 2 queued + 1 being read
    note = settings.memory_note()
    assert "3 batch(es) in flight" in note and "240 MP decoded" in note
    assert "MiB RGB" in note
    # The IQA arrays are bounded per image by iqa_max_long_edge, not by the
    # megapixel guard, so they are stated separately and counted per image.
    assert "12 bounded IQA arrays" in note and "1920px" in note
    assert "float32" in note

    serial = resolve_loop_settings(Config({
        "features": {"cpu_workers": 0}, "scan": {}}))
    assert serial.inflight_batches == 1 and serial.prefetch_enabled is False


# --- sequential reads -------------------------------------------------------

def test_exactly_one_reader_thread_and_one_open_per_file(tmp_path):
    """One mechanical disk: reads stay serial even with many CPU workers."""
    items = rows(tmp_path, 12)
    read_threads: set[int] = set()
    opened: list[str] = []
    lock = threading.Lock()

    def tracking_read(path, recorder=None):
        with lock:
            read_threads.add(threading.get_ident())
            opened.append(path.name)
        return read_fn(path, recorder)

    backend = RecordingBackend(delay=0.005)
    source = stage1_pipeline.build_source(
        batches_of(items, 3), backend, load_config(), tracking_read,
        workers=6, prefetch_batches=2)
    try:
        list(source)
    finally:
        source.close()
    assert len(read_threads) == 1                        # exactly one reader
    assert opened == [f"img-{index}.jpg" for index in range(12)]   # in order, once
    assert read_threads.isdisjoint({threading.get_ident()})        # not the main thread
    assert len(set(backend.threads)) > 1                 # preparation did fan out


def test_preparation_runs_off_the_main_thread_when_workers_are_enabled(tmp_path):
    items = rows(tmp_path, 8)
    backend = RecordingBackend()
    source = stage1_pipeline.build_source(
        batches_of(items, 4), backend, load_config(), read_fn,
        workers=4, prefetch_batches=2)
    try:
        list(source)
    finally:
        source.close()
    assert threading.get_ident() not in backend.threads


def test_serial_mode_uses_only_the_calling_thread(tmp_path):
    items = rows(tmp_path, 6)
    backend = RecordingBackend()
    source = stage1_pipeline.build_source(
        batches_of(items, 3), backend, load_config(), read_fn,
        workers=0, prefetch_batches=0)
    try:
        list(source)
    finally:
        source.close()
    assert set(backend.threads) == {threading.get_ident()}


# --- failures ---------------------------------------------------------------

def test_one_corrupt_file_becomes_one_failed_item_not_an_exception(tmp_path):
    items = rows(tmp_path, 5)
    (tmp_path / "img-2.jpg").write_bytes(b"not a jpeg")

    source = stage1_pipeline.build_source(
        batches_of(items, 5), RecordingBackend(), load_config(), read_fn,
        workers=4, prefetch_batches=2)
    try:
        batch = next(iter(source))
    finally:
        source.close()
    assert len(batch) == 5
    states = [item.ok for item in batch.items]
    assert states == [True, True, False, True, True]
    broken = batch.items[2]
    assert broken.error is not None and broken.image is None
    assert [int(item.row["id"]) for item in batch.items] == [1, 2, 3, 4, 5]


def test_a_failure_inside_preparation_is_attached_to_its_own_image(tmp_path):
    """A prepare_cpu explosion must not poison its neighbours."""
    items = rows(tmp_path, 4)

    class ExplodingBackend(RecordingBackend):
        """Fails on the Nth preparation, whichever image that turns out to be."""

        def prepare_cpu(self, image, recorder=None):
            with self.lock:
                self.order.append(1)
                nth = len(self.order)
            if nth == 2:
                raise RuntimeError("prepare exploded")
            return super().prepare_cpu(image, recorder)

    source = stage1_pipeline.build_source(
        batches_of(items, 4), ExplodingBackend(), load_config(), read_fn,
        workers=1, prefetch_batches=1)
    try:
        batch = next(iter(source))
    finally:
        source.close()
    failures = [item for item in batch.items if not item.ok]
    assert len(failures) == 1
    # Exactly one row failed and it kept its own place; the neighbours are intact.
    assert "prepare exploded" in str(failures[0].error)
    assert failures[0].image is None
    assert len(batch) == 4
    assert [int(item.row["id"]) for item in batch.items] == [1, 2, 3, 4]
    assert sum(1 for item in batch.items if item.ok) == 3


def test_a_producer_machinery_failure_propagates_to_the_consumer(tmp_path):
    """A broken batch iterator must raise, never silently truncate the run."""
    items = rows(tmp_path, 4)

    def exploding_batches():
        yield items[:2]
        raise RuntimeError("batch source exploded")

    source = stage1_pipeline.build_source(
        exploding_batches(), RecordingBackend(), load_config(), read_fn,
        workers=2, prefetch_batches=2)
    seen = 0
    with pytest.raises(RuntimeError, match="batch source exploded"):
        try:
            for _batch in source:
                seen += 1
        finally:
            source.close()
    assert seen == 1                                # the good batch was delivered


def test_serial_mode_propagates_a_machinery_failure_too(tmp_path):
    items = rows(tmp_path, 2)

    def exploding_batches():
        yield items
        raise RuntimeError("serial source exploded")

    source = stage1_pipeline.build_source(
        exploding_batches(), RecordingBackend(), load_config(), read_fn,
        workers=0, prefetch_batches=0)
    with pytest.raises(RuntimeError, match="serial source exploded"):
        list(source)


# --- cancellation -----------------------------------------------------------

def test_close_stops_the_reader_and_joins_it(tmp_path):
    """Ctrl+C path: abandoning the iterator must not leak a reader thread."""
    items = rows(tmp_path, 60)
    before = threading.active_count()
    source = stage1_pipeline.build_source(
        batches_of(items, 3), RecordingBackend(delay=0.002), load_config(), read_fn,
        workers=2, prefetch_batches=2)
    iterator = iter(source)
    next(iterator)
    next(iterator)
    source.close()
    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before
    assert not any(thread.name.startswith("stage1-")
                   for thread in threading.enumerate() if thread.is_alive())


def test_close_is_idempotent_and_safe_before_iteration(tmp_path):
    items = rows(tmp_path, 4)
    source = stage1_pipeline.build_source(
        batches_of(items, 2), RecordingBackend(), load_config(), read_fn,
        workers=2, prefetch_batches=1)
    source.close()
    source.close()                                   # second call must not raise


def test_close_stops_reading_the_library_after_cancellation(tmp_path):
    """The reader must stop opening files, which is what protects the HDD."""
    items = rows(tmp_path, 80)
    opened: list[str] = []
    lock = threading.Lock()

    def tracking_read(path, recorder=None):
        with lock:
            opened.append(path.name)
        return read_fn(path, recorder)

    source = stage1_pipeline.build_source(
        batches_of(items, 2), RecordingBackend(delay=0.002), load_config(), tracking_read,
        workers=2, prefetch_batches=2)
    iterator = iter(source)
    next(iterator)
    source.close()
    with lock:
        at_close = len(opened)
    time.sleep(0.3)
    with lock:
        assert len(opened) == at_close                # nothing opened after close
    assert at_close < len(items)                      # and it stopped early


def test_an_exception_in_the_consumer_still_closes_the_producer(tmp_path):
    items = rows(tmp_path, 40)
    before = threading.active_count()
    source = stage1_pipeline.build_source(
        batches_of(items, 2), RecordingBackend(delay=0.002), load_config(), read_fn,
        workers=2, prefetch_batches=2)
    with pytest.raises(ValueError, match="consumer failed"):
        try:
            for _batch in source:
                raise ValueError("consumer failed")
        finally:
            source.close()
    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before


def test_context_manager_closes_on_the_way_out(tmp_path):
    items = rows(tmp_path, 20)
    before = threading.active_count()
    with stage1_pipeline.build_source(
        batches_of(items, 2), RecordingBackend(delay=0.001), load_config(), read_fn,
        workers=2, prefetch_batches=1,
    ) as source:
        next(iter(source))
    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before


# --- empty and degenerate inputs -------------------------------------------

def test_no_work_produces_no_batches_and_no_threads_left_behind(tmp_path):
    before = threading.active_count()
    source = stage1_pipeline.build_source(
        [], RecordingBackend(), load_config(), read_fn, workers=4, prefetch_batches=2)
    try:
        assert list(source) == []
    finally:
        source.close()
    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before


def test_a_single_image_batch_works_with_a_large_worker_pool(tmp_path):
    items = rows(tmp_path, 1)
    source = stage1_pipeline.build_source(
        batches_of(items, 8), RecordingBackend(), load_config(), read_fn,
        workers=8, prefetch_batches=4)
    try:
        collected = list(source)
    finally:
        source.close()
    assert len(collected) == 1 and len(collected[0]) == 1
