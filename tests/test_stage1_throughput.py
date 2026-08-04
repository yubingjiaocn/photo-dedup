"""Stage 1 throughput changes, at run level: same results, honest reporting.

Batched IQA and prefetch are only worth having if the run they produce is
indistinguishable from the serial one. These tests run the real Stage 1 (stub
backend) and compare byte-for-byte, then pin the things concurrency could
plausibly break: SQLite touched from one thread only, resume still skipping
completed work, error rows in the right place, and configuration that fails
loudly instead of being silently clamped.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest
import yaml
from PIL import Image

from src import db, stage0_inventory, stage1_features
from src.config import Config
from src.stage1_settings import (
    DEFAULT_PREFETCH_BATCHES,
    default_cpu_workers,
    resolve_loop_settings,
)


def library(root, count=6, corrupt=()):
    """Mixed orientations and sizes, so IQA shape grouping is exercised."""
    root.mkdir(parents=True, exist_ok=True)
    sizes = [(320, 240), (240, 320), (320, 240), (200, 200), (240, 320), (320, 240)]
    for index in range(count):
        path = root / f"IMG_2026010{index}_120000.jpg"
        if index in corrupt:
            path.write_bytes(b"not a jpeg at all")
            continue
        width, height = sizes[index % len(sizes)]
        Image.new("RGB", (width, height), (10 + index * 30, 60, 120)).save(path, "JPEG")
    return root


def config(tmp_path, root, *, commit_every=None, **features):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(tmp_path / "inventory.sqlite"),
                  "output_dir": str(tmp_path / "out"),
                  "models_dir": str(tmp_path / "models"),
                  "trash": str(tmp_path / "trash")},
        "scan": ({"commit_every": commit_every} if commit_every is not None else {}),
        "features": {"backend": "stub", "batch_size": 3,
                     "thumbnails": {"enabled": True, "max_px": 64},
                     "telemetry": {"enabled": True}, **features},
    }), encoding="utf-8")
    return str(path)


def feature_rows(db_path):
    conn = db.open_db(db_path)
    rows = conn.execute(
        "SELECT file_id, phash, content_sha256, dinov2_embedding, quality_score, "
        "quality_meta, face_count, faces_json, status FROM features ORDER BY file_id"
    ).fetchall()
    result = [dict(row) for row in rows]
    conn.close()
    return result


# --- serial and prefetched runs must agree ---------------------------------

def test_prefetched_run_produces_identical_features_to_the_serial_run(tmp_path):
    """The optimisation must be invisible in the output. Every column, every row."""
    root = library(tmp_path / "photos", 6)

    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    serial_config = config(serial_dir, root, cpu_workers=0, prefetch_batches=0)
    stage0_inventory.run(config_path=serial_config)
    serial_stats = stage1_features.run(config_path=serial_config, backend_override="stub")

    fast_dir = tmp_path / "fast"
    fast_dir.mkdir()
    fast_config = config(fast_dir, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=fast_config)
    fast_stats = stage1_features.run(config_path=fast_config, backend_override="stub")

    serial_rows = feature_rows(serial_dir / "inventory.sqlite")
    fast_rows = feature_rows(fast_dir / "inventory.sqlite")
    assert len(serial_rows) == len(fast_rows) == 6
    for expected, actual in zip(serial_rows, fast_rows):
        assert expected["file_id"] == actual["file_id"]
        assert expected["phash"] == actual["phash"]
        assert expected["content_sha256"] == actual["content_sha256"]
        assert expected["dinov2_embedding"] == actual["dinov2_embedding"]
        assert expected["quality_score"] == pytest.approx(actual["quality_score"])
        assert expected["face_count"] == actual["face_count"]
        assert expected["faces_json"] == actual["faces_json"]
        assert expected["status"] == actual["status"]
        # quality_meta carries the calibrated IQA bounds and the exposure block.
        assert json.loads(expected["quality_meta"]) == json.loads(actual["quality_meta"])
    assert serial_stats["processed"] == fast_stats["processed"] == 6
    assert serial_stats["loop_settings"]["prefetch_enabled"] is False
    assert fast_stats["loop_settings"]["prefetch_enabled"] is True


def test_row_order_and_error_placement_survive_prefetch(tmp_path):
    """One corrupt file in the middle: same error row, same position, both modes."""
    root = library(tmp_path / "photos", 6, corrupt=(2,))

    results = {}
    for name, workers, prefetch in (("serial", 0, 0), ("fast", 4, 2)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        run_config = config(run_dir, root, cpu_workers=workers, prefetch_batches=prefetch)
        stage0_inventory.run(config_path=run_config)
        stats = stage1_features.run(config_path=run_config, backend_override="stub")
        results[name] = (stats, feature_rows(run_dir / "inventory.sqlite"))

    for name, (stats, rows) in results.items():
        assert stats["failed"] == 1, name
        assert stats["succeeded"] == 5, name
        statuses = [row["status"] for row in rows]
        assert statuses.count("done_error") == 1, name
        broken = next(row for row in rows if row["status"] == "done_error")
        assert broken["phash"] is None and broken["content_sha256"] is None
        assert json.loads(broken["quality_meta"])["error"] is True
    serial_rows, fast_rows = results["serial"][1], results["fast"][1]
    assert [row["status"] for row in serial_rows] == [row["status"] for row in fast_rows]
    assert [row["file_id"] for row in serial_rows] == [row["file_id"] for row in fast_rows]


def test_partial_final_batch_is_written_and_committed(tmp_path):
    """7 images at batch_size 3 -> 3/3/1; the tail must not be lost."""
    root = library(tmp_path / "photos", 7)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)
    stats = stage1_features.run(config_path=run_config, backend_override="stub")

    assert stats["processed"] == 7
    assert stats["phase_telemetry"]["batches"] == 3
    rows = feature_rows(tmp_path / "inventory.sqlite")
    assert len(rows) == 7 and all(row["status"] == "done" for row in rows)


# --- thread discipline ------------------------------------------------------

def test_sqlite_is_only_ever_touched_by_the_main_thread(tmp_path, monkeypatch):
    """SQLite connections are not thread-safe; a worker writing would corrupt it.

    ``sqlite3.Connection.execute``/``executemany`` are immutable C attributes, so
    the connection factory is wrapped instead: every statement the stage issues
    goes through this recorder.
    """
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    threads: set[int] = set()
    original_open = db.open_db

    class RecordingConnection(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            threads.add(threading.get_ident())
            return super().execute(*args, **kwargs)

        def executemany(self, *args, **kwargs):
            threads.add(threading.get_ident())
            return super().executemany(*args, **kwargs)

        def commit(self, *args, **kwargs):
            threads.add(threading.get_ident())
            return super().commit(*args, **kwargs)

    def recording_open(path):
        real = sqlite3.connect

        def factory(*args, **kwargs):
            kwargs["factory"] = RecordingConnection
            return real(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", factory)
        try:
            return original_open(path)
        finally:
            monkeypatch.setattr(sqlite3, "connect", real)

    monkeypatch.setattr(stage1_features.db, "open_db", recording_open)
    stats = stage1_features.run(config_path=run_config, backend_override="stub")

    assert stats["processed"] == 6
    assert threads == {threading.get_ident()}


def test_the_thumbnail_writer_is_only_used_by_the_main_thread(tmp_path, monkeypatch):
    """The thumbnailer accumulates rows for one DB statement; it is not shared."""
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    threads: set[int] = set()
    from src import thumbnails

    original = thumbnails.Thumbnailer.capture

    def tracking_capture(self, row, image):
        threads.add(threading.get_ident())
        return original(self, row, image)

    monkeypatch.setattr(thumbnails.Thumbnailer, "capture", tracking_capture)
    stats = stage1_features.run(config_path=run_config, backend_override="stub")

    assert stats["thumbnails"]["created"] == 6
    assert threads == {threading.get_ident()}


def test_the_face_detector_is_only_used_by_the_main_thread(tmp_path, monkeypatch):
    """YuNet carries a mutable input size, so cross-thread use would corrupt it."""
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    threads: set[int] = set()
    from src import stage1_backends

    original = stage1_backends.StubBackend.faces_prepared

    def tracking_faces(self, prepared, image=None):
        threads.add(threading.get_ident())
        return original(self, prepared, image)

    monkeypatch.setattr(stage1_backends.StubBackend, "faces_prepared", tracking_faces)
    stage1_features.run(config_path=run_config, backend_override="stub")

    assert threads == {threading.get_ident()}


def test_no_producer_threads_survive_the_run(tmp_path):
    root = library(tmp_path / "photos", 8)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)
    before = threading.active_count()

    stage1_features.run(config_path=run_config, backend_override="stub")

    assert threading.active_count() <= before
    assert not [thread for thread in threading.enumerate()
                if thread.name.startswith("stage1-")]


# --- configuration ----------------------------------------------------------

def test_defaults_are_conservative_and_windows_safe():
    settings = resolve_loop_settings(Config({"features": {}, "scan": {}}))
    assert settings.cpu_workers == default_cpu_workers() <= 4
    assert settings.prefetch_batches == DEFAULT_PREFETCH_BATCHES == 2
    assert settings.iqa_batch_size == 4
    assert settings.prefetch_enabled is True
    assert settings.inflight_batches == 3


def test_auto_cpu_workers_tracks_the_machine_and_is_capped():
    """``auto`` must adapt to small machines instead of hardcoding four threads."""
    for value in ("auto", "AUTO", " auto ", None):
        settings = resolve_loop_settings(Config({
            "features": {"cpu_workers": value}, "scan": {}}))
        assert settings.cpu_workers == default_cpu_workers()
    assert 1 <= default_cpu_workers() <= 4


def test_the_shipped_config_defaults_are_the_documented_ones():
    """config.yaml and the in-code defaults must not drift apart."""
    from src.config import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config(DEFAULT_CONFIG_PATH)
    settings = resolve_loop_settings(cfg)
    assert cfg.features.get("cpu_workers") == "auto"
    assert cfg.features.get("prefetch_batches") == 2
    assert cfg.features.get("iqa_batch_size") == 4
    assert settings.cpu_workers == default_cpu_workers()
    assert settings.prefetch_enabled is True


@pytest.mark.parametrize("key,value,message", [
    ("cpu_workers", -1, "cpu_workers must be at least 0"),
    ("cpu_workers", "four", "cpu_workers must be an integer"),
    ("cpu_workers", 2.5, "cpu_workers must be an integer"),
    ("prefetch_batches", -3, "prefetch_batches must be at least 0"),
    ("prefetch_batches", "lots", "prefetch_batches must be an integer"),
    ("iqa_batch_size", 0, "iqa_batch_size must be at least 1"),
    ("iqa_batch_size", -2, "iqa_batch_size must be at least 1"),
    ("iqa_batch_size", None, "iqa_batch_size must be an integer"),
    ("batch_size", 0, "batch_size must be at least 1"),
    ("max_inflight_megapixels", 0, "max_inflight_megapixels must be greater than zero"),
    ("max_inflight_megapixels", -5, "max_inflight_megapixels must be greater than zero"),
    ("max_inflight_megapixels", "big", "max_inflight_megapixels must be a number"),
])
def test_bad_values_fail_loudly_and_are_never_clamped(key, value, message):
    with pytest.raises(ValueError, match=message):
        resolve_loop_settings(Config({"features": {key: value}, "scan": {}}))


def test_booleans_are_rejected_rather_than_read_as_one(key="cpu_workers"):
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_loop_settings(Config({"features": {key: True}, "scan": {}}))


def test_a_bad_knob_stops_the_run_before_any_work(tmp_path):
    """Validation happens before the DB, the models, or a single read."""
    root = library(tmp_path / "photos", 3)
    good = config(tmp_path, root)
    stage0_inventory.run(config_path=good)
    bad = config(tmp_path, root, cpu_workers=-2)

    with pytest.raises(ValueError, match="cpu_workers must be at least 0"):
        stage1_features.run(config_path=bad, backend_override="stub")

    assert feature_rows(tmp_path / "inventory.sqlite") == []


def test_string_numbers_from_yaml_are_accepted():
    settings = resolve_loop_settings(Config({
        "features": {"cpu_workers": "3", "prefetch_batches": "1", "iqa_batch_size": "8"},
        "scan": {"commit_every": "50"}}))
    assert (settings.cpu_workers, settings.prefetch_batches) == (3, 1)
    assert settings.iqa_batch_size == 8 and settings.commit_every == 50


def test_cli_override_wins_over_config(tmp_path):
    root = library(tmp_path / "photos", 4)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    stats = stage1_features.run(config_path=run_config, backend_override="stub",
                                cpu_workers=0)

    assert stats["loop_settings"]["cpu_workers"] == 0
    assert stats["loop_settings"]["prefetch_enabled"] is False
    assert stats["processed"] == 4


def test_cli_exposes_the_cpu_workers_flag():
    import argparse
    from unittest import mock

    with mock.patch.object(stage1_features, "run") as runner:
        stage1_features.main(["--backend", "stub", "--cpu-workers", "0"])
    assert runner.call_args.kwargs["cpu_workers"] == 0
    assert isinstance(argparse.ArgumentParser(), argparse.ArgumentParser)


# --- telemetry honesty ------------------------------------------------------

def test_batching_counters_reach_the_run_stats(tmp_path):
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    stats = stage1_features.run(config_path=run_config, backend_override="stub")

    counters = stats["phase_telemetry"]["counters"]
    assert counters["embed_calls"] == 2                  # 6 images, batch_size 3
    assert counters["embed_images"] == 6
    assert counters["prefetch_batches_delivered"] == 2
    assert counters["prefetch_images_prepared"] == 6


def test_producer_seconds_are_never_added_to_the_loop_total(tmp_path):
    """Concurrent work must not inflate the loop; the wait is what the loop paid."""
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    snapshot = stage1_features.run(
        config_path=run_config, backend_override="stub")["phase_telemetry"]

    assert snapshot["loop_accounted_seconds"] <= snapshot["loop_seconds"] + 1e-6
    assert snapshot["worker_seconds"] > 0
    loop_keys = {phase["key"] for phase in snapshot["phases"]
                 if phase["scope"] == "loop"}
    worker_keys = {phase["key"] for phase in snapshot["worker_phases"]}
    assert loop_keys.isdisjoint(worker_keys)
    # The whole outer wall clock still adds up, wait included.
    total = (snapshot["loop_seconds"] + snapshot["setup_accounted_seconds"]
             + snapshot["wait_accounted_seconds"]
             + snapshot["outer_unaccounted_seconds"])
    assert total == pytest.approx(snapshot["outer_seconds"], abs=1e-3)
    # The report says so in words, too.
    from src import telemetry_report

    text = "\n".join(telemetry_report.render_lines(snapshot))
    assert "producer-thread phases" in text
    assert "they are not added to it" in text
    assert "hidden behind main-thread work" in text
    assert "between-batch phases" in text
    assert "it is what prefetch cost" in text


def test_no_cuda_synchronisation_is_introduced_by_default(tmp_path):
    root = library(tmp_path / "photos", 4)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)

    snapshot = stage1_features.run(
        config_path=run_config, backend_override="stub")["phase_telemetry"]

    assert snapshot["gpu_event_sampling_every"] == 0
    assert snapshot["gpu_event_synchronisations"] == 0
    assert snapshot["gpu_event_sampled_batches"] == 0


def test_memory_bound_is_printed_for_the_operator(tmp_path, capsys):
    root = library(tmp_path / "photos", 3)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)
    stage1_features.run(config_path=run_config, backend_override="stub")

    console = capsys.readouterr().out
    assert "memory bound:" in console and "batch(es) in flight" in console
    assert "prefetch: 4 CPU worker(s)" in console
