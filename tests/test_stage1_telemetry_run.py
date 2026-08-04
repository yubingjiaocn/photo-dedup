"""Stage 1 telemetry, end to end: does a real run report what it did?

The collector's own arithmetic is covered by ``test_stage1_telemetry.py``. Here a
Stage 1 run is executed for real (stub backend) and asked to account for itself:
every phase that ran appears, setup and model load are timed separately from the
loop, unreadable files are counted as failures instead of vanishing, the numbers
reconcile against Stage 1's outer wall time, and the same breakdown reaches both
the log and ``performance.txt``.
"""

from __future__ import annotations

import json

from src import db, pipeline_report, stage0_inventory, stage1_features, stage1_telemetry
from tests import stage1_telemetry_fixtures as fixtures

# --- integration -----------------------------------------------------------

def test_stage1_run_reports_the_phases_that_actually_ran(tmp_path):
    root = tmp_path / "photos"
    fixtures.library(root, 3)
    config = fixtures.config(tmp_path, root)

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")

    snapshot = stats["phase_telemetry"]
    loop_keys = {phase["key"] for phase in snapshot["phases"]}
    worker_keys = {phase["key"] for phase in snapshot["worker_phases"]}
    # Setup and the main-thread lanes must be measured where they run.
    for expected in ("setup_admission", "setup_pending_query", "setup_thumb_listing",
                     "setup_model_load", "embed_inference", "faces_yunet",
                     "thumbnail_resize", "thumbnail_encode_write", "db_write",
                     "db_commit_final", "finalize_stats"):
        assert expected in loop_keys, expected
    # With the default cpu_workers>0 the read/decode/prepare lanes ran on producer
    # threads, so they are reported in the producer section -- not silently
    # dropped, and not added to the loop's books.
    for expected in ("source_open_read", "decode", "hash_sha256", "phash",
                     "embed_preprocess", "quality_preprocess", "quality_sharpness"):
        assert expected in worker_keys, expected
        assert expected not in loop_keys, expected
    assert snapshot["images_attempted"] == 3
    assert snapshot["images_succeeded"] == 3 and snapshot["images_failed"] == 0
    assert snapshot["batches"] == 2                     # batch_size=2 over 3 images
    assert snapshot["outer_seconds"] is not None
    assert snapshot["outer_unaccounted_seconds"] is not None
    by_key = {phase["key"]: phase for phase in snapshot["phases"]}
    by_worker = {phase["key"]: phase for phase in snapshot["worker_phases"]}
    assert by_key["db_write"]["calls"] == 2             # one per batch
    assert by_worker["decode"]["calls"] == 3            # one per image
    assert by_key["setup_model_load"]["unit"] == "step"
    # Producer seconds are reported against their own total, never the loop's.
    assert snapshot["worker_seconds"] > 0
    assert snapshot["overlap_seconds"] is not None
    assert stats["loop_settings"]["prefetch_enabled"] is True


def test_serial_mode_bills_preparation_to_the_loop_instead_of_producers(tmp_path):
    """cpu_workers=0 must be the old pipeline: same phases, measured in-loop."""
    root = tmp_path / "photos"
    fixtures.library(root, 3)
    config = fixtures.config(tmp_path, root)

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub", cpu_workers=0)

    snapshot = stats["phase_telemetry"]
    loop_keys = {phase["key"] for phase in snapshot["phases"]}
    for expected in ("source_open_read", "decode", "hash_sha256", "phash",
                     "embed_preprocess", "quality_preprocess", "quality_sharpness"):
        assert expected in loop_keys, expected
    assert snapshot["worker_phases"] == []
    assert snapshot["worker_seconds"] == 0
    assert snapshot["overlap_seconds"] is None
    assert snapshot["prefetch_wait_seconds"] == 0
    assert stats["loop_settings"]["prefetch_enabled"] is False
    assert stats["processed"] == 3 and stats["failed"] == 0


def test_stage1_counts_unreadable_sources_as_failed_and_times_them(tmp_path):
    root = tmp_path / "photos"
    fixtures.library(root, 2)
    (root / "IMG_20260109_120000.jpg").write_bytes(b"not a jpeg at all")
    config = fixtures.config(tmp_path, root)

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")

    snapshot = stats["phase_telemetry"]
    assert stats["failed"] == 1 and stats["succeeded"] == 2
    assert snapshot["images_attempted"] == 3
    assert snapshot["images_succeeded"] == 2
    assert snapshot["images_failed"] == 1
    by_key = {phase["key"]: phase for phase in snapshot["phases"]}
    by_worker = {phase["key"]: phase for phase in snapshot["worker_phases"]}
    # error_handling is main-thread work: the router, the thumbnail failure record
    # and the error row are all stateful, so they never run on a producer thread.
    assert by_key["error_handling"]["calls"] == 1
    # A phase counts *attempts*: the corrupt file's decode really did run (and
    # raised), so billing it to `decode` is honest. `error_handling` then covers
    # only the failure bookkeeping, and images_failed reports the outcome.
    assert by_worker["decode"]["calls"] == 3
    assert by_worker["phash"]["calls"] == 2            # only successful images


def test_stage1_summary_is_written_to_the_log_and_performance_file(tmp_path, capsys):
    root = tmp_path / "photos"
    fixtures.library(root, 2)
    config = fixtures.config(tmp_path, root)

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")
    console = capsys.readouterr().out
    assert "[stage1] Stage 1 phase breakdown" in console
    assert "[stage1] JPEG decode" in console
    assert "[stage1] reconciliation: Stage 1 outer" in console

    performance = pipeline_report.build_performance(
        {"stage0": 1.0, "stage1": 2.0, "stage2": 0.1, "stage3": 0.1},
        {"files": 2}, stats, library_still_images=2,
    )
    text = pipeline_report.write_performance_file(
        tmp_path / "out", performance).read_text(encoding="utf-8")
    assert "Stage 1 phase breakdown" in text
    assert "setup / finalize phases" in text
    assert "setup: model load" in text
    assert "reconciliation: Stage 1 outer" in text
    assert "host-wall" in text
    assert "batch wall time p50=" in text


def test_telemetry_can_be_switched_off_without_affecting_results(tmp_path):
    root = tmp_path / "photos"
    fixtures.library(root, 2)

    on_config = fixtures.config(tmp_path, root)
    stage0_inventory.run(config_path=on_config)
    with_telemetry = stage1_features.run(config_path=on_config, backend_override="stub")

    plain = tmp_path / "plain"
    plain.mkdir()
    off_config = fixtures.config(plain, root, enabled=False)
    stage0_inventory.run(config_path=off_config)
    without = stage1_features.run(config_path=off_config, backend_override="stub")

    assert with_telemetry["processed"] == without["processed"] == 2
    assert without["phase_telemetry"]["enabled"] is False
    assert without["phase_telemetry"]["phases"] == []
    for path in (tmp_path, plain):
        conn = db.open_db(path / "inventory.sqlite")
        rows = conn.execute(
            "SELECT content_sha256, quality_score FROM features ORDER BY file_id"
        ).fetchall()
        conn.close()
        assert len(rows) == 2 and all(row["content_sha256"] for row in rows)


def test_optional_components_absent_report_zero_instead_of_breaking(tmp_path):
    """No eye detector, no CLIP-IQA, no CUDA: the format still renders."""
    root = tmp_path / "photos"
    fixtures.library(root, 1)
    config = fixtures.config(tmp_path, root)

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")
    snapshot = stats["phase_telemetry"]
    keys = {phase["key"] for phase in snapshot["phases"]}
    assert "eye_detection" not in keys and "quality_clipiqa" not in keys
    assert "CUDA event timing:" in fixtures.rendered_line(snapshot, "CUDA event timing:")
    assert json.loads(json.dumps(snapshot))["images_attempted"] == 1


def test_hash_is_billed_separately_from_the_read_it_rides_along(tmp_path):
    root = tmp_path / "photos"
    paths = fixtures.library(root, 1)
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(1):
        stage1_features._open_image_and_sha(paths[0], telemetry)

    snapshot = telemetry.snapshot()
    keys = {phase["key"]: phase for phase in snapshot["phases"]}
    # Tiny fixtures hash in microseconds, below the reported resolution, so the
    # raw accumulator is what proves the split happened.
    assert telemetry.seconds["hash_sha256"] > 0
    assert keys["hash_sha256"]["calls"] == 1
    assert telemetry.seconds["source_open_read"] >= 0     # never negative
    assert keys["source_open_read"]["calls"] == 1
    assert keys["decode"]["seconds"] > 0
    assert snapshot["loop_accounted_seconds"] <= snapshot["loop_seconds"] + 1e-6
