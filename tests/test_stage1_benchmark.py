"""The Stage 1 benchmark harness must prove equivalence, not assert a speedup.

Deliberately asymmetric assertions:

* **Asserted:** the two pipelines produce identical features, the model-call
  counters show batching, and neither run is a catastrophic regression.
* **Reported only:** wall-clock times. With the stub backend "inference" takes
  microseconds, so there is no GPU work for the producer to overlap and the
  threaded run can legitimately be slower. A CI assertion on speedup would be
  either flaky or false; the real gains are measured with the actual weights in
  ``docs/STAGE1_THROUGHPUT.md``.
"""

from __future__ import annotations

import json

from src import stage1_benchmark


def test_identical_features_between_serial_and_optimised(tmp_path):
    root = tmp_path / "photos"
    stage1_benchmark.build_library(root, 9)

    serial = stage1_benchmark.run_variant(
        tmp_path / "serial", root, cpu_workers=0, prefetch_batches=0,
        batch_size=4, iqa_batch_size=4)
    optimised = stage1_benchmark.run_variant(
        tmp_path / "fast", root, cpu_workers=4, prefetch_batches=2,
        batch_size=4, iqa_batch_size=4)

    result = stage1_benchmark.compare(serial, optimised)
    assert result["identical_features"] is True
    assert result["mismatched_file_ids"] == []
    assert result["processed"] == (9, 9)
    assert serial["failed"] == optimised["failed"] == 0
    assert serial["batches"] == optimised["batches"] == 3


def test_the_synthetic_library_is_deterministic(tmp_path):
    """Same bytes on every machine, so results are comparable across runs."""
    first = stage1_benchmark.build_library(tmp_path / "a", 4)
    second = stage1_benchmark.build_library(tmp_path / "b", 4)
    assert [path.name for path in first] == [path.name for path in second]
    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes()


def test_the_library_spans_several_iqa_shape_groups(tmp_path):
    """A single-shape library would not exercise the interesting batching path."""
    from PIL import Image

    paths = stage1_benchmark.build_library(tmp_path / "photos", 6)
    sizes = set()
    for path in paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    assert len(sizes) >= 3


def test_batching_counters_are_reported_for_both_variants(tmp_path):
    root = tmp_path / "photos"
    stage1_benchmark.build_library(root, 8)

    serial = stage1_benchmark.run_variant(
        tmp_path / "serial", root, cpu_workers=0, prefetch_batches=0,
        batch_size=4, iqa_batch_size=4)
    optimised = stage1_benchmark.run_variant(
        tmp_path / "fast", root, cpu_workers=2, prefetch_batches=2,
        batch_size=4, iqa_batch_size=4)

    # Two batches of four: the embedding lane is called twice, not eight times.
    for variant in (serial, optimised):
        assert variant["counters"]["embed_calls"] == 2
        assert variant["counters"]["embed_images"] == 8
    assert optimised["counters"]["prefetch_batches_delivered"] == 2
    assert optimised["counters"]["prefetch_images_prepared"] == 8
    assert "prefetch_batches_delivered" not in serial["counters"]


def test_the_optimised_run_reports_overlap_and_the_serial_one_does_not(tmp_path):
    root = tmp_path / "photos"
    stage1_benchmark.build_library(root, 8)

    serial = stage1_benchmark.run_variant(
        tmp_path / "serial", root, cpu_workers=0, prefetch_batches=0,
        batch_size=4, iqa_batch_size=4)
    optimised = stage1_benchmark.run_variant(
        tmp_path / "fast", root, cpu_workers=4, prefetch_batches=2,
        batch_size=4, iqa_batch_size=4)

    assert serial["worker_seconds"] == 0 and serial["overlap_seconds"] is None
    assert optimised["worker_seconds"] > 0
    assert optimised["overlap_seconds"] is not None
    assert serial["loop_settings"]["prefetch_enabled"] is False
    assert optimised["loop_settings"]["prefetch_enabled"] is True


def test_timing_is_reported_but_only_gross_regression_is_asserted(tmp_path):
    """No flaky speedup assertion; only a sanity ceiling on the stub backend."""
    root = tmp_path / "photos"
    stage1_benchmark.build_library(root, 12)

    serial = stage1_benchmark.run_variant(
        tmp_path / "serial", root, cpu_workers=0, prefetch_batches=0,
        batch_size=4, iqa_batch_size=4)
    optimised = stage1_benchmark.run_variant(
        tmp_path / "fast", root, cpu_workers=4, prefetch_batches=2,
        batch_size=4, iqa_batch_size=4)

    result = stage1_benchmark.compare(serial, optimised)
    assert result["serial_wall_seconds"] > 0
    assert result["optimised_wall_seconds"] > 0
    assert result["wall_speedup"] is not None
    # The stub does microseconds of "inference", so prefetch has nothing to hide
    # behind: no speedup is promised. Thread overhead must stay bounded, though.
    assert result["optimised_wall_seconds"] < serial["stage1_wall_seconds"] * 3 + 5.0
    assert "indicative only" in result["note"]


def test_the_caveat_describes_the_backend_that_actually_ran():
    """A torch run must not be captioned with the stub's disclaimer."""
    fingerprint = [{"file_id": 1, "quality_score": 1.0}]
    left = {"fingerprint": fingerprint, "processed": 1, "counters": {},
            "stage1_wall_seconds": 2.0}
    right = {"fingerprint": fingerprint, "processed": 1, "counters": {},
             "stage1_wall_seconds": 1.0}

    stub = stage1_benchmark.compare(left, right)
    torch = stage1_benchmark.compare(left, right, backend="torch")

    assert stub["backend"] == "stub" and "indicative only" in stub["note"]
    assert torch["backend"] == "torch"
    assert "real model stack" in torch["note"]
    assert "indicative only" not in torch["note"]
    assert "stub" not in torch["note"]


def test_cli_runs_end_to_end_and_can_write_json(tmp_path, capsys):
    output = tmp_path / "benchmark.json"
    exit_code = stage1_benchmark.main([
        "--images", "6", "--batch-size", "3", "--cpu-workers", "2",
        "--prefetch-batches", "1", "--workdir", str(tmp_path / "work"),
        "--json", str(output),
    ])
    console = capsys.readouterr().out
    assert exit_code == 0
    assert "identical features       : True" in console
    assert "note:" in console
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["identical_features"] is True
    assert payload["processed"] == [6, 6]
    assert payload["optimised_counters"]["embed_calls"] == 2


def test_a_mismatch_is_reported_and_fails_the_comparison():
    """The harness must not paper over a difference it was built to catch."""
    serial = {"fingerprint": [{"file_id": 1, "quality_score": 1.0},
                              {"file_id": 2, "quality_score": 2.0}],
              "processed": 2, "counters": {}, "stage1_wall_seconds": 1.0}
    optimised = {"fingerprint": [{"file_id": 1, "quality_score": 1.0},
                                 {"file_id": 2, "quality_score": 2.5}],
                 "processed": 2, "counters": {}, "stage1_wall_seconds": 0.5}

    result = stage1_benchmark.compare(serial, optimised)

    assert result["identical_features"] is False
    assert result["mismatched_file_ids"] == [2]
