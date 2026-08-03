"""Performance/ETA/disk reporting must be honest about what it measured."""

from __future__ import annotations

from pathlib import Path

from src import pipeline_report, thumbnails

GIB = 1024 ** 3


def _timings(**overrides):
    base = {"stage0": 1.0, "stage1": 4.0, "stage2": 0.5, "stage3": 0.5}
    base.update(overrides)
    return base


def test_stage_times_and_throughput_are_derived_from_the_run():
    performance = pipeline_report.build_performance(
        _timings(), {"files": 100}, {"processed": 80}, library_still_images=80
    )
    assert performance["stage_seconds"] == {"stage0": 1.0, "stage1": 4.0,
                                           "stage2": 0.5, "stage3": 0.5}
    assert performance["total_seconds"] == 6.0
    assert performance["stage0_files_per_second"] == 100.0
    assert performance["stage1_images_per_second"] == 20.0


def test_eta_scales_from_this_inventory_not_a_hardcoded_library_size():
    performance = pipeline_report.build_performance(
        _timings(), {"files": 100}, {"processed": 100}, library_still_images=100_000
    )
    # Stage 1 is the sampled per-image work: 4s/100 -> 4000s/100k.
    assert performance["eta_seconds"] == 4000.0
    assert abs(performance["eta_hours"] - 4000.0 / 3600.0) < 1e-9
    assert "100000 discovered still image(s)" in performance["eta_basis"]
    assert "TiB" not in performance["eta_basis"]
    assert "TiB" not in performance["eta_disclaimer"]


def test_eta_is_clearly_labelled_as_a_rough_linear_estimate():
    performance = pipeline_report.build_performance(
        _timings(), {"files": 10}, {"processed": 10}, library_still_images=1000
    )
    lines = "\n".join(pipeline_report.render_lines(performance))
    assert "ROUGH full-library Stage 1 ETA" in lines
    assert "ROUGH LINEAR ESTIMATE ONLY" in lines
    assert "rough Stage 1 scale-up" in lines


def test_eta_is_unknown_without_a_sample():
    performance = pipeline_report.build_performance(
        _timings(), {"files": 0}, {"processed": 0}, library_still_images=0
    )
    assert performance["eta_seconds"] is None
    assert performance["eta_hours"] is None
    assert "unknown" in performance["eta_basis"]
    assert "ROUGH full-library Stage 1 ETA: unknown" in "\n".join(
        pipeline_report.render_lines(performance))


def test_zero_duration_stages_report_unknown_rate_instead_of_dividing():
    performance = pipeline_report.build_performance(
        _timings(stage0=0.0, stage1=0.0), {"files": 5}, {"processed": 5}
    )
    assert performance["stage0_files_per_second"] is None
    assert performance["stage1_images_per_second"] is None
    assert "unknown files/s" in "\n".join(pipeline_report.render_lines(performance))


# --- disk accounting -------------------------------------------------------

def test_disk_report_measures_actual_usage_and_estimates_the_rest(tmp_path):
    directory = thumbnails.thumbs_dir(tmp_path)
    directory.mkdir(parents=True)
    for file_id in range(4):
        thumbnails.thumb_path(directory, file_id).write_bytes(b"x" * 30_000)

    report = pipeline_report.thumbnail_disk_report(tmp_path, 100_000)

    assert report["cached_files"] == 4
    assert report["actual_bytes"] == 120_000
    assert report["average_bytes"] == 30_000.0
    assert report["estimated_total_bytes"] == 3_000_000_000
    assert report["estimated_remaining_bytes"] == 30_000 * 99_996
    assert report["basis"] == "measured sample average"
    assert report["free_bytes"] is not None


def test_disk_report_says_unknown_when_no_sample_exists(tmp_path):
    report = pipeline_report.thumbnail_disk_report(tmp_path, 100_000)
    assert report["cached_files"] == 0
    assert report["actual_bytes"] == 0
    assert report["estimated_total_bytes"] is None
    assert report["average_bytes"] is None
    assert "unknown" in report["basis"]
    lines = "\n".join(pipeline_report.render_lines(
        pipeline_report.build_performance(_timings(), {}, {}), report))
    assert "estimated full-library usage: unknown" in lines


def test_low_space_only_warns_and_never_requires_fifty_gib(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_report, "free_bytes", lambda _path: 5 * GIB)
    report = pipeline_report.thumbnail_disk_report(tmp_path, 1000)
    assert report["low_space"] is True
    assert report["warn_free_gib"] == 20.0
    lines = "\n".join(pipeline_report.render_lines(
        pipeline_report.build_performance(_timings(), {"files": 1}, {"processed": 1}), report))
    assert "WARNING: less than 20 GiB free" in lines
    assert "advisory only" in lines
    assert "does not stop" in lines
    assert "50 GiB" in lines and "never" in lines  # explicitly denies a 50 GiB rule


def test_ample_space_produces_no_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_report, "free_bytes", lambda _path: 200 * GIB)
    report = pipeline_report.thumbnail_disk_report(tmp_path, 1000)
    assert report["low_space"] is False
    lines = "\n".join(pipeline_report.render_lines(
        pipeline_report.build_performance(_timings(), {"files": 1}, {"processed": 1}), report))
    assert "WARNING" not in lines


def test_free_bytes_falls_back_to_an_existing_parent(tmp_path):
    assert pipeline_report.free_bytes(tmp_path / "does" / "not" / "exist") is not None


def test_startup_plan_reports_estimate_actual_and_free(tmp_path, capsys):
    directory = thumbnails.thumbs_dir(tmp_path)
    directory.mkdir(parents=True)
    thumbnails.thumb_path(directory, 1).write_bytes(b"y" * 40_000)

    plan = pipeline_report.print_thumbnail_plan(tmp_path, 50_000)

    assert plan["existing_files"] == 1
    assert plan["estimated_total_bytes"] == 40_000 * 50_000
    output = capsys.readouterr().out
    assert "estimated thumbnail usage for 50000 still image(s)" in output
    assert "thumbnails already cached: 1" in output
    assert "SSD free:" in output


def test_startup_plan_without_a_sample_is_explicit(tmp_path, capsys):
    plan = pipeline_report.print_thumbnail_plan(tmp_path, 100_000)
    assert plan["estimated_total_bytes"] is None
    output = capsys.readouterr().out
    assert "unknown until the first thumbnails are written" in output
    assert "25-45 KiB" in output


def test_performance_file_is_written_with_both_sections(tmp_path):
    performance = pipeline_report.build_performance(
        _timings(), {"files": 10}, {"processed": 10}, library_still_images=10)
    report = pipeline_report.thumbnail_disk_report(tmp_path, 10)
    path = pipeline_report.write_performance_file(tmp_path, performance, report)
    assert path == Path(tmp_path) / "performance.txt"
    text = path.read_text(encoding="utf-8")
    assert "observed performance" in text
    assert "Thumbnail cache (SSD)" in text
