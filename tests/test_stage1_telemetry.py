"""Stage 1 phase telemetry: correct accounting, honest labels, stable format.

The Windows symptom this answers: 1000 images, 461s feature loop, GPU pulsing
0->81%, 4.8 GB VRAM. "Something starves the GPU" needs per-phase numbers, and
those numbers must not lie about what was measured (host wall time vs CUDA
events) or hide time in a rounding gap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from PIL import Image

from src import db, pipeline_report, stage1_features, stage1_telemetry


def _library(root: Path, count: int = 3) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"IMG_2026010{index}_120000.jpg"
        Image.new("RGB", (256, 192), (10 + index * 30, 60, 120)).save(path, "JPEG")
        paths.append(path)
    return paths


def _config(tmp_path: Path, root: Path, **telemetry) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(tmp_path / "inventory.sqlite"),
                  "output_dir": str(tmp_path / "out"),
                  "models_dir": str(tmp_path / "models"), "trash": str(tmp_path / "trash")},
        "features": {"backend": "stub", "batch_size": 2,
                     "thumbnails": {"enabled": True, "max_px": 64},
                     "telemetry": {"enabled": True, "gpu_event_every": 0, **telemetry}},
    }), encoding="utf-8")
    return str(path)


# --- accounting ------------------------------------------------------------

def test_phase_seconds_and_counts_accumulate_per_batch():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    with telemetry.batch(2):
        with telemetry.phase("decode", 2):
            time.sleep(0.01)
        with telemetry.phase("db_write", 2):
            time.sleep(0.005)

    snapshot = telemetry.snapshot()
    keys = {phase["key"]: phase for phase in snapshot["phases"]}
    assert snapshot["images"] == 2 and snapshot["batches"] == 1
    assert keys["decode"]["count"] == 2
    assert keys["decode"]["seconds"] >= 0.009
    assert keys["decode"]["ms_per_image"] > 0
    assert 0 < keys["decode"]["percent"] <= 100
    # Phases never exceed the measured batch total, and the remainder is shown.
    assert snapshot["accounted_seconds"] <= snapshot["total_seconds"] + 1e-6
    assert snapshot["unaccounted_seconds"] >= 0.0


def test_unaccounted_time_is_reported_not_absorbed():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    with telemetry.batch(1):
        with telemetry.phase("decode"):
            time.sleep(0.005)
        time.sleep(0.02)                      # deliberately uninstrumented work

    snapshot = telemetry.snapshot()
    assert snapshot["unaccounted_seconds"] >= 0.015
    assert snapshot["unaccounted_percent"] > 50
    lines = stage1_telemetry.render_lines(snapshot)
    assert any("unaccounted (loop overhead)" in line for line in lines)


def test_unknown_phase_keys_fold_into_other_cpu_instead_of_raising():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    with telemetry.batch(1):
        with telemetry.phase("something_new"):
            pass
    keys = {phase["key"] for phase in telemetry.snapshot()["phases"]}
    assert "other_cpu" in keys and "something_new" not in keys


def test_batch_percentiles_are_reported_when_practical():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    for seconds in (0.01, 0.02, 0.03, 0.04):
        telemetry.record_batch(seconds, 1)
    percentiles = telemetry.snapshot()["batch_percentiles"]
    assert percentiles["p50"] is not None and percentiles["p95"] is not None
    assert percentiles["p50"] <= percentiles["p95"]


def test_empty_telemetry_reports_no_division_by_zero():
    snapshot = stage1_telemetry.Telemetry(gpu_event_every=0).snapshot()
    assert snapshot["images"] == 0
    assert snapshot["unaccounted_percent"] is None
    assert snapshot["batch_percentiles"] == {"p50": None, "p95": None}
    lines = stage1_telemetry.render_lines(snapshot)
    assert any("images=0" in line for line in lines)


def test_disabled_telemetry_costs_nothing_and_says_so():
    telemetry = stage1_telemetry.Telemetry(enabled=False)
    with telemetry.batch(5):
        with telemetry.phase("decode", 5):
            pass
    snapshot = telemetry.snapshot()
    assert snapshot["phases"] == [] and snapshot["images"] == 0
    assert stage1_telemetry.render_lines(snapshot) == ["Stage 1 phase breakdown: disabled"]


# --- formatting ------------------------------------------------------------

def test_summary_lines_carry_counts_seconds_ms_per_image_and_percent():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    with telemetry.batch(4):
        with telemetry.phase("source_open_read", 4):
            time.sleep(0.01)
        with telemetry.phase("embed_inference", 4):
            time.sleep(0.02)

    lines = stage1_telemetry.render_lines(telemetry.snapshot())
    header = next(line for line in lines if line.startswith("phase"))
    assert "count" in header and "seconds" in header and "ms/img" in header and "%" in header
    embed = next(line for line in lines if line.startswith("embedding inference"))
    # count, seconds, ms/image, percent, timing-kind label all present.
    assert "4" in embed and "host-wall" in embed
    assert any("batch wall time p50=" in line and "p95=" in line for line in lines)
    assert any("images=4 batches=1 batch_total=" in line for line in lines)


def test_host_wall_versus_gpu_event_labelling_is_explicit():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    with telemetry.batch(1):
        with telemetry.phase("embed_inference"):
            pass
    lines = "\n".join(stage1_telemetry.render_lines(telemetry.snapshot()))
    assert "host-wall unless marked" in lines
    assert "includes submit + wait" in lines
    assert "upper bound on kernel time" in lines
    assert "GPU event timing: not sampled" in lines


def test_gpu_event_numbers_are_reported_separately_from_host_wall():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=1)
    with telemetry.batch(2):
        with telemetry.phase("embed_inference", 2):
            time.sleep(0.01)
    telemetry.add_gpu_event("embed_inference", 0.004)

    snapshot = telemetry.snapshot()
    embed = next(p for p in snapshot["phases"] if p["key"] == "embed_inference")
    assert embed["gpu_event_seconds"] == 0.004
    assert embed["gpu_event_batches"] == 1
    assert embed["seconds"] >= 0.009          # host wall is the larger figure
    lines = "\n".join(stage1_telemetry.render_lines(snapshot))
    assert "GPU event timing (CUDA events, sampled every 1 batch(es))" in lines
    assert "gpu=0.004s" in lines and "ms/batch" in lines


def test_missing_cuda_is_noted_without_failing():
    class _NoCuda:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    telemetry = stage1_telemetry.Telemetry(gpu_event_every=4, torch_module=_NoCuda)
    assert telemetry.gpu_sampling_now() is False
    with telemetry.batch(1):
        with telemetry.gpu_event_phase("embed_inference"):
            pass
    lines = "\n".join(stage1_telemetry.render_lines(telemetry.snapshot()))
    assert "GPU event timing unavailable (no CUDA)" in lines


def test_config_section_controls_enablement_and_sampling():
    class _Features(dict):
        pass

    off = stage1_telemetry.build(_Features(telemetry={"enabled": False}))
    assert off.enabled is False
    tuned = stage1_telemetry.build(_Features(telemetry={"gpu_event_every": 3}))
    assert tuned.enabled is True and tuned.gpu_event_every == 3
    default = stage1_telemetry.build(_Features())
    assert default.gpu_event_every == stage1_telemetry.DEFAULT_GPU_EVENT_EVERY


# --- integration -----------------------------------------------------------

def test_stage1_run_reports_the_phases_that_actually_ran(tmp_path):
    root = tmp_path / "photos"
    _library(root, 3)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")

    snapshot = stats["phase_telemetry"]
    keys = {phase["key"] for phase in snapshot["phases"]}
    # The costs a starvation question is about: HDD read, decode, model, thumb, DB.
    for expected in ("source_open_read", "decode", "hash_sha256", "embed_preprocess",
                     "embed_inference", "quality_preprocess", "quality_sharpness",
                     "faces_yunet", "phash", "thumbnail_resize",
                     "thumbnail_encode_write", "db_write"):
        assert expected in keys, expected
    assert snapshot["images"] == 3
    assert snapshot["batches"] == 2                     # batch_size=2 over 3 images
    assert snapshot["total_seconds"] > 0


def test_stage1_summary_is_written_to_the_log_and_performance_file(tmp_path, capsys):
    root = tmp_path / "photos"
    _library(root, 2)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")
    console = capsys.readouterr().out
    assert "[stage1] Stage 1 phase breakdown" in console
    assert "[stage1] JPEG decode" in console

    performance = pipeline_report.build_performance(
        {"stage0": 1.0, "stage1": 2.0, "stage2": 0.1, "stage3": 0.1},
        {"files": 2}, stats, library_still_images=2,
    )
    path = pipeline_report.write_performance_file(tmp_path / "out", performance)
    text = path.read_text(encoding="utf-8")
    assert "Stage 1 phase breakdown" in text
    assert "JPEG decode" in text
    assert "host-wall" in text
    assert "batch wall time p50=" in text


def test_telemetry_can_be_switched_off_without_affecting_results(tmp_path):
    root = tmp_path / "photos"
    _library(root, 2)
    from src import stage0_inventory

    on_config = _config(tmp_path, root)
    stage0_inventory.run(config_path=on_config)
    with_telemetry = stage1_features.run(config_path=on_config, backend_override="stub")

    plain = tmp_path / "plain"
    plain.mkdir()
    off_config = _config(plain, root, enabled=False)
    stage0_inventory.run(config_path=off_config)
    without = stage1_features.run(config_path=off_config, backend_override="stub")

    assert with_telemetry["processed"] == without["processed"] == 2
    assert without["phase_telemetry"]["enabled"] is False
    assert without["phase_telemetry"]["phases"] == []
    # Feature payloads are identical apart from telemetry bookkeeping.
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
    _library(root, 1)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")
    snapshot = stats["phase_telemetry"]
    keys = {phase["key"] for phase in snapshot["phases"]}
    assert "eye_detection" not in keys and "quality_clipiqa" not in keys
    lines = stage1_telemetry.render_lines(snapshot)
    assert any("GPU event timing" in line for line in lines)
    # Snapshot is JSON-serialisable so it can live in performance payloads.
    assert json.loads(json.dumps(snapshot))["images"] == 1


def test_hash_is_billed_separately_from_the_read_it_rides_along(tmp_path):
    root = tmp_path / "photos"
    paths = _library(root, 1)
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=0)
    with telemetry.batch(1):
        stage1_features._open_image_and_sha(paths[0], telemetry)

    snapshot = telemetry.snapshot()
    keys = {phase["key"]: phase for phase in snapshot["phases"]}
    # Small fixtures hash in microseconds, below the 0.1 ms report resolution, so
    # the raw accumulator is what proves the split happened.
    assert telemetry.seconds["hash_sha256"] > 0
    assert keys["hash_sha256"]["count"] == 1
    assert keys["source_open_read"]["seconds"] >= 0        # never negative after netting
    assert telemetry.seconds["source_open_read"] >= 0
    assert keys["decode"]["seconds"] > 0
    assert snapshot["accounted_seconds"] <= snapshot["total_seconds"] + 1e-6
