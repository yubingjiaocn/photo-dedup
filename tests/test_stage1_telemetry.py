"""Stage 1 telemetry: correct accounting, honest labels, no observer effect.

The Windows symptom this answers: 1000 images, 461s feature loop, GPU pulsing
0->81%, 4.8 GB VRAM. Per-phase numbers are needed, and they must not lie about
what was measured (host wall time vs CUDA events), must not hide time in a
rounding gap or in un-timed setup, and must not perturb the thing they measure.

This file tests the collector in isolation. What a real Stage 1 run produces is
in ``test_stage1_telemetry_run.py``; the fakes are in
``stage1_telemetry_fixtures.py``.
"""

from __future__ import annotations

import time

from src import stage1_telemetry, telemetry_report
from tests import stage1_telemetry_fixtures as fixtures

# --- accounting ------------------------------------------------------------

def test_phase_seconds_and_calls_accumulate_per_instrumented_call():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(2):
        with telemetry.phase("decode", 2):
            time.sleep(0.01)
        with telemetry.phase("db_write"):
            time.sleep(0.005)

    snapshot = telemetry.snapshot()
    keys = {phase["key"]: phase for phase in snapshot["phases"]}
    assert snapshot["images_attempted"] == 2 and snapshot["batches"] == 1
    assert keys["decode"]["calls"] == 2 and keys["decode"]["unit"] == "img"
    assert keys["db_write"]["calls"] == 1 and keys["db_write"]["unit"] == "batch"
    assert keys["decode"]["seconds"] >= 0.009
    assert 0 < keys["decode"]["percent"] <= 100
    assert snapshot["loop_accounted_seconds"] <= snapshot["loop_seconds"] + 1e-6


def test_attempted_succeeded_and_failed_are_reported_separately():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(4) as batch:
        with telemetry.phase("error_handling"):
            time.sleep(0.002)
        batch.counted(succeeded=3, failed=1)

    snapshot = telemetry.snapshot()
    assert snapshot["images_attempted"] == 4
    assert snapshot["images_succeeded"] == 3
    assert snapshot["images_failed"] == 1
    keys = {phase["key"] for phase in snapshot["phases"]}
    assert "error_handling" in keys              # the failed read is instrumented
    line = fixtures.rendered_line(snapshot, "images attempted=")
    assert "attempted=4" in line and "succeeded=3" in line and "failed=1" in line


def test_unaccounted_in_loop_time_is_reported_not_absorbed():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(1):
        with telemetry.phase("decode"):
            time.sleep(0.005)
        time.sleep(0.02)                      # deliberately uninstrumented work

    snapshot = telemetry.snapshot()
    assert snapshot["loop_unaccounted_seconds"] >= 0.015
    assert snapshot["loop_unaccounted_percent"] > 50
    assert "unaccounted in-loop" in fixtures.rendered_line(snapshot, "unaccounted in-loop")


def test_setup_and_outer_wall_time_reconcile():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.phase("setup_model_load"):
        time.sleep(0.02)
    with telemetry.batch(1):
        with telemetry.phase("decode"):
            time.sleep(0.005)
    telemetry.set_outer_seconds(0.30)

    snapshot = telemetry.snapshot()
    assert snapshot["setup_accounted_seconds"] >= 0.019
    assert snapshot["outer_seconds"] == 0.3
    gap = snapshot["outer_unaccounted_seconds"]
    assert gap is not None and gap > 0
    total = (snapshot["loop_seconds"] + snapshot["setup_accounted_seconds"] + gap)
    assert abs(total - snapshot["outer_seconds"]) < 1e-3
    line = fixtures.rendered_line(snapshot, "reconciliation:")
    assert "Stage 1 outer" in line and "timed setup/finalize" in line
    assert "setup: model load" in "\n".join(telemetry_report.render_lines(snapshot))


def test_missing_outer_wall_time_is_stated_not_faked():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(1):
        pass
    snapshot = telemetry.snapshot()
    assert snapshot["outer_seconds"] is None
    assert snapshot["outer_unaccounted_seconds"] is None
    assert "outer wall time not supplied" in fixtures.rendered_line(snapshot, "reconciliation:")


def test_unknown_phase_keys_fold_into_other_cpu_instead_of_raising():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(1):
        with telemetry.phase("something_new"):
            pass
    keys = {phase["key"] for phase in telemetry.snapshot()["phases"]}
    assert "other_cpu" in keys and "something_new" not in keys


def test_batch_percentiles_are_reported_when_practical():
    telemetry = stage1_telemetry.Telemetry()
    for seconds in (0.01, 0.02, 0.03, 0.04):
        telemetry.record_batch(seconds, 1)
    percentiles = telemetry.snapshot()["batch_percentiles"]
    assert percentiles["p50"] is not None and percentiles["p95"] is not None
    assert percentiles["p50"] <= percentiles["p95"]


def test_empty_telemetry_reports_no_division_by_zero():
    snapshot = stage1_telemetry.Telemetry().snapshot()
    assert snapshot["images_attempted"] == 0
    assert snapshot["loop_unaccounted_percent"] is None
    assert snapshot["batch_percentiles"] == {"p50": None, "p95": None}
    assert "attempted=0" in fixtures.rendered_line(snapshot, "images attempted=")


def test_disabled_telemetry_costs_nothing_and_says_so():
    telemetry = stage1_telemetry.Telemetry(enabled=False)
    with telemetry.batch(5):
        with telemetry.phase("decode", 5):
            pass
    snapshot = telemetry.snapshot()
    assert snapshot["phases"] == [] and snapshot["images_attempted"] == 0
    assert telemetry_report.render_lines(snapshot) == [
        "Stage 1 phase breakdown: disabled (features.telemetry.enabled=false)"
    ]

# --- CUDA-event sampling ---------------------------------------------------

def test_cuda_event_sampling_is_off_by_default():
    assert stage1_telemetry.DEFAULT_GPU_EVENT_EVERY == 0
    fake = fixtures.FakeCuda()
    telemetry = stage1_telemetry.Telemetry(torch_module=fake)
    with telemetry.batch(4):
        for _ in range(4):
            with telemetry.gpu_event_phase("quality_musiq"):
                pass
    assert fake.records == 0 and fake.syncs == 0
    snapshot = telemetry.snapshot()
    assert snapshot["gpu_event_synchronisations"] == 0
    assert "disabled (features.telemetry.gpu_event_every=0)" in fixtures.rendered_line(
        snapshot, "CUDA event timing:")


def test_sampled_batch_synchronises_exactly_once_regardless_of_batch_size():
    """The old design synced 1+2N times per sampled batch; this must be 1."""
    fake = fixtures.FakeCuda()
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=1, torch_module=fake)
    with telemetry.batch(8):
        with telemetry.phase("embed_inference"):
            with telemetry.gpu_event_phase("embed_inference"):
                pass
        for _ in range(8):                      # per-image lane, 8 calls
            with telemetry.phase("quality_musiq"):
                with telemetry.gpu_event_phase("quality_musiq"):
                    pass
        assert fake.syncs == 0                  # nothing synced inside the batch

    assert fake.syncs == 1                      # one sync, after the batch clock
    snapshot = telemetry.snapshot()
    assert snapshot["gpu_event_synchronisations"] == 1
    assert snapshot["gpu_event_sampled_batches"] == 1
    samples = {phase["key"]: phase["gpu_event_samples"]
               for phase in snapshot["phases"] if phase["gpu_event_samples"]}
    # At most one sample per lane per sampled batch -> not 8 for MUSIQ.
    assert samples == {"embed_inference": 1, "quality_musiq": 1}


def test_the_synchronisation_happens_after_the_batch_clock_is_taken():
    """The measurement must not pay for its own instrument.

    A ``cuda.synchronize`` can take milliseconds. If it ran before the batch's
    wall clock was read, the sampled batches would look slower than the others
    and the reported host-wall totals would include the observer -- exactly the
    distortion the design claims to avoid. The fake sync is made deliberately
    expensive here, so the batch total proves the ordering.
    """
    sync_cost = 0.05
    fake = fixtures.FakeCuda(sync_seconds=sync_cost)
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=1, torch_module=fake)

    for _ in range(2):
        with telemetry.batch(1):
            with telemetry.phase("embed_inference"):
                with telemetry.gpu_event_phase("embed_inference"):
                    time.sleep(0.005)

    assert fake.syncs == 2                       # both batches were sampled
    snapshot = telemetry.snapshot()
    # Each batch did ~5 ms of work and paid a 50 ms sync; if the sync were inside
    # the batch clock, loop_seconds would be >= 0.1s instead of ~0.01s.
    assert snapshot["loop_seconds"] < 2 * sync_cost
    for value in snapshot["batch_percentiles"].values():
        assert value is not None and value < sync_cost
    embed = next(p for p in snapshot["phases"] if p["key"] == "embed_inference")
    assert embed["seconds"] < sync_cost          # the host-wall phase is clean too


def test_sampling_decision_is_latched_at_batch_entry():
    """batches increments at batch exit, so sampling must not flip mid-batch."""
    fake = fixtures.FakeCuda()
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=2, torch_module=fake)
    observed = []
    for _ in range(4):
        with telemetry.batch(2):
            observed.append(telemetry.sampling_active())
            mid = telemetry.sampling_active()
            with telemetry.phase("quality_musiq"):
                with telemetry.gpu_event_phase("quality_musiq"):
                    pass
            assert telemetry.sampling_active() is mid      # stable within a batch
    assert observed == [True, False, True, False]          # every 2nd batch
    assert fake.syncs == 2


def test_unsampled_batches_never_touch_cuda():
    fake = fixtures.FakeCuda()
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=100, torch_module=fake)
    with telemetry.batch(2):                     # batch 0 -> sampled
        with telemetry.gpu_event_phase("embed_inference"):
            pass
    baseline_records = fake.records
    for _ in range(5):                           # batches 1..5 -> not sampled
        with telemetry.batch(2):
            with telemetry.gpu_event_phase("embed_inference"):
                pass
    assert fake.records == baseline_records
    assert fake.syncs == 1


def test_gpu_numbers_are_labelled_as_samples_with_ms_per_sample():
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=1)
    with telemetry.batch(2):
        with telemetry.phase("embed_inference"):
            time.sleep(0.01)
    telemetry.add_gpu_event("embed_inference", 0.004)

    snapshot = telemetry.snapshot()
    embed = next(p for p in snapshot["phases"] if p["key"] == "embed_inference")
    assert embed["gpu_event_seconds"] == 0.004 and embed["gpu_event_samples"] == 1
    assert embed["seconds"] >= 0.009            # host wall is the larger figure
    text = "\n".join(telemetry_report.render_lines(snapshot))
    assert "samples=1" in text and "gpu=0.0040s" in text
    assert "ms per sampled batch" in text       # embed lane is batch-unit
    assert "at most one sample per lane per sampled batch" in text
    assert "host-wall (+CUDA below)" in text


def test_host_wall_versus_cuda_labelling_is_explicit():
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.batch(1):
        with telemetry.phase("embed_inference"):
            pass
    text = "\n".join(telemetry_report.render_lines(telemetry.snapshot()))
    assert "host-wall unless stated otherwise" in text
    assert "includes submit + wait" in text
    assert "upper bound on kernel time, not kernel time" in text


def test_missing_cuda_is_noted_without_failing():
    fake = fixtures.FakeCuda(available=False)
    telemetry = stage1_telemetry.Telemetry(gpu_event_every=4, torch_module=fake)
    with telemetry.batch(1):
        with telemetry.gpu_event_phase("embed_inference"):
            pass
    assert fake.records == 0 and fake.syncs == 0
    text = "\n".join(telemetry_report.render_lines(telemetry.snapshot()))
    assert "CUDA event timing unavailable (no CUDA)" in text


def test_config_section_controls_enablement_and_sampling():
    class _Features(dict):
        pass

    off = stage1_telemetry.build(_Features(telemetry={"enabled": False}))
    assert off.enabled is False
    tuned = stage1_telemetry.build(_Features(telemetry={"gpu_event_every": 3}))
    assert tuned.enabled is True and tuned.gpu_event_every == 3
    default = stage1_telemetry.build(_Features())
    assert default.gpu_event_every == 0        # opt-in, no observer effect

# --- units -----------------------------------------------------------------

def test_every_phase_declares_its_unit_and_scope():
    for phase in stage1_telemetry.PHASES:
        assert phase.unit in ("img", "batch", "step"), phase.key
        assert phase.scope in (stage1_telemetry.LOOP, stage1_telemetry.SETUP,
                               stage1_telemetry.WORKER, stage1_telemetry.WAIT), phase.key
        assert phase.kind in ("host", "gpu_event"), phase.key
    # Setup phases are one-off steps, never per image.
    setup = [p for p in stage1_telemetry.PHASES if p.scope == stage1_telemetry.SETUP]
    assert setup and all(p.unit == "step" for p in setup)
    # Producer-scope phases are per-image preparation: they must never claim to
    # be batch or setup units, because that is what would let concurrent worker
    # seconds be misread as main-thread batch time.
    worker = [p for p in stage1_telemetry.PHASES if p.scope == stage1_telemetry.WORKER]
    assert worker and all(p.unit == "img" for p in worker)
    assert all(p.kind == "host" for p in worker)
    # Between-batch waiting is per batch and is host-measured by definition.
    wait = [p for p in stage1_telemetry.PHASES if p.scope == stage1_telemetry.WAIT]
    assert wait and all(p.unit == "batch" and p.kind == "host" for p in wait)


def test_wait_scope_is_outside_the_batch_total_but_inside_the_outer_clock():
    """A batch is claimed before its span opens, so waiting is not batch time.

    If ``prefetch_wait`` were counted inside the loop, ``loop_accounted_seconds``
    could exceed ``loop_seconds`` and the reconciliation would read as if time
    had been invented. It belongs to the outer wall clock instead.
    """
    telemetry = stage1_telemetry.Telemetry()
    with telemetry.phase("prefetch_wait"):
        time.sleep(0.02)
    with telemetry.batch(1):
        with telemetry.phase("embed_inference"):
            time.sleep(0.005)
    telemetry.set_outer_seconds(0.30)

    snapshot = telemetry.snapshot()
    assert snapshot["wait_accounted_seconds"] >= 0.019
    assert snapshot["loop_accounted_seconds"] <= snapshot["loop_seconds"] + 1e-6
    assert snapshot["loop_seconds"] < 0.019          # the wait is not in the batch
    total = (snapshot["loop_seconds"] + snapshot["setup_accounted_seconds"]
             + snapshot["wait_accounted_seconds"]
             + snapshot["outer_unaccounted_seconds"])
    assert abs(total - snapshot["outer_seconds"]) < 1e-3
    text = "\n".join(telemetry_report.render_lines(snapshot))
    assert "between-batch phases" in text
    assert "prefetch wait" in fixtures.rendered_line(snapshot, "reconciliation:")
