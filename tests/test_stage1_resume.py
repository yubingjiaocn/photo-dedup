"""Resume and interruption: committed work survives, uncommitted work does not.

Stage 1 is the multi-hour stage, so Ctrl+C is a normal event rather than an
error, and its behaviour is part of the contract:

* a re-run must skip completed work entirely — no decode, no model call, no
  rewrite — whether the previous run was serial or prefetched;
* an interruption must release the database. An abandoned connection keeps its
  write lock, and the *next* run would then fail with ``database is locked``
  instead of resuming, which on a 100k-photo library looks like corruption;
* rows committed before the interruption must be durable and skipped; the batch
  in flight must roll back cleanly and be recomputed, never persist half-formed;
* no producer thread may outlive the run.
"""

from __future__ import annotations

import threading

import pytest
import yaml
from PIL import Image

from src import db, stage0_inventory, stage1_features


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


# --- resume -----------------------------------------------------------------

def test_resume_skips_completed_work_and_does_not_recompute(tmp_path):
    """A normal re-run must stay cheap: no decode, no model call, no rewrite."""
    root = library(tmp_path / "photos", 4)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)
    first = stage1_features.run(config_path=run_config, backend_override="stub")
    before = feature_rows(tmp_path / "inventory.sqlite")

    second = stage1_features.run(config_path=run_config, backend_override="stub")

    assert first["processed"] == 4
    assert second["processed"] == 0 and second["total"] == 0
    assert second["phase_telemetry"]["batches"] == 0
    assert second["phase_telemetry"]["counters"] == {}
    assert feature_rows(tmp_path / "inventory.sqlite") == before


def test_resume_after_partial_work_only_processes_the_remainder(tmp_path):
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    stage0_inventory.run(config_path=run_config)
    first = stage1_features.run(config_path=run_config, backend_override="stub", limit=4)
    done_ids = {row["file_id"] for row in feature_rows(tmp_path / "inventory.sqlite")}

    second = stage1_features.run(config_path=run_config, backend_override="stub")

    assert first["processed"] == 4 and second["processed"] == 2
    all_rows = feature_rows(tmp_path / "inventory.sqlite")
    assert len(all_rows) == 6
    assert done_ids.issubset({row["file_id"] for row in all_rows})
    assert all(row["status"] == "done" for row in all_rows)


def test_switching_between_serial_and_prefetch_resumes_cleanly(tmp_path):
    """Changing cpu_workers must not invalidate or duplicate completed work."""
    root = library(tmp_path / "photos", 6)
    serial_config = config(tmp_path, root, cpu_workers=0, prefetch_batches=0)
    stage0_inventory.run(config_path=serial_config)
    stage1_features.run(config_path=serial_config, backend_override="stub", limit=3)
    partial = feature_rows(tmp_path / "inventory.sqlite")

    fast_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2)
    resumed = stage1_features.run(config_path=fast_config, backend_override="stub")

    assert len(partial) == 3
    assert resumed["processed"] == 3
    rows = feature_rows(tmp_path / "inventory.sqlite")
    assert len(rows) == 6
    # The rows written by the serial run are untouched.
    for original in partial:
        current = next(row for row in rows if row["file_id"] == original["file_id"])
        assert current == original


def test_an_interrupted_run_releases_the_database_so_the_next_one_resumes(tmp_path,
                                                                         monkeypatch):
    """Ctrl+C must not leave the SQLite file locked by an abandoned connection.

    Without closing the connection on the way out, the interrupted process keeps
    its write lock and the *next* run dies with ``database is locked`` instead of
    resuming -- which on a multi-hour Stage 1 would look like corruption.

    ``commit_every=3`` matches one batch here, so the first batch is genuinely
    committed before the interrupt. That is what makes this a test of *resume*
    rather than of rollback: committed rows must survive and be skipped, while the
    uncommitted batch is correctly rolled back and recomputed.
    """
    root = library(tmp_path / "photos", 9)
    run_config = config(tmp_path, root, cpu_workers=4, prefetch_batches=2,
                        commit_every=3)
    stage0_inventory.run(config_path=run_config)

    original = stage1_features._consume_batch
    seen = {"batches": 0}

    def interrupting(*args, **kwargs):
        seen["batches"] += 1
        if seen["batches"] == 3:
            raise KeyboardInterrupt("user pressed Ctrl+C")
        return original(*args, **kwargs)

    monkeypatch.setattr(stage1_features, "_consume_batch", interrupting)
    with pytest.raises(KeyboardInterrupt):
        stage1_features.run(config_path=run_config, backend_override="stub")
    monkeypatch.undo()

    # No producer thread survived the interruption.
    assert not [thread for thread in threading.enumerate()
                if thread.name.startswith("stage1-")]
    # The committed batches are durable; the interrupted one is not yet written.
    after_interrupt = feature_rows(tmp_path / "inventory.sqlite")
    assert 0 < len(after_interrupt) < 9
    assert all(row["status"] == "done" for row in after_interrupt)

    resumed = stage1_features.run(config_path=run_config, backend_override="stub")

    assert resumed["processed"] == 9 - len(after_interrupt)
    rows = feature_rows(tmp_path / "inventory.sqlite")
    assert len(rows) == 9 and all(row["status"] == "done" for row in rows)
    # Nothing the first run committed was rewritten by the second.
    for original_row in after_interrupt:
        current = next(row for row in rows if row["file_id"] == original_row["file_id"])
        assert current == original_row


def test_an_interrupt_before_any_commit_rolls_back_rather_than_half_writing(tmp_path,
                                                                           monkeypatch):
    """Uncommitted rows must vanish, not persist half-formed."""
    root = library(tmp_path / "photos", 6)
    # commit_every well above the run size: nothing is committed before the abort.
    run_config = config(tmp_path, root, cpu_workers=2, prefetch_batches=1,
                        commit_every=1000)
    stage0_inventory.run(config_path=run_config)

    original = stage1_features._consume_batch
    seen = {"batches": 0}

    def interrupting(*args, **kwargs):
        seen["batches"] += 1
        if seen["batches"] == 2:
            raise KeyboardInterrupt("user pressed Ctrl+C")
        return original(*args, **kwargs)

    monkeypatch.setattr(stage1_features, "_consume_batch", interrupting)
    with pytest.raises(KeyboardInterrupt):
        stage1_features.run(config_path=run_config, backend_override="stub")
    monkeypatch.undo()

    assert feature_rows(tmp_path / "inventory.sqlite") == []
    resumed = stage1_features.run(config_path=run_config, backend_override="stub")
    assert resumed["processed"] == 6
    assert len(feature_rows(tmp_path / "inventory.sqlite")) == 6


def test_a_failing_batch_still_closes_the_database(tmp_path, monkeypatch):
    """The same guarantee for an ordinary exception, not just Ctrl+C."""
    root = library(tmp_path / "photos", 6)
    run_config = config(tmp_path, root, cpu_workers=2, prefetch_batches=1)
    stage0_inventory.run(config_path=run_config)

    monkeypatch.setattr(stage1_features, "_consume_batch",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            RuntimeError("consumer exploded")))
    with pytest.raises(RuntimeError, match="consumer exploded"):
        stage1_features.run(config_path=run_config, backend_override="stub")
    monkeypatch.undo()

    # The database is usable again: a fresh run does all the work.
    resumed = stage1_features.run(config_path=run_config, backend_override="stub")
    assert resumed["processed"] == 6
    assert len(feature_rows(tmp_path / "inventory.sqlite")) == 6
