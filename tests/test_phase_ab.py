import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from scripts.phase_ab import run
from src.quality import embedding_to_blob


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_inventory(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE files (
          id INTEGER PRIMARY KEY, basename TEXT, size_bytes INTEGER, mtime_ns INTEGER,
          exif_timestamp INTEGER, width INTEGER, height INTEGER
        );
        CREATE TABLE features (
          file_id INTEGER PRIMARY KEY, phash BLOB, content_sha256 TEXT,
          dinov2_embedding BLOB, quality_score REAL, quality_meta TEXT,
          face_count INTEGER, faces_json TEXT, status TEXT
        );
        CREATE TABLE groups (
          id INTEGER PRIMARY KEY, group_type TEXT, keep_file_id INTEGER, member_count INTEGER
        );
        CREATE TABLE group_members (group_id INTEGER, file_id INTEGER, decision TEXT);
    """)
    vectors = ([1.0, 0.0], [0.99, 0.01], [0.8, 0.6])
    for fid, vector in enumerate(vectors, 1):
        conn.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (fid, f"{fid}.jpg", 100 + fid, fid, fid, 100, 100))
        conn.execute("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            fid, b"12345678", str(fid),
            embedding_to_blob(np.asarray(vector, dtype=np.float32)),
            50 + fid, json.dumps({"face_quality": 0.7, "exposure": {"clip_hi": 0, "clip_lo": 0}}),
            1, "[]", "done",
        ))
        conn.execute("INSERT INTO group_members VALUES (1, ?, ?)", (fid, "KEEP" if fid == 1 else "MAYBE"))
    conn.execute("INSERT INTO groups VALUES (1, 'burst', 1, 3)")
    conn.commit()
    conn.close()


def test_harness_is_source_immutable_and_writes_standalone_review_artifacts(tmp_path):
    inventory = tmp_path / "inventory.sqlite"
    output = tmp_path / "ab"
    make_inventory(inventory)
    before = digest(inventory)
    report = run("fixture", inventory, output, source_sha="test-sha")
    assert digest(inventory) == before
    assert report["sqlite_mode"] == "ro+immutable"
    assert report["opened_original_media"] is False
    assert report["physical_group_split"] is False
    assert report["source_sha"] == "test-sha"
    assert report["metrics"]["groups"] == 1
    assert report["metrics"]["candidate_keepers"] < report["metrics"]["members"]
    assert (output / "fixture.json").is_file()
    assert (output / "fixture-groups.csv").is_file()
    assert (output / "fixture-review-queue.json").is_file()
    assert (output / "fixture-secondary-diagnostics.json").is_file()
    assert (output / "fixture-review.html").is_file()


def test_review_queue_separates_primary_change_from_secondary_evidence_gap(tmp_path):
    inventory = tmp_path / "inventory.sqlite"
    output = tmp_path / "ab"
    make_inventory(inventory)
    report = run("fixture", inventory, output)
    row = report["groups"][0]
    assert row["changed"] is True
    assert row["review_tier"] == "primary"
    primary = json.loads((output / "fixture-review-queue.json").read_text())
    secondary = json.loads((output / "fixture-secondary-diagnostics.json").read_text())
    assert primary["count"] == 1
    assert secondary["count"] == 0

    # Turn the fixture into an unchanged ordinary pair. Optional close-score
    # face evidence still merits diagnostics, but must not enter primary.
    conn = sqlite3.connect(inventory)
    conn.execute("UPDATE groups SET keep_file_id=2, member_count=2 WHERE id=1")
    conn.execute("DELETE FROM group_members WHERE file_id=3")
    conn.execute("DELETE FROM features WHERE file_id=3")
    conn.execute("DELETE FROM files WHERE id=3")
    conn.commit()
    conn.close()
    output2 = tmp_path / "ab-secondary"
    report2 = run("fixture", inventory, output2)
    row2 = report2["groups"][0]
    assert row2["changed"] is False
    assert row2["review_tier"] == "secondary"
    assert json.loads((output2 / "fixture-review-queue.json").read_text())["count"] == 0
    assert json.loads(
        (output2 / "fixture-secondary-diagnostics.json").read_text()
    )["count"] == 1


def test_mandatory_unchanged_group_is_not_counted_in_both_review_tiers(tmp_path, monkeypatch):
    from scripts import phase_ab
    inventory = tmp_path / "synthetic.sqlite"
    make_inventory(inventory)
    monkeypatch.setattr(phase_ab.PS, "select_phase_keepers", lambda members, phases: {
        "keepers": [0], "phases": [{"phase_id": 0, "members": [0, 1, 2], "keepers": [0]}],
        "review_required": True, "mandatory_review": True,
        "group_keeper_budget": 2, "phase_coverage_diagnostics": [],
    })
    report = run("synthetic", inventory, tmp_path / "report")
    assert report["groups"][0]["changed"] is False
    assert report["groups"][0]["review_tier"] == "primary"
    assert report["metrics"]["ab_review_groups"] == 1
    assert report["metrics"].get("secondary_review_groups", 0) == 0
