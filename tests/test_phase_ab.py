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
    report = run("fixture", inventory, output)
    assert digest(inventory) == before
    assert report["sqlite_mode"] == "ro+immutable"
    assert report["opened_original_media"] is False
    assert report["physical_group_split"] is False
    assert report["metrics"]["groups"] == 1
    assert report["metrics"]["candidate_keepers"] < report["metrics"]["members"]
    assert (output / "fixture.json").is_file()
    assert (output / "fixture-groups.csv").is_file()
    assert (output / "fixture-review-queue.json").is_file()
    assert (output / "fixture-review.html").is_file()
