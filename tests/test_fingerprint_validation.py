"""Tests for group fingerprint validation to prevent stage2 rerun misapplication."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src import db, review_state, thumbnails


def _build_test_output(tmp_path: Path, member_pairs: list[tuple[int, int]]) -> tuple[Path, Path]:
    """Build output with groups. member_pairs: [(keeper_file_id, candidate_file_id)]."""
    output = tmp_path / "output"
    photos = tmp_path / "photos"
    output.mkdir(parents=True)
    photos.mkdir(parents=True)
    thumbs = thumbnails.thumbs_dir(output)
    thumbs.mkdir(parents=True, exist_ok=True)
    conn = db.open_db(output / "inventory.sqlite")

    # Create files
    for idx in range(max(max(p) for p in member_pairs) + 1):
        source = photos / f"IMG_{idx:04d}.jpg"
        Image.new("RGB", (3, 2), (idx % 255, 20, 30)).save(source, "JPEG")
        stat = source.stat()
        db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "exif_datetime": f"2026-01-01 12:00:{idx:02d}",
            "exif_timestamp": 1000 + idx, "width": 4000, "height": 3000,
            "file_kind": "jpg", "scan_status": "done",
        })
        db.batch_insert_features(conn, [{
            "file_id": idx + 1, "phash": None, "dinov2_embedding": None,
            "quality_score": 50.0 + idx, "quality_meta": "{}", "face_count": 0,
            "faces_json": "[]", "status": "done",
        }])
        Image.new("RGB", (32, 24), (idx % 255, 40, 90)).save(
            thumbnails.thumb_path(thumbs, idx + 1), "JPEG")
        db.batch_upsert_thumbnails(conn, [{
            "file_id": idx + 1, "status": "ok", "max_px": 320, "bytes": 900,
            "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "error": None, "created_at": 1,
        }])

    # Create groups from member_pairs
    for idx, (keeper, candidate) in enumerate(member_pairs):
        db.insert_group(conn, "burst", keeper,
                       [(keeper, True, "keep"), (candidate, False, "dup")], 1)
        db.update_member_decisions(conn, idx + 1, [
            {"file_id": keeper, "decision": "KEEP", "confidence": 1.0,
             "reason": "best", "evidence_json": "{}"},
            {"file_id": candidate, "decision": "MAYBE", "confidence": 0.5,
             "reason": "near duplicate", "evidence_json": "{}"},
        ])

    db.build_all_view_index(conn)
    conn.commit()
    conn.close()

    (output / "review.html").write_text('<h1>Review</h1>', encoding="utf-8")
    (output / "review_summary.json").write_text(json.dumps({"views": {}}), encoding="utf-8")
    return output, photos


def test_fingerprint_mismatch_ignored(tmp_path):
    """Decision with mismatched fingerprint is ignored."""
    # Create group 1 with members (1, 2)
    output, _photos = _build_test_output(tmp_path, [(1, 2)])

    # Manually write state with wrong fingerprint
    state_path = output / "review_state.json"
    wrong_fp = "0000000000000000"  # definitely not the real fingerprint
    bad_state = {
        "version": 1,
        "groups": {
            "1": {
                "action": "accept",
                "timestamp": 1234567890,
                "member_fingerprint": wrong_fp,
            },
        },
    }
    state_path.write_text(json.dumps(bad_state), encoding="utf-8")

    # Start server, which validates fingerprints
    conn = db.open_db(output / "inventory.sqlite")
    scope = None
    state = review_state.ReviewState(output)
    stale_count = state.validate_and_prune_stale(conn, scope)
    conn.close()

    # Decision should be pruned
    assert stale_count == 1
    assert state.get_group(1) is None


def test_old_state_without_fingerprint_ignored(tmp_path):
    """Old state without fingerprint field is fail-closed ignored."""
    output, _photos = _build_test_output(tmp_path, [(1, 2)])

    # Write state without fingerprint (legacy format)
    state_path = output / "review_state.json"
    old_state = {
        "version": 1,
        "groups": {
            "1": {
                "action": "accept",
                "timestamp": 1234567890,
                # no member_fingerprint field
            },
        },
    }
    state_path.write_text(json.dumps(old_state), encoding="utf-8")

    conn = db.open_db(output / "inventory.sqlite")
    scope = None
    state = review_state.ReviewState(output)
    stale_count = state.validate_and_prune_stale(conn, scope)
    conn.close()

    # Decision should be pruned (fail closed)
    assert stale_count == 1
    assert state.get_group(1) is None


def test_correct_fingerprint_preserved(tmp_path):
    """Decision with matching fingerprint is kept."""
    output, _photos = _build_test_output(tmp_path, [(1, 2)])

    # Compute correct fingerprint for group 1 members
    conn = db.open_db(output / "inventory.sqlite")
    member_file_ids = [1, 2]
    correct_fp = review_state.compute_member_fingerprint(member_file_ids)

    # Write state with correct fingerprint
    state_path = output / "review_state.json"
    good_state = {
        "version": 1,
        "groups": {
            "1": {
                "action": "accept",
                "timestamp": 1234567890,
                "member_fingerprint": correct_fp,
            },
        },
    }
    state_path.write_text(json.dumps(good_state), encoding="utf-8")

    scope = None
    state = review_state.ReviewState(output)
    stale_count = state.validate_and_prune_stale(conn, scope)
    conn.close()

    # Decision should be preserved
    assert stale_count == 0
    assert state.get_group(1) is not None
    assert state.get_group(1)["action"] == "accept"


def test_set_group_stores_fingerprint(tmp_path):
    """set_group with fingerprint argument stores it in state file."""
    output, _photos = _build_test_output(tmp_path, [(1, 2)])

    state = review_state.ReviewState(output)
    fp = "abcd1234"
    state.set_group(1, {"action": "accept", "timestamp": 1234567890}, member_fingerprint=fp)

    # Reload and check fingerprint is persisted
    state2 = review_state.ReviewState(output)
    decision = state2.get_group(1)
    assert decision is not None
    assert decision["member_fingerprint"] == fp
