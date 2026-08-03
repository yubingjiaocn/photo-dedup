"""P0 exposure, selective decision, migration and manifest safety tests."""
import json

import numpy as np
from PIL import Image

from src import db, decision, exposure
from src.stage3_report import collect_deletions


def _exp_meta(arr, faces=()):
    metrics = exposure.extract(Image.fromarray(arr.astype(np.uint8)), faces)
    return json.dumps({"exposure": metrics, "face_quality": 0.0})


def _member(arr=None, *, score=50.0, phash=b"12345678", emb=b"x", faces=0):
    arr = np.full((128, 128, 3), 128, dtype=np.uint8) if arr is None else arr
    return {
        "quality_score": score, "quality_meta": _exp_meta(arr),
        "width": 128, "height": 128, "size_bytes": 100,
        "phash": phash, "dinov2_embedding": emb, "face_count": faces,
        "content_sha256": "same",
    }


def test_artistic_silhouette_not_rejected():
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, 48:] = 120  # broad normally exposed anchor survives
    state, _, _ = exposure.classify(exposure.extract(Image.fromarray(arr), []))
    assert state != "reject"


def test_local_highlight_not_rejected():
    arr = np.full((128, 128, 3), 110, dtype=np.uint8)
    arr[5:18, 5:18] = 255
    state, _, _ = exposure.classify(exposure.extract(Image.fromarray(arr), []))
    assert state == "ok"


def test_severe_global_exposure_failure():
    blown = np.full((128, 128, 3), 255, dtype=np.uint8)
    state, reason, _ = exposure.classify(exposure.extract(Image.fromarray(blown), []))
    assert state == "reject"
    assert "blown" in reason


def test_group_floor_and_low_margin_maybe():
    members = [_member(score=50), _member(score=49)]
    result = decision.decide_group(members, 0, {0: .50, 1: .49}, "burst")
    assert result["members"][0]["decision"] == "KEEP"
    assert result["members"][1]["decision"] == "MAYBE"
    assert result["members"][1]["reason"] == "LOW_MARGIN"


def test_missing_features_unknown():
    members = [_member(score=80), _member(score=20, emb=None)]
    result = decision.decide_group(members, 0, {0: .8, 1: .2}, "burst")
    assert result["members"][1]["decision"] == "UNKNOWN"


def test_exact_duplicate_auto_remove():
    members = [_member(score=80), _member(score=20)]
    result = decision.decide_group(
        members, 0, {0: .8, 1: .2}, "exact_dup", phash_distances={0: 0, 1: 0},
        safe_duplicates={0: True, 1: True},
    )
    assert result["members"][0]["decision"] == "KEEP"
    assert result["members"][1]["decision"] == "AUTO_REMOVE"


def test_phash_near_duplicate_never_auto_without_byte_identity():
    members = [_member(score=80), _member(score=20)]
    result = decision.decide_group(
        members, 0, {0: .8, 1: .2}, "exact_dup", phash_distances={0: 0, 1: 1},
        safe_duplicates={0: True, 1: False},
    )
    assert result["members"][1]["decision"] == "MAYBE"


def test_exposure_alone_never_auto_for_any_profile():
    normal = np.full((128, 128, 3), 128, dtype=np.uint8)
    scenes = [
        np.full((128, 128, 3), 255, dtype=np.uint8),  # snow/white product
        np.zeros((128, 128, 3), dtype=np.uint8),     # night/sky/fireworks
    ]
    for profile in decision.PROFILES:
        for extreme in scenes:
            members = [_member(normal, score=80), _member(extreme, score=10)]
            result = decision.decide_group(members, 0, {0: .8, 1: .1}, "burst", profile=profile)
            assert result["members"][1]["decision"] != "AUTO_REMOVE"


def _file(path):
    return {"path": path, "basename": path.rsplit("/", 1)[-1], "size_bytes": 10,
            "mtime_ns": 1, "exif_timestamp": 1, "width": 10, "height": 10,
            "file_kind": "jpg", "scan_status": "done"}


def test_manifest_excludes_maybe_and_unknown(tmp_path):
    conn = db.open_db(tmp_path / "db.sqlite")
    ids = [db.insert_file(conn, _file(f"/x/{x}.jpg")) for x in "abc"]
    gid = db.insert_group(conn, "burst", ids[0],
                          [(ids[0], True, "keep"), (ids[1], False, "maybe"),
                           (ids[2], False, "unknown")], 1)
    db.update_member_decisions(conn, gid, [
        {"file_id": ids[0], "decision": "KEEP", "confidence": 1., "reason": "keep", "evidence_json": "{}"},
        {"file_id": ids[1], "decision": "MAYBE", "confidence": 0., "reason": "LOW_MARGIN", "evidence_json": "{}"},
        {"file_id": ids[2], "decision": "UNKNOWN", "confidence": 0., "reason": "FEATURE_MISSING", "evidence_json": "{}"},
    ])
    data = collect_deletions(conn)
    assert data["delete_paths"] == []
    assert len(data["queues"]["MAYBE"]) == 1
    assert len(data["queues"]["UNKNOWN"]) == 1


def test_cloud_manifest_requires_timestamp_and_size(tmp_path):
    conn = db.open_db(tmp_path / "db.sqlite")
    keep = db.insert_file(conn, _file("/x/a.jpg"))
    candidate = _file("/x/b.jpg")
    candidate["exif_datetime"] = None
    remove = db.insert_file(conn, candidate)
    gid = db.insert_group(conn, "exact_dup", keep, [(keep, True, "k"), (remove, False, "d")], 1)
    db.update_member_decisions(conn, gid, [
        {"file_id": keep, "decision": "KEEP", "confidence": 1., "reason": "k", "evidence_json": "{}"},
        {"file_id": remove, "decision": "AUTO_REMOVE", "confidence": 1., "reason": "d", "evidence_json": "{}"},
    ])
    data = collect_deletions(conn)
    assert data["delete_paths"] == ["/x/b.jpg"]
    assert data["cloud_items"] == []


def test_legacy_db_migration(tmp_path):
    import sqlite3
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, basename TEXT,
          size_bytes INTEGER, mtime_ns INTEGER, exif_datetime TEXT, exif_timestamp INTEGER,
          width INTEGER, height INTEGER, file_kind TEXT, motion_partner_id INTEGER,
          scan_status TEXT, scan_error TEXT);
        CREATE TABLE features(file_id INTEGER PRIMARY KEY, phash BLOB,
          dinov2_embedding BLOB, quality_score REAL, quality_meta TEXT,
          face_count INTEGER, faces_json TEXT, status TEXT);
        CREATE TABLE groups(id INTEGER PRIMARY KEY, group_type TEXT, keep_file_id INTEGER,
          member_count INTEGER, created_at INTEGER);
        CREATE TABLE group_members(group_id INTEGER,file_id INTEGER,is_keep INTEGER,reason TEXT,
          PRIMARY KEY(group_id,file_id));
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
    """)
    conn.close()
    migrated = db.open_db(path)
    group_cols = {r[1] for r in migrated.execute("PRAGMA table_info(groups)")}
    member_cols = {r[1] for r in migrated.execute("PRAGMA table_info(group_members)")}
    assert {"decision_state", "policy_version"} <= group_cols
    assert {"decision", "evidence_json"} <= member_cols
    feature_cols = {r[1] for r in migrated.execute("PRAGMA table_info(features)")}
    assert "content_sha256" in feature_cols
