"""Tests for the SQLite schema + helpers."""

import numpy as np

from src import db
from src import quality as Q


def _meta(path, kind="jpg", ts=1000):
    return {
        "path": path, "basename": path.split("/")[-1], "size_bytes": 123,
        "mtime_ns": 1, "exif_datetime": "2026-01-01 12:00:00",
        "exif_timestamp": ts, "width": 4000, "height": 3000,
        "file_kind": kind, "scan_status": "done",
    }


def test_open_db_is_wal(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    conn.close()


def test_insert_file_idempotent(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    fid1 = db.insert_file(conn, _meta("/x/a.jpg"))
    fid2 = db.insert_file(conn, _meta("/x/a.jpg"))  # same path
    assert fid1 == fid2
    assert db.count_files(conn) == 1
    conn.close()


def test_motion_partner_bidirectional(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    a = db.insert_file(conn, _meta("/x/a.jpg", "jpg_motion"))
    b = db.insert_file(conn, _meta("/x/a.mp4", "mp4_paired"))
    db.set_motion_partner(conn, a, b)
    ra = db.get_file(conn, a)
    rb = db.get_file(conn, b)
    assert ra["motion_partner_id"] == b
    assert rb["motion_partner_id"] == a
    conn.close()


def test_features_roundtrip_and_resumability(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    ids = [db.insert_file(conn, _meta(f"/x/{i}.jpg", ts=1000 + i)) for i in range(3)]
    conn.commit()

    # All three pending initially.
    assert len(list(db.iter_files_for_features(conn))) == 3

    emb = np.arange(768, dtype=np.float32)
    rows = [{
        "file_id": ids[0], "phash": (1234).to_bytes(8, "big"),
        "dinov2_embedding": Q.embedding_to_blob(emb), "quality_score": 55.0,
        "quality_meta": "{}", "face_count": 0, "faces_json": "[]", "status": "done",
    }]
    db.batch_insert_features(conn, rows)
    conn.commit()

    # One done -> two pending remain (resumability).
    assert len(list(db.iter_files_for_features(conn))) == 2

    joined = db.load_features_joined(conn)
    assert len(joined) == 1
    back = Q.blob_to_embedding(joined[0]["dinov2_embedding"])
    assert back.shape[0] == 768
    assert np.allclose(back[:5], emb[:5], atol=1e-2)
    conn.close()


def test_groups_and_clear(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    a = db.insert_file(conn, _meta("/x/a.jpg"))
    b = db.insert_file(conn, _meta("/x/b.jpg"))
    gid = db.insert_group(conn, "burst", a, [(a, True, "keep"), (b, False, "dup")], 111)
    conn.commit()
    assert len(list(db.iter_groups(conn))) == 1
    members = db.group_members(conn, gid)
    assert len(members) == 2
    assert members[0]["is_keep"] == 1  # keep sorted first
    db.clear_groups(conn)
    assert len(list(db.iter_groups(conn))) == 0
    conn.close()


def test_meta_kv(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    db.set_meta(conn, "k", "v1")
    db.set_meta(conn, "k", "v2")  # upsert
    assert db.get_meta(conn, "k") == "v2"
    assert db.get_meta(conn, "missing", "d") == "d"
    conn.close()


def test_done_error_is_terminal(tmp_path):
    conn = db.open_db(tmp_path / "t.sqlite")
    fid = db.insert_file(conn, _meta("/x/bad.jpg"))
    db.batch_insert_features(conn, [{
        "file_id": fid, "phash": None, "dinov2_embedding": None,
        "quality_score": None, "quality_meta": "{}", "face_count": 0,
        "faces_json": "[]", "status": "done_error",
    }])
    assert list(db.iter_files_for_features(conn)) == []
