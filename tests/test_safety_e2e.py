"""End-to-end regressions for safety boundaries through real Stage 2/3 paths."""
import json
import hashlib

import numpy as np
import pytest
import yaml
from PIL import Image

from src import db, exposure, quality
from src import execute_local
from src import stage2_cluster as s2
from src import stage3_report as s3


def _config(tmp_path):
    cfg = {
        "paths": {"root": str(tmp_path / "photos"), "db": str(tmp_path / "db.sqlite"),
                  "trash": str(tmp_path / "trash"), "output_dir": str(tmp_path / "out")},
        "cluster": {"dinov2_threshold": 0.92, "phash_hamming_threshold": 2},
        "execute": {"mode": "move", "dry_run": True},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _add(conn, path, ts, emb, phash, sha, score):
    path.write_bytes((sha or "none").encode())
    fid = db.insert_file(conn, {
        "path": str(path), "basename": path.name, "size_bytes": path.stat().st_size,
        "mtime_ns": 1, "exif_datetime": "2026-01-01 12:00:00", "exif_timestamp": ts,
        "width": 100, "height": 100, "file_kind": "jpg", "scan_status": "done",
    })
    neutral = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    meta = {"exposure": exposure.extract(neutral, []), "face_quality": 0.}
    db.batch_insert_features(conn, [{
        "file_id": fid, "phash": phash.to_bytes(8, "big"), "content_sha256": sha,
        "dinov2_embedding": quality.embedding_to_blob(emb), "quality_score": score,
        "quality_meta": json.dumps(meta), "face_count": 0, "faces_json": "[]", "status": "done",
    }])
    return fid


def test_chained_exact_component_only_direct_byte_duplicate_can_manifest(tmp_path):
    """A-B and B-C pHash edges must not make non-identical C trusted."""
    cfg = _config(tmp_path)
    root = tmp_path / "photos"
    root.mkdir()
    conn = db.open_db(tmp_path / "db.sqlite")
    # pHash A=0, B=1, C=7: A-B=1, B-C=2, A-C=3 (no direct A-C edge).
    # Embeddings form A~B, B~C while A-C is below the representative purity threshold.
    a = np.zeros(768, np.float32)
    a[0] = 1
    c = np.zeros(768, np.float32)
    c[0] = .90
    c[1] = np.sqrt(1 - .90**2)
    b = a + c
    b /= np.linalg.norm(b)
    _add(conn, root / "a.jpg", 1, a, 0, "same", 90)
    _add(conn, root / "b.jpg", 2, b, 1, "same", 80)
    _add(conn, root / "c.jpg", 3, c, 7, "different", 10)
    conn.commit()
    conn.close()

    stats = s2.run(cfg)
    assert stats["auto_remove"] == 1
    report = s3.run(cfg, no_thumbs=True)
    assert report["delete_files"] == 1
    manifest = (tmp_path / "out" / "delete_local.txt").read_text()
    assert "b.jpg" in manifest
    assert "c.jpg" not in manifest


def test_invalid_embeddings_do_not_crash_or_cluster(tmp_path):
    cfg = _config(tmp_path)
    root = tmp_path / "photos"
    root.mkdir()
    conn = db.open_db(tmp_path / "db.sqlite")
    good = np.zeros(768, np.float32)
    good[0] = 1
    _add(conn, root / "a.jpg", 1, good, 0, "a", 80)
    fid = _add(conn, root / "b.jpg", 2, good, 99, "b", 20)
    conn.execute("UPDATE features SET dinov2_embedding=? WHERE file_id=?", (b"odd", fid))
    conn.commit()
    conn.close()
    assert s2.run(cfg)["groups"] == 0


def test_invalid_embedding_in_phash_group_becomes_unknown(tmp_path):
    cfg = _config(tmp_path)
    root = tmp_path / "photos"
    root.mkdir()
    conn = db.open_db(tmp_path / "db.sqlite")
    good = np.zeros(768, np.float32)
    good[0] = 1
    _add(conn, root / "a.jpg", 1, good, 0, "same", 80)
    fid = _add(conn, root / "b.jpg", 2, good, 0, "same", 20)
    conn.execute("UPDATE features SET dinov2_embedding=? WHERE file_id=?", (b"odd", fid))
    conn.commit()
    conn.close()
    stats = s2.run(cfg)
    assert stats["unknown"] == 1
    s3.run(cfg, no_thumbs=True)
    assert (tmp_path / "out" / "delete_local.txt").read_text() == ""


def test_motion_partner_protected_keeper_not_expanded(tmp_path):
    conn = db.open_db(tmp_path / "db.sqlite")
    still = db.insert_file(conn, {"path": "/x/d.jpg", "basename": "d.jpg", "size_bytes": 5,
        "file_kind": "jpg_motion", "scan_status": "done"})
    side = db.insert_file(conn, {"path": "/x/d.mp4", "basename": "d.mp4", "size_bytes": 7,
        "file_kind": "mp4_paired", "scan_status": "done"})
    keeper = db.insert_file(conn, {"path": "/x/k.jpg", "basename": "k.jpg", "size_bytes": 5,
        "file_kind": "jpg", "scan_status": "done"})
    db.set_motion_partner(conn, still, side)
    gid = db.insert_group(conn, "exact_dup", keeper, [(keeper, True, "k"), (still, False, "d")], 1)
    db.update_member_decisions(conn, gid, [
        {"file_id": keeper, "decision": "KEEP", "confidence": 1., "reason": "k", "evidence_json": "{}"},
        {"file_id": still, "decision": "AUTO_REMOVE", "confidence": 1., "reason": "d", "evidence_json": "{}"},
    ])
    # Simulate an inconsistent second owner: reverse ownership is no longer unique.
    other = db.insert_file(conn, {"path": "/x/o.jpg", "basename": "o.jpg", "size_bytes": 5,
        "file_kind": "jpg_motion", "motion_partner_id": side, "scan_status": "done"})
    gid2 = db.insert_group(conn, "burst", other, [(other, True, "k")], 1)
    db.update_member_decisions(conn, gid2, [{"file_id": other, "decision": "KEEP", "confidence": 1.,
                                             "reason": "k", "evidence_json": "{}"}])
    assert db.get_file(conn, side)["motion_partner_id"] == still
    data = s3.collect_deletions(conn)
    assert data["delete_paths"] == ["/x/d.jpg"]


def test_execute_refuses_tampered_or_stale_manifest(tmp_path):
    cfg = _config(tmp_path)
    root = tmp_path / "photos"
    root.mkdir()
    victim = root / "victim.jpg"
    victim.write_bytes(b"x")
    conn = db.open_db(tmp_path / "db.sqlite")
    db.set_meta(conn, "stage2_done_at", "run-1")
    conn.commit()
    conn.close()
    out = tmp_path / "out"
    out.mkdir()
    delete = out / "delete_local.txt"
    delete.write_text(str(victim) + "\n", encoding="utf-8")
    # Missing metadata is allowed for dry run, but never for mutation.
    assert execute_local.run(cfg, dry_run=True)["dry_run"] is True
    with pytest.raises(RuntimeError, match="metadata missing"):
        execute_local.run(cfg, dry_run=False)
    assert victim.exists()


def test_execute_moves_only_with_matching_run_manifest(tmp_path):
    cfg = _config(tmp_path)
    root = tmp_path / "photos"
    root.mkdir()
    victim = root / "victim.jpg"
    victim.write_bytes(b"x")
    conn = db.open_db(tmp_path / "db.sqlite")
    db.set_meta(conn, "stage2_done_at", "run-1")
    conn.commit()
    conn.close()
    out = tmp_path / "out"
    out.mkdir()
    delete = out / "delete_local.txt"
    delete.write_text(str(victim) + "\n", encoding="utf-8")
    (out / "delete_local.meta.json").write_text(json.dumps({
        "stage2_run_id": "run-1", "policy_version": None, "count": 1,
        "paths_sha256": hashlib.sha256(delete.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    result = execute_local.run(cfg, dry_run=False)
    assert result["moved"] == 1
    assert not victim.exists()
