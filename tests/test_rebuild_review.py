"""Tests for rebuild_review.py fail-closed behavior."""
import tempfile
from pathlib import Path
import pytest
import numpy as np

from src import db
from src.rebuild_review import rebuild


def test_rebuild_review_missing_db():
    """Rebuild must fail closed when DB path doesn't exist (e.g., typo in config)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Point to a non-existent DB
        config_path = Path(tmpdir) / "bad-config.yaml"
        bad_db = Path(tmpdir) / "nonexistent.sqlite"
        config_path.write_text(f"""
paths:
  db: {bad_db}
  output_dir: {tmpdir}/output
""")
        with pytest.raises(FileNotFoundError, match="inventory DB not found"):
            rebuild(str(config_path), None)


def test_rebuild_review_missing_schema():
    """Rebuild must fail closed when DB has no essential tables."""
    import sqlite3
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "incomplete.sqlite"
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(f"""
paths:
  db: {db_path}
  output_dir: {tmpdir}/output
""")
        # Create empty DB without schema (just raw SQLite)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="missing essential tables"):
            rebuild(str(config_path), None)


def test_rebuild_review_stage1_not_done():
    """Rebuild must fail closed when stage1_done_at is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "incomplete.sqlite"
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(f"""
paths:
  db: {db_path}
  output_dir: {tmpdir}/output
""")
        conn = db.open_db(db_path)
        # Schema exists (from open_db), but no stage1_done_at
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="has no stage1_done_at"):
            rebuild(str(config_path), None)


def test_rebuild_review_no_features():
    """Rebuild must fail closed when DB has 0 features rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.sqlite"
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(f"""
paths:
  db: {db_path}
  output_dir: {tmpdir}/output
""")
        conn = db.open_db(db_path)
        db.set_meta(conn, "stage1_done_at", "1234567890")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="has 0 features rows"):
            rebuild(str(config_path), None)


def test_rebuild_review_valid_db_succeeds():
    """Rebuild succeeds when DB has valid schema + stage1 completion + features."""
    from src import root_scope
    with tempfile.TemporaryDirectory() as tmpdir:
        photos_root = Path(tmpdir) / "photos"
        photos_root.mkdir()
        db_path = Path(tmpdir) / "valid.sqlite"
        out_dir = Path(tmpdir) / "output"
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(f"""
paths:
  db: {db_path}
  root: {photos_root}
  output_dir: {out_dir}
""")
        # Create minimal valid DB with proper root binding
        conn = db.open_db(db_path)
        root_scope.bind(conn, str(photos_root), db_path=str(db_path))
        conn.execute(
            "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
            (f"{photos_root}/test.jpg", 1000000, 100, 1920, 1080, "jpg"),
        )
        conn.commit()
        emb = np.random.randn(768).astype(np.float16)
        emb /= np.linalg.norm(emb)
        conn.execute(
            "INSERT INTO features (file_id, phash, content_sha256, dinov2_embedding, quality_score, quality_meta, faces_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, (0x1234567812345678).to_bytes(8, "big"), "a" * 64, emb.tobytes(), 50.0,
             '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
             "[]", "done"),
        )
        db.set_meta(conn, "stage1_done_at", "1234567890")
        conn.commit()
        conn.close()

        # Should succeed
        result = rebuild(str(config_path), str(photos_root))
        assert result == 0
        # Verify stage 3 output exists
        assert (out_dir / "review.html").exists()
        assert (out_dir / "delete_local.txt").exists()
