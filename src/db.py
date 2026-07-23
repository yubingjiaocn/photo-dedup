"""SQLite schema + helpers for the photo-dedup pipeline.

Everything the pipeline computes lives in a single SQLite file (default
``inventory.sqlite``). This keeps the expensive, HDD-bound work (stage0/1)
separate from the cheap, re-runnable work (stage2/3): tweak a threshold and
re-run clustering without re-reading a single JPEG.

Design notes
------------
* WAL journalling + ``synchronous=NORMAL`` -> fast, crash-safe-enough commits.
* Features are stored as compact BLOBs (float16 embedding, 8-byte pHash).
* ``iter_files_for_features`` yields only work that is still ``pending`` so
  stage1 is naturally resumable after a Ctrl+C.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

# --- schema ----------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  basename TEXT,
  size_bytes INTEGER,
  mtime_ns INTEGER,
  exif_datetime TEXT,
  exif_timestamp INTEGER,
  width INTEGER,
  height INTEGER,
  file_kind TEXT,               -- 'jpg' | 'jpg_motion' | 'mp4_paired' | 'mp4_only'
  motion_partner_id INTEGER,    -- the other half of a motion photo (NULL if none)
  scan_status TEXT DEFAULT 'pending',
  scan_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_kind ON files(file_kind);
CREATE INDEX IF NOT EXISTS idx_files_ts   ON files(exif_timestamp);
CREATE INDEX IF NOT EXISTS idx_files_base ON files(basename);

CREATE TABLE IF NOT EXISTS features (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  phash BLOB,                   -- 8-byte 64-bit perceptual hash
  dinov2_embedding BLOB,        -- 768-dim float16 = 1536 bytes
  quality_score REAL,           -- MUSIQ 0-100 (or stub proxy)
  quality_meta TEXT,            -- JSON: {sharpness, clipiqa, ...}
  face_count INTEGER,
  faces_json TEXT,              -- YuNet: [{bbox, landmarks, score}, ...]
  status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS groups (
  id INTEGER PRIMARY KEY,
  group_type TEXT,              -- 'exact_dup' | 'burst' | 'similar_scene'
  keep_file_id INTEGER,
  member_count INTEGER,
  created_at INTEGER
);

CREATE TABLE IF NOT EXISTS group_members (
  group_id INTEGER,
  file_id INTEGER,
  is_keep INTEGER,              -- SQLite has no bool; 0/1
  reason TEXT,
  PRIMARY KEY (group_id, file_id)
);
CREATE INDEX IF NOT EXISTS idx_gm_file ON group_members(file_id);

-- Simple key/value for pipeline bookkeeping (stage completion, timings...).
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

FILE_COLUMNS = (
    "path", "basename", "size_bytes", "mtime_ns", "exif_datetime",
    "exif_timestamp", "width", "height", "file_kind",
    "motion_partner_id", "scan_status", "scan_error",
)


# --- connection ------------------------------------------------------------

def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite DB and ensure the schema exists.

    Uses WAL + ``synchronous=NORMAL`` for fast, resumable batch writes. Rows
    are returned as ``sqlite3.Row`` so callers can use column names.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --- files -----------------------------------------------------------------

def insert_file(conn: sqlite3.Connection, meta: Dict[str, Any]) -> int:
    """Insert (or ignore-if-duplicate) one file row. Returns the row id.

    ``meta`` may contain any subset of :data:`FILE_COLUMNS`; missing keys are
    stored as NULL. Uniqueness is on ``path`` so re-scans are idempotent.
    """
    cols = [c for c in FILE_COLUMNS if c in meta]
    placeholders = ", ".join("?" for _ in cols)
    col_sql = ", ".join(cols)
    values = [meta[c] for c in cols]
    cur = conn.execute(
        f"INSERT OR IGNORE INTO files ({col_sql}) VALUES ({placeholders})",
        values,
    )
    if cur.lastrowid and cur.rowcount:
        return int(cur.lastrowid)
    # Row already existed -> look up its id.
    row = conn.execute("SELECT id FROM files WHERE path = ?", (meta["path"],)).fetchone()
    return int(row["id"]) if row else int(cur.lastrowid or 0)


def set_motion_partner(conn: sqlite3.Connection, file_id: int, partner_id: int) -> None:
    """Link two rows as a motion-photo pair (bidirectional)."""
    conn.execute("UPDATE files SET motion_partner_id = ? WHERE id = ?", (partner_id, file_id))
    conn.execute("UPDATE files SET motion_partner_id = ? WHERE id = ?", (file_id, partner_id))


def get_file_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()


def count_files(conn: sqlite3.Connection, where: str = "") -> int:
    sql = "SELECT COUNT(*) AS n FROM files"
    if where:
        sql += f" WHERE {where}"
    return int(conn.execute(sql).fetchone()["n"])


# --- features --------------------------------------------------------------

FEATURE_KINDS = ("jpg", "jpg_motion")


def iter_files_for_features(
    conn: sqlite3.Connection, batch_size: int = 256
) -> Iterator[sqlite3.Row]:
    """Yield still-image rows that have no completed feature row yet.

    Resumable: a file counts as pending unless it has a ``features`` row with
    ``status='done'``. Streams in batches to keep memory flat on 66k files.
    """
    placeholders = ", ".join("?" for _ in FEATURE_KINDS)
    sql = f"""
        SELECT f.* FROM files f
        LEFT JOIN features fe ON fe.file_id = f.id
        WHERE f.file_kind IN ({placeholders})
          AND (fe.file_id IS NULL OR fe.status != 'done')
        ORDER BY f.id
    """
    cur = conn.execute(sql, FEATURE_KINDS)
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield row


def batch_insert_features(conn: sqlite3.Connection, rows: Sequence[Dict[str, Any]]) -> None:
    """Upsert a batch of feature rows. Caller commits (stage1 commits per batch)."""
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO features
            (file_id, phash, dinov2_embedding, quality_score,
             quality_meta, face_count, faces_json, status)
        VALUES
            (:file_id, :phash, :dinov2_embedding, :quality_score,
             :quality_meta, :face_count, :faces_json, :status)
        ON CONFLICT(file_id) DO UPDATE SET
            phash=excluded.phash,
            dinov2_embedding=excluded.dinov2_embedding,
            quality_score=excluded.quality_score,
            quality_meta=excluded.quality_meta,
            face_count=excluded.face_count,
            faces_json=excluded.faces_json,
            status=excluded.status
        """,
        rows,
    )


def load_features_joined(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Load every still image joined with its features, for clustering.

    ~66k rows * ~1.6KB embedding ~= 100MB in memory, which is fine.
    """
    placeholders = ", ".join("?" for _ in FEATURE_KINDS)
    sql = f"""
        SELECT f.id, f.path, f.basename, f.size_bytes, f.mtime_ns,
               f.exif_datetime, f.exif_timestamp, f.width, f.height,
               f.file_kind, f.motion_partner_id,
               fe.phash, fe.dinov2_embedding, fe.quality_score,
               fe.quality_meta, fe.face_count, fe.faces_json
        FROM files f
        JOIN features fe ON fe.file_id = f.id
        WHERE f.file_kind IN ({placeholders})
          AND fe.status = 'done'
        ORDER BY COALESCE(f.exif_timestamp, f.mtime_ns/1000000000), f.id
    """
    return conn.execute(sql, FEATURE_KINDS).fetchall()


# --- groups ----------------------------------------------------------------

def clear_groups(conn: sqlite3.Connection) -> None:
    """Wipe clustering output so stage2 can be re-run idempotently."""
    conn.execute("DELETE FROM group_members")
    conn.execute("DELETE FROM groups")
    conn.commit()


def insert_group(
    conn: sqlite3.Connection,
    group_type: str,
    keep_file_id: int,
    members: Sequence[Tuple[int, bool, str]],
    created_at: int,
) -> int:
    """Insert one group + its members.

    ``members`` is a sequence of ``(file_id, is_keep, reason)``.
    """
    cur = conn.execute(
        "INSERT INTO groups (group_type, keep_file_id, member_count, created_at) "
        "VALUES (?, ?, ?, ?)",
        (group_type, keep_file_id, len(members), created_at),
    )
    group_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT INTO group_members (group_id, file_id, is_keep, reason) "
        "VALUES (?, ?, ?, ?)",
        [(group_id, fid, 1 if keep else 0, reason) for (fid, keep, reason) in members],
    )
    return group_id


def iter_groups(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    return conn.execute("SELECT * FROM groups ORDER BY id").fetchall()


def group_members(conn: sqlite3.Connection, group_id: int) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT gm.*, f.path, f.basename, f.size_bytes, f.width, f.height,
               f.exif_datetime, f.motion_partner_id,
               fe.quality_score, fe.quality_meta, fe.face_count
        FROM group_members gm
        JOIN files f ON f.id = gm.file_id
        LEFT JOIN features fe ON fe.file_id = gm.file_id
        WHERE gm.group_id = ?
        ORDER BY gm.is_keep DESC, f.id
        """,
        (group_id,),
    ).fetchall()


def get_file(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()


# --- meta ------------------------------------------------------------------

def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default
