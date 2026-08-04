"""SQLite DDL + forward migrations for the photo-dedup pipeline.

Split out of :mod:`src.db` so that module stays about connections and queries.
Two rules hold here:

* **Additive only.** Migrations add columns/indexes and backfill derived values.
  They never drop or delete a user's rows -- an older database keeps every
  photo record it had, and an incompatible reuse is rejected up front by
  :mod:`src.root_scope` instead of being "cleaned up" behind the user's back.
* **Idempotent.** ``apply`` runs on every :func:`src.db.open_db`.
"""

from __future__ import annotations

import sqlite3
from typing import Dict

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
  scan_error TEXT,
  path_key TEXT,                -- normalised path (case/separator folded)
  root_key TEXT,                -- normalised photo root this row was scanned under
  last_run_id TEXT              -- scope run id that last saw this file
);
CREATE INDEX IF NOT EXISTS idx_files_kind ON files(file_kind);
CREATE INDEX IF NOT EXISTS idx_files_ts   ON files(exif_timestamp);
CREATE INDEX IF NOT EXISTS idx_files_base ON files(basename);
-- Indexes on the identity columns are created by the migration step, after the
-- columns are guaranteed to exist on an older database too.

CREATE TABLE IF NOT EXISTS features (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  phash BLOB,                   -- 8-byte 64-bit perceptual hash
  content_sha256 TEXT,          -- byte identity; pHash alone is never exact proof
  dinov2_embedding BLOB,        -- 768-dim float16 = 1536 bytes
  quality_score REAL,           -- MUSIQ 0-100 (or stub proxy)
  quality_meta TEXT,            -- JSON: {sharpness, clipiqa, ...}
  face_count INTEGER,
  faces_json TEXT,              -- YuNet: [{bbox, landmarks, score}, ...]
  status TEXT DEFAULT 'pending'
);

-- SSD thumbnail cache bookkeeping. One row per still image Stage 1 handled.
-- Identity columns capture the source file as it was when the JPEG was made,
-- so a changed original can never reuse a stale thumbnail.
CREATE TABLE IF NOT EXISTS thumbnails (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  status TEXT,                  -- 'ok' | 'error'
  max_px INTEGER,
  bytes INTEGER,
  source_size_bytes INTEGER,
  source_mtime_ns INTEGER,
  error TEXT,
  created_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_thumbs_status ON thumbnails(status);

-- Compact, ordered pagination index built by stage 3 (one row per visible
-- item per view) so the review server can page 100k photos with LIMIT/OFFSET
-- instead of shipping a huge JSON document to the browser.
CREATE TABLE IF NOT EXISTS review_index (
  view TEXT,                    -- 'ALL' | 'MAYBE' | 'UNKNOWN' | 'GROUPS'
  position INTEGER,
  file_id INTEGER,
  group_id INTEGER,
  decision TEXT,
  risk REAL,
  PRIMARY KEY (view, position)
);
CREATE INDEX IF NOT EXISTS idx_review_index_file ON review_index(file_id);

CREATE TABLE IF NOT EXISTS groups (
  id INTEGER PRIMARY KEY,
  group_type TEXT,              -- 'exact_dup' | 'burst' | 'similar_scene'
  keep_file_id INTEGER,
  member_count INTEGER,
  created_at INTEGER,
  decision_state TEXT,
  confidence REAL,
  policy_version TEXT,
  decision_json TEXT
);

CREATE TABLE IF NOT EXISTS group_members (
  group_id INTEGER,
  file_id INTEGER,
  is_keep INTEGER,              -- SQLite has no bool; 0/1
  reason TEXT,
  decision TEXT,
  confidence REAL,
  evidence_json TEXT,
  user_override TEXT,
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
    "path_key", "root_key", "last_run_id",
)

# Columns added by later releases, per table.
_ADDITIONS: Dict[str, Dict[str, str]] = {
    "features": {"content_sha256": "TEXT"},
    "groups": {"decision_state": "TEXT", "confidence": "REAL",
               "policy_version": "TEXT", "decision_json": "TEXT"},
    "group_members": {"decision": "TEXT", "confidence": "REAL",
                      "evidence_json": "TEXT", "user_override": "TEXT"},
    "files": {"path_key": "TEXT", "root_key": "TEXT", "last_run_id": "TEXT"},
}


def apply(conn: sqlite3.Connection) -> Dict[str, int]:
    """Create the schema if needed, then run additive migrations."""
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    from . import root_scope

    return root_scope.migrate(conn)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDITIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
