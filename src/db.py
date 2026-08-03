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

# Thumbnail bookkeeping and the review pagination index live in their own module
# to keep this file focused on the core schema. They are re-exported here so
# every caller can keep using ``db.<helper>``.
from .review_queries import (  # noqa: F401 (intentional re-export)
    ALL_VIEW_ORDER,
    FEATURE_KINDS,
    REVIEW_VIEWS,
    batch_upsert_thumbnails,
    build_all_view_index,
    count_groups,
    group_page,
    replace_review_index,
    review_index_count,
    review_page,
    thumbnail_failures,
    thumbnail_ids_recorded_ok,
    thumbnail_stats,
)

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
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add P0 decision columns to databases created by pre-P0 releases."""
    additions = {
        "features": {"content_sha256": "TEXT"},
        "groups": {"decision_state": "TEXT", "confidence": "REAL",
                   "policy_version": "TEXT", "decision_json": "TEXT"},
        "group_members": {"decision": "TEXT", "confidence": "REAL",
                          "evidence_json": "TEXT", "user_override": "TEXT"},
    }
    for table, columns in additions.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


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


def refresh_file_identity(conn: sqlite3.Connection, file_id: int,
                         meta: Dict[str, Any]) -> bool:
    """Update a re-scanned row when the file on disk actually changed.

    ``insert_file`` is INSERT-OR-IGNORE, so without this a replaced photo would
    keep the identity it had at first scan. That is not merely cosmetic: the
    stored ``content_sha256`` drives the byte-identity rule that is the *only*
    automatic removal, and the thumbnail cache is validated against
    ``size_bytes``/``mtime_ns``. So when either changes we rewrite the inventory
    fields and invalidate the derived feature row, which makes Stage 1 recompute
    features **and** the thumbnail from one fresh decode.

    Returns True when something changed.
    """
    row = conn.execute(
        "SELECT size_bytes, mtime_ns FROM files WHERE id = ?", (int(file_id),)
    ).fetchone()
    if row is None:
        return False
    if (row["size_bytes"] == meta.get("size_bytes")
            and row["mtime_ns"] == meta.get("mtime_ns")):
        return False
    updatable = [c for c in FILE_COLUMNS if c in meta and c != "path"]
    assignments = ", ".join(f"{name} = ?" for name in updatable)
    conn.execute(f"UPDATE files SET {assignments} WHERE id = ?",
                 [meta[name] for name in updatable] + [int(file_id)])
    # Any cached derivative now describes bytes that no longer exist.
    conn.execute("UPDATE features SET status = 'pending' WHERE file_id = ?", (int(file_id),))
    conn.execute("DELETE FROM thumbnails WHERE file_id = ?", (int(file_id),))
    return True


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

THUMB_PRESENCE_TABLE = "temp.thumb_present"

# A cached thumbnail is only trusted when the recorded identity still matches
# the live inventory row, the pixel size matches the current setting, and the
# JPEG is actually present on the SSD (registered in the presence table).
_THUMB_CURRENT_SQL = f"""
    COALESCE(t.status, '') = 'ok'
    AND COALESCE(t.max_px, -1) = :thumb_max_px
    AND COALESCE(t.source_size_bytes, -1) = COALESCE(f.size_bytes, -2)
    AND COALESCE(t.source_mtime_ns, -1) = COALESCE(f.mtime_ns, -2)
    AND EXISTS (SELECT 1 FROM {THUMB_PRESENCE_TABLE} p WHERE p.file_id = f.id)
"""

_THUMB_SELECT = """
    t.status AS thumb_status, t.max_px AS thumb_max_px, t.bytes AS thumb_bytes,
    t.source_size_bytes AS thumb_source_size_bytes,
    t.source_mtime_ns AS thumb_source_mtime_ns, t.error AS thumb_error
"""


def _ensure_thumb_presence_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TEMP TABLE IF NOT EXISTS {THUMB_PRESENCE_TABLE.split('.', 1)[1]} "
        "(file_id INTEGER PRIMARY KEY)"
    )


def register_thumb_presence(conn: sqlite3.Connection, file_ids: Iterable[int]) -> int:
    """Record which thumbnail JPEGs currently exist on disk (temp table).

    Called once per Stage 1 run from a single cheap SSD directory listing, so a
    manually deleted thumbnail is regenerated instead of silently missing.
    """
    _ensure_thumb_presence_table(conn)
    conn.execute(f"DELETE FROM {THUMB_PRESENCE_TABLE}")
    ids = [(int(value),) for value in file_ids]
    conn.executemany(f"INSERT OR IGNORE INTO {THUMB_PRESENCE_TABLE} (file_id) VALUES (?)", ids)
    return len(ids)


def iter_files_for_features(
    conn: sqlite3.Connection, batch_size: int = 256, thumb_max_px: int | None = None
) -> Iterator[sqlite3.Row]:
    """Yield still-image rows that still need Stage 1 work.

    Resumable: a file counts as pending unless it has a ``features`` row with
    ``status='done'``. When ``thumb_max_px`` is given, a file whose features are
    done but whose SSD thumbnail is missing/stale/failed is pending as well, so
    the single decode of that repair pass serves features *and* the thumbnail.
    Streams in batches to keep memory flat on a 100k+ library.
    """
    if thumb_max_px is None:
        thumb_clause = ""
        params: Dict[str, Any] = {}
    else:
        _ensure_thumb_presence_table(conn)
        thumb_clause = f"OR (fe.status = 'done' AND NOT ({_THUMB_CURRENT_SQL}))"
        params = {"thumb_max_px": int(thumb_max_px)}
    kind_params = {f"kind{i}": kind for i, kind in enumerate(FEATURE_KINDS)}
    kind_placeholders = ", ".join(f":{name}" for name in kind_params)
    sql = f"""
        SELECT f.*, {_THUMB_SELECT} FROM files f
        LEFT JOIN features fe ON fe.file_id = f.id
        LEFT JOIN thumbnails t ON t.file_id = f.id
        WHERE f.file_kind IN ({kind_placeholders})
          AND (
            fe.file_id IS NULL OR fe.status NOT IN ('done', 'done_error')
            {thumb_clause}
          )
        ORDER BY f.id
    """
    cur = conn.execute(sql, {**kind_params, **params})
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
    normalized = [{"content_sha256": None, **row} for row in rows]
    conn.executemany(
        """
        INSERT INTO features
            (file_id, phash, content_sha256, dinov2_embedding, quality_score,
             quality_meta, face_count, faces_json, status)
        VALUES
            (:file_id, :phash, :content_sha256, :dinov2_embedding, :quality_score,
             :quality_meta, :face_count, :faces_json, :status)
        ON CONFLICT(file_id) DO UPDATE SET
            phash=excluded.phash,
            content_sha256=excluded.content_sha256,
            dinov2_embedding=excluded.dinov2_embedding,
            quality_score=excluded.quality_score,
            quality_meta=excluded.quality_meta,
            face_count=excluded.face_count,
            faces_json=excluded.faces_json,
            status=excluded.status
        """,
        normalized,
    )


def count_still_images(conn: sqlite3.Connection) -> int:
    """Number of still images Stage 1 is responsible for (thumbnail denominator)."""
    placeholders = ", ".join("?" for _ in FEATURE_KINDS)
    return int(conn.execute(
        f"SELECT COUNT(*) AS n FROM files WHERE file_kind IN ({placeholders})", FEATURE_KINDS
    ).fetchone()["n"])


def load_features_joined(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Load every still image joined with its features, for clustering.

    ~66k rows * ~1.6KB embedding ~= 100MB in memory, which is fine.
    """
    placeholders = ", ".join("?" for _ in FEATURE_KINDS)
    sql = f"""
        SELECT f.id, f.path, f.basename, f.size_bytes, f.mtime_ns,
               f.exif_datetime, f.exif_timestamp, f.width, f.height,
               f.file_kind, f.motion_partner_id,
               fe.phash, fe.content_sha256, fe.dinov2_embedding, fe.quality_score,
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
    *, decision_state: str | None = None, confidence: float | None = None,
    policy_version: str | None = None, decision_json: str | None = None,
) -> int:
    """Insert one group + its members.

    ``members`` is a sequence of ``(file_id, is_keep, reason)``.
    """
    cur = conn.execute(
        "INSERT INTO groups (group_type, keep_file_id, member_count, created_at, "
        "decision_state, confidence, policy_version, decision_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (group_type, keep_file_id, len(members), created_at, decision_state,
         confidence, policy_version, decision_json),
    )
    group_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT INTO group_members (group_id, file_id, is_keep, reason) "
        "VALUES (?, ?, ?, ?)",
        [(group_id, fid, 1 if keep else 0, reason) for (fid, keep, reason) in members],
    )
    return group_id


def update_member_decisions(conn: sqlite3.Connection, group_id: int,
                            records: Sequence[Dict[str, Any]]) -> None:
    conn.executemany(
        "UPDATE group_members SET decision=:decision, confidence=:confidence, "
        "reason=:reason, evidence_json=:evidence_json WHERE group_id=:group_id AND file_id=:file_id",
        [{**r, "group_id": group_id} for r in records],
    )


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
