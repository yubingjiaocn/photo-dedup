"""Thumbnail bookkeeping + the compact review pagination index.

Split out of :mod:`src.db` to keep both modules small and focused. Everything
here is read/written by ordinary SQL against the schema declared in ``db.py``;
no image is ever opened. ``db`` re-exports these names, so existing callers and
``db.<helper>`` usage keep working.

The review index is the mechanism that makes a 100k+ photo library browsable:
stage 3 writes one dense, ordered row per visible item per view, so the server
answers a page with ``LIMIT/OFFSET`` instead of materialising a giant JSON blob.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Sequence

# Still-image kinds Stage 1 analyses (videos are inventoried but not decoded).
FEATURE_KINDS = ("jpg", "jpg_motion")

# --- thumbnails ------------------------------------------------------------

def batch_upsert_thumbnails(conn: sqlite3.Connection, rows: Sequence[Dict[str, Any]]) -> None:
    """Upsert thumbnail bookkeeping rows. Caller commits with the feature batch."""
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO thumbnails
            (file_id, status, max_px, bytes, source_size_bytes, source_mtime_ns,
             error, created_at)
        VALUES
            (:file_id, :status, :max_px, :bytes, :source_size_bytes,
             :source_mtime_ns, :error, :created_at)
        ON CONFLICT(file_id) DO UPDATE SET
            status=excluded.status, max_px=excluded.max_px, bytes=excluded.bytes,
            source_size_bytes=excluded.source_size_bytes,
            source_mtime_ns=excluded.source_mtime_ns,
            error=excluded.error, created_at=excluded.created_at
        """,
        rows,
    )


def thumbnail_ids_recorded_ok(conn: sqlite3.Connection) -> List[int]:
    return [int(r["file_id"]) for r in
            conn.execute("SELECT file_id FROM thumbnails WHERE status = 'ok' ORDER BY file_id")]


def thumbnail_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Recorded thumbnail counts/bytes, independent of what stage 1 just did."""
    row = conn.execute(
        "SELECT COALESCE(SUM(status='ok'), 0) AS ok, COALESCE(SUM(status='error'), 0) AS failed, "
        "COALESCE(SUM(CASE WHEN status='ok' THEN bytes END), 0) AS total_bytes, "
        "COUNT(*) AS rows_total FROM thumbnails"
    ).fetchone()
    ok = int(row["ok"])
    total_bytes = int(row["total_bytes"])
    return {
        "recorded_ok": ok,
        "recorded_failed": int(row["failed"]),
        "recorded_rows": int(row["rows_total"]),
        "recorded_bytes": total_bytes,
        "average_bytes": (total_bytes / ok) if ok else None,
    }


def thumbnail_failures(conn: sqlite3.Connection, limit: int = 50) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT t.file_id, f.basename, t.error FROM thumbnails t "
        "LEFT JOIN files f ON f.id = t.file_id "
        "WHERE t.status = 'error' ORDER BY t.file_id LIMIT ?",
        (int(limit),),
    ).fetchall()
    return [{"file_id": int(r["file_id"]), "basename": r["basename"], "error": r["error"]}
            for r in rows]


# --- review pagination index ----------------------------------------------

REVIEW_VIEWS = ("ALL", "MAYBE", "UNKNOWN", "GROUPS")

# Chronological then directory/name order: capture time first (EXIF, else mtime
# derived in stage 0), then the containing directory, then the filename. This
# keeps an ALL timeline readable even when EXIF is missing for whole folders.
ALL_VIEW_ORDER = (
    "ORDER BY COALESCE(f.exif_timestamp, f.mtime_ns / 1000000000), "
    "rtrim(substr(replace(f.path, '\\', '/'), 1, "
    "length(replace(f.path, '\\', '/')) - length(f.basename)), '/'), "
    "f.basename, f.id"
)


def replace_review_index(conn: sqlite3.Connection, view: str,
                         rows: Sequence[Dict[str, Any]]) -> int:
    """Rewrite one view's ordered index. Positions are dense and 0-based."""
    if view not in REVIEW_VIEWS:
        raise ValueError(f"unknown review view: {view}")
    conn.execute("DELETE FROM review_index WHERE view = ?", (view,))
    conn.executemany(
        "INSERT INTO review_index (view, position, file_id, group_id, decision, risk) "
        "VALUES (:view, :position, :file_id, :group_id, :decision, :risk)",
        [{"view": view, "position": index, "file_id": row.get("file_id"),
          "group_id": row.get("group_id"), "decision": row.get("decision"),
          "risk": row.get("risk")} for index, row in enumerate(rows)],
    )
    return len(rows)


def build_all_view_index(conn: sqlite3.Connection) -> int:
    """Index every still image in timeline order, with its decision if any.

    Reads only inventory/feature/decision tables -- no image is opened. The
    membership sub-select collapses to at most one row per file so a file can
    never appear twice in the ALL timeline.
    """
    placeholders = ", ".join("?" for _ in FEATURE_KINDS)
    conn.execute("DELETE FROM review_index WHERE view = 'ALL'")
    conn.execute(
        f"""
        INSERT INTO review_index (view, position, file_id, group_id, decision, risk)
        SELECT 'ALL',
               ROW_NUMBER() OVER ({ALL_VIEW_ORDER}) - 1,
               f.id, m.group_id, m.decision, NULL
        FROM files f
        LEFT JOIN features fe ON fe.file_id = f.id
        LEFT JOIN (
            SELECT file_id, MIN(group_id) AS group_id, MIN(decision) AS decision
            FROM group_members GROUP BY file_id
        ) m ON m.file_id = f.id
        WHERE f.file_kind IN ({placeholders})
          AND COALESCE(fe.status, 'pending') != 'skipped_oversize'
        """,
        FEATURE_KINDS,
    )
    return int(conn.execute(
        "SELECT COUNT(*) AS n FROM review_index WHERE view = 'ALL'").fetchone()["n"])


def review_index_count(conn: sqlite3.Connection, view: str) -> int:
    if view not in REVIEW_VIEWS:
        raise ValueError(f"unknown review view: {view}")
    return int(conn.execute(
        "SELECT COUNT(*) AS n FROM review_index WHERE view = ?", (view,)).fetchone()["n"])


def review_page(conn: sqlite3.Connection, view: str, offset: int,
                limit: int) -> List[sqlite3.Row]:
    """One page of a view, joined with display metadata and thumbnail status.

    Source paths stay in the DB; the server projects only path-free fields.
    """
    if view not in REVIEW_VIEWS:
        raise ValueError(f"unknown review view: {view}")
    if offset < 0 or limit < 1:
        raise ValueError("offset must be >= 0 and limit >= 1")
    return conn.execute(
        """
        SELECT ri.position, ri.file_id, ri.group_id, ri.decision, ri.risk,
               f.basename, f.width, f.height, f.size_bytes, f.exif_datetime,
               f.file_kind, fe.quality_score, fe.face_count, gm.reason,
               t.status AS thumb_status, t.error AS thumb_error
        FROM review_index ri
        JOIN files f ON f.id = ri.file_id
        LEFT JOIN features fe ON fe.file_id = ri.file_id
        LEFT JOIN thumbnails t ON t.file_id = ri.file_id
        LEFT JOIN group_members gm
               ON gm.file_id = ri.file_id AND gm.group_id IS ri.group_id
        WHERE ri.view = ?
        GROUP BY ri.position
        ORDER BY ri.position
        LIMIT ? OFFSET ?
        """,
        (view, int(limit), int(offset)),
    ).fetchall()


def group_page(conn: sqlite3.Connection, offset: int, limit: int) -> List[sqlite3.Row]:
    """One page of groups (ordered by id) with their member rows."""
    if offset < 0 or limit < 1:
        raise ValueError("offset must be >= 0 and limit >= 1")
    return conn.execute(
        """
        SELECT g.id AS group_id, g.group_type, g.member_count,
               gm.file_id, gm.is_keep, gm.decision, gm.reason,
               f.basename, f.width, f.height, f.size_bytes, f.exif_datetime,
               fe.quality_score, fe.face_count,
               t.status AS thumb_status, t.error AS thumb_error
        FROM (SELECT id FROM groups ORDER BY id LIMIT ? OFFSET ?) page
        JOIN groups g ON g.id = page.id
        JOIN group_members gm ON gm.group_id = g.id
        JOIN files f ON f.id = gm.file_id
        LEFT JOIN features fe ON fe.file_id = gm.file_id
        LEFT JOIN thumbnails t ON t.file_id = gm.file_id
        ORDER BY g.id, gm.is_keep DESC, gm.file_id
        """,
        (int(limit), int(offset)),
    ).fetchall()


def group_page_by_id(conn: sqlite3.Connection, group_id: int) -> List[sqlite3.Row]:
    """One group's path-free display join for on-demand lightbox navigation."""
    return conn.execute(
        """
        SELECT g.id AS group_id, g.group_type, g.member_count,
               gm.file_id, gm.is_keep, gm.decision, gm.reason,
               f.basename, f.width, f.height, f.size_bytes, f.exif_datetime,
               f.file_kind, fe.quality_score, fe.face_count,
               t.status AS thumb_status, t.error AS thumb_error
        FROM groups g
        JOIN group_members gm ON gm.group_id = g.id
        JOIN files f ON f.id = gm.file_id
        LEFT JOIN features fe ON fe.file_id = gm.file_id
        LEFT JOIN thumbnails t ON t.file_id = gm.file_id
        WHERE g.id = ?
        ORDER BY gm.is_keep DESC, gm.file_id
        """,
        (int(group_id),),
    ).fetchall()


def count_groups(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM groups").fetchone()["n"])
