"""Persist Stage-1 exclusions decided from cheap inventory metadata."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List

from .review_queries import FEATURE_KINDS


def mark_unprocessable_skipped(
    conn: sqlite3.Connection, max_pixels: int, max_aspect_ratio: float
) -> List[Dict[str, Any]]:
    """Exclude pixel-heavy/extreme-aspect stills, including old done rows."""
    placeholders = ", ".join("?" for _ in FEATURE_KINDS)
    rows = conn.execute(
        f"SELECT id, path, width, height FROM files WHERE file_kind IN ({placeholders}) "
        "AND width > 0 AND height > 0 AND (width * height > ? "
        "OR MAX(CAST(width AS REAL) / height, CAST(height AS REAL) / width) > ?)",
        (*FEATURE_KINDS, int(max_pixels), float(max_aspect_ratio)),
    ).fetchall()
    if not rows:
        return []
    skipped = []
    for row in rows:
        pixels = int(row["width"]) * int(row["height"])
        ratio = max(
            int(row["width"]) / int(row["height"]),
            int(row["height"]) / int(row["width"]),
        )
        reason = "PIXEL_LIMIT" if pixels > max_pixels else "ASPECT_RATIO"
        skipped.append({
            **dict(row), "skip_reason": reason,
            "megapixels": pixels / 1_000_000, "aspect_ratio": ratio,
        })
    payload = [{
        "file_id": int(row["id"]), "phash": None, "content_sha256": None,
        "dinov2_embedding": None, "quality_score": None,
        "quality_meta": json.dumps({"skip_reason": row["skip_reason"]}),
        "face_count": 0, "faces_json": "[]", "status": "skipped_oversize",
    } for row in skipped]
    conn.executemany(
        """
        INSERT INTO features
            (file_id, phash, content_sha256, dinov2_embedding, quality_score,
             quality_meta, face_count, faces_json, status)
        VALUES
            (:file_id, :phash, :content_sha256, :dinov2_embedding, :quality_score,
             :quality_meta, :face_count, :faces_json, :status)
        ON CONFLICT(file_id) DO UPDATE SET
            phash=excluded.phash, content_sha256=excluded.content_sha256,
            dinov2_embedding=excluded.dinov2_embedding,
            quality_score=excluded.quality_score, quality_meta=excluded.quality_meta,
            face_count=excluded.face_count, faces_json=excluded.faces_json,
            status=excluded.status
        """,
        payload,
    )
    ids = [int(row["id"]) for row in rows]
    id_slots = ", ".join("?" for _ in ids)
    affected_groups = [int(row[0]) for row in conn.execute(
        f"SELECT DISTINCT group_id FROM group_members WHERE file_id IN ({id_slots})",
        ids,
    ).fetchall()]
    conn.executemany("DELETE FROM thumbnails WHERE file_id = ?", [(value,) for value in ids])
    if affected_groups:
        conn.executemany(
            "DELETE FROM group_members WHERE group_id = ?",
            [(value,) for value in affected_groups],
        )
        conn.executemany(
            "DELETE FROM groups WHERE id = ?", [(value,) for value in affected_groups]
        )
    return skipped


def mark_oversize_skipped(
    conn: sqlite3.Connection, max_pixels: int
) -> List[Dict[str, Any]]:
    """Pixel-limit-only compatibility helper."""
    return mark_unprocessable_skipped(conn, max_pixels, float("inf"))
