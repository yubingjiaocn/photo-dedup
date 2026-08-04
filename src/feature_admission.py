"""Persist Stage-1 exclusions decided from cheap inventory metadata."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, List

from .review_queries import FEATURE_KINDS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .root_scope import RootScope


def mark_unprocessable_skipped(
    conn: sqlite3.Connection, max_pixels: int, max_aspect_ratio: float,
    scope: "RootScope | None" = None,
) -> List[Dict[str, Any]]:
    """Exclude pixel-heavy/extreme-aspect stills, including old done rows.

    Scoped: only the current root's rows are examined, so a run never re-labels
    (or reports) exclusions belonging to a different library.
    """
    kind_params = {f"kind{index}": kind for index, kind in enumerate(FEATURE_KINDS)}
    placeholders = ", ".join(f":{name}" for name in kind_params)
    predicate: str = "1"
    scope_params: Dict[str, Any] = {}
    if scope is not None:
        predicate, scope_params = scope.clause("f")
    rows = conn.execute(
        f"SELECT f.id, f.path, f.width, f.height FROM files f "
        f"WHERE f.file_kind IN ({placeholders}) AND {predicate} "
        "AND f.width > 0 AND f.height > 0 AND (f.width * f.height > :max_pixels "
        "OR MAX(CAST(f.width AS REAL) / f.height, CAST(f.height AS REAL) / f.width) "
        "> :max_ratio)",
        {**kind_params, **scope_params, "max_pixels": int(max_pixels),
         "max_ratio": float(max_aspect_ratio)},
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
