"""Shared frozen SQLite input contract: scoped identities, full features, explicit close."""

from __future__ import annotations

import base64
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

FEATURE_SCHEMA = "full-db-features-v1"


@contextmanager
def readonly_inventory(path):
    path = Path(path).resolve(strict=True)
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("frozen inventory has nonempty WAL")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        yield conn
        if wal.exists() and wal.stat().st_size:
            raise ValueError("WAL appeared during frozen read")
    finally:
        conn.close()


def decode_dino(blob):
    if not isinstance(blob, bytes) or len(blob) != 1536:
        raise ValueError("DINO requires <f2 x 768 (1536 bytes)")
    vector = np.frombuffer(blob, dtype="<f2").astype(np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid DINO vector")
    return vector / norm


def feature_rows(path, date, identities):
    """Return ALL persisted feature columns, not the lossy VLM L2 projection.

    Numeric file IDs are meaningful only in the explicitly supplied date DB.
    Original paths are returned separately to the local copy-only exporter.
    """
    ids = sorted(identities)
    if not ids or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError("invalid file allowlist")
    with readonly_inventory(path) as conn:
        rows = conn.execute(
            f"SELECT fe.*, f.path AS source_path FROM features fe JOIN files f ON f.id=fe.file_id WHERE fe.file_id IN ({','.join('?' for _ in ids)})",
            ids,
        ).fetchall()
    if len(rows) != len(ids) or {r["file_id"] for r in rows} != set(ids):
        raise ValueError("missing or duplicate feature identity")
    result, sources = [], {}
    for row in rows:
        record = dict(row)
        fid = record["file_id"]
        if record["content_sha256"] != identities[fid]:
            raise ValueError("feature SHA identity mismatch")
        sources[fid] = Path(record.pop("source_path"))
        decode_dino(record["dinov2_embedding"])
        result.append(
            {
                "date": date,
                "columns": {
                    key: {
                        "encoding": "base64",
                        "data": base64.b64encode(value).decode(),
                    }
                    if isinstance(value, bytes)
                    else value
                    for key, value in record.items()
                },
            }
        )
    return sorted(result, key=lambda r: r["columns"]["file_id"]), sources


def group_embedding(records):
    vectors = [
        decode_dino(
            base64.b64decode(r["columns"]["dinov2_embedding"]["data"], validate=True)
        )
        for r in records
    ]
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid group embedding")
    return mean / norm
