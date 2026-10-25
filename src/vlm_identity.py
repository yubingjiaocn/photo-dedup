"""Stable review identity; display aliases are never join keys."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{64}$")


def identity(row: Mapping[str, Any]) -> str:
    file_id, sha = row.get("file_id"), row.get("content_sha256")
    if isinstance(file_id, bool) or not isinstance(file_id, int) or file_id <= 0:
        raise ValueError("identity requires positive integer file_id")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ValueError(f"identity requires content_sha256 for file_id={file_id}")
    return f"I{file_id}@{sha}"


def alias_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    seen = set()
    for row in rows:
        key = identity(row)
        alias = row["frame_alias"]
        if key in seen or alias in result:
            raise ValueError("duplicate identity or display alias")
        seen.add(key)
        result[alias] = {"file_id": row["file_id"], "content_sha256": row["content_sha256"]}
    return result


def join_maps(left: Mapping[str, Mapping[str, Any]],
              right: Mapping[str, Mapping[str, Any]], *, by: str = "identity") -> dict[str, str]:
    """Resolve old/new presentation labels only AFTER a bijective identity join.

    Explicit alias joins are rejected even when a particular ordering agrees.
    Old manifests missing hashes must first be enriched from a verified inventory.
    """
    if by != "identity":
        raise ValueError("alias join forbidden: use file_id + content_sha256")
    a = {identity(row): label for label, row in left.items()}
    b = {identity(row): label for label, row in right.items()}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("identity join is not bijective or content SHA changed")
    return {a[key]: b[key] for key in sorted(a)}
