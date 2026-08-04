"""Human review state: never touches originals, writes only derived data.

All decisions are recorded in ``output/review_state.json``, atomically written
every time. Original photos stay read-only; the AI decisions/manifests remain
unchanged. This is a parallel, independent layer.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from . import db

_SCHEMA_VERSION = 1


class ReviewState:
    """Thread-safe review state backed by atomic JSON writes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.path = self.output_dir / "review_state.json"
        self._data: Dict[str, Any] = {"version": _SCHEMA_VERSION, "groups": {}}
        self._load()

    def _load(self) -> None:
        """Load existing state. Invalid files are left intact and ignored."""
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict) or raw.get("version") != _SCHEMA_VERSION:
                return
            groups = raw.get("groups", {})
            if not isinstance(groups, dict):
                return
            self._data["groups"] = groups
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        """Atomic write: temp file + rename."""
        payload = json.dumps(self._data, ensure_ascii=False, indent=2).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.output_dir, prefix=".review_state_", suffix=".tmp",
            delete=False
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)

    def get_group(self, group_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve human decision for one group, or None."""
        return self._data["groups"].get(str(group_id))

    def set_group(self, group_id: int, decision: Dict[str, Any]) -> None:
        """Record one group's human decision and persist atomically."""
        self._data["groups"][str(group_id)] = decision
        self._save()

    def clear_group(self, group_id: int) -> None:
        """Remove human decision for one group."""
        self._data["groups"].pop(str(group_id), None)
        self._save()

    def summary(self) -> Dict[str, Any]:
        """Return counts for display."""
        groups = self._data["groups"]
        reviewed = sum(1 for g in groups.values() if g.get("action") in ("accept", "pick"))
        marked = sum(1 for g in groups.values() if g.get("action") == "mark")
        return {"total": len(groups), "reviewed": reviewed, "marked": marked}


def validate_action(
    conn: sqlite3.Connection,
    group_id: int,
    file_id: Optional[int],
    action: str,
    scope: Any,
) -> Optional[str]:
    """Validate that group/file_id belong to scope and group.

    Returns error message or None. Never opens an original.
    """
    if action not in ("accept", "pick", "mark", "clear"):
        return f"invalid action: {action}"

    # Check group exists and is in scope
    predicate, params = db.scope_sql(scope, "f")
    row = conn.execute(
        f"""
        SELECT 1 FROM groups g WHERE g.id = :group_id
        AND EXISTS (
            SELECT 1 FROM group_members gm JOIN files f ON f.id = gm.file_id
            WHERE gm.group_id = g.id AND {predicate}
        )
        """,
        {**params, "group_id": int(group_id)},
    ).fetchone()
    if row is None:
        return "group not found or out of scope"

    # For pick action, verify file_id is a member of this group
    if action == "pick":
        if file_id is None:
            return "pick action requires file_id"
        member_row = conn.execute(
            f"""
            SELECT 1 FROM group_members gm
            JOIN files f ON f.id = gm.file_id
            WHERE gm.group_id = :group_id AND gm.file_id = :file_id AND {predicate}
            """,
            {**params, "group_id": int(group_id), "file_id": int(file_id)},
        ).fetchone()
        if member_row is None:
            return "file_id not a member of this group"

    return None
