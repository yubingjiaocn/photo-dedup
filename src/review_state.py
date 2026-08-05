"""Human review state: never touches originals, writes only derived data.

All decisions are recorded in ``output/review_state.json``, atomically written
every time. Original photos stay read-only; the AI decisions/manifests remain
unchanged. This is a parallel, independent layer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from . import db

_SCHEMA_VERSION = 1


def compute_member_fingerprint(member_file_ids: list[int]) -> str:
    """Compute stable fingerprint of group members to detect group_id reassignment."""
    sorted_ids = sorted(member_file_ids)
    payload = ",".join(str(fid) for fid in sorted_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class ReviewState:
    """Thread-safe review state backed by atomic JSON writes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.path = self.output_dir / "review_state.json"
        self._data: Dict[str, Any] = {"version": _SCHEMA_VERSION, "groups": {}}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        """Load existing state. Invalid files are left intact and ignored."""
        with self._lock:
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
                # P0-8: Validate each group's value shape
                validated_groups = {}
                for gid_str, value in groups.items():
                    # Group key must be positive int string
                    if not isinstance(gid_str, str):
                        continue
                    try:
                        gid = int(gid_str)
                        if gid <= 0:
                            continue
                    except (ValueError, OverflowError):
                        continue
                    if not isinstance(value, dict):
                        continue
                    action = value.get("action")
                    if action not in ("accept", "pick", "mark"):
                        continue
                    # Timestamp must be int, not bool
                    timestamp = value.get("timestamp")
                    if timestamp is not None and (not isinstance(timestamp, int) or isinstance(timestamp, bool)):
                        continue
                    file_id = value.get("file_id")
                    if action == "pick":
                        if not isinstance(file_id, int) or isinstance(file_id, bool):
                            continue
                    elif file_id is not None:
                        continue  # accept/mark must not have file_id
                    # member_fingerprint is optional string (added for rebind safety)
                    member_fp = value.get("member_fingerprint")
                    if member_fp is not None and not isinstance(member_fp, str):
                        continue
                    validated_groups[gid_str] = value
                self._data["groups"] = validated_groups
            except (OSError, ValueError):
                pass

    def _save(self) -> None:
        """Atomic write: temp file + rename. Caller must hold lock."""
        payload = json.dumps(self._data, ensure_ascii=False, indent=2).encode("utf-8")
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.output_dir, prefix=".review_state_", suffix=".tmp",
                delete=False
            ) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            tmp_path.replace(self.path)
        except Exception:
            # P0-8: Clean up temp file on failure
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def get_group(self, group_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve human decision for one group, or None. Returns immutable copy."""
        with self._lock:
            value = self._data["groups"].get(str(group_id))
            return dict(value) if value else None

    def set_group(self, group_id: int, decision: Dict[str, Any], member_fingerprint: Optional[str] = None) -> None:
        """Record one group's human decision and persist atomically.

        Args:
            group_id: Group ID to save decision for
            decision: Decision dict with action/timestamp/file_id
            member_fingerprint: SHA256 prefix of sorted member file_ids (prevents rebind after stage2 rerun)
        """
        if not isinstance(group_id, int) or isinstance(group_id, bool) or group_id <= 0:
            raise ValueError("group_id must be positive int")
        if not isinstance(decision, dict):
            raise ValueError("decision must be dict")
        action = decision.get("action")
        if action not in ("accept", "pick", "mark"):
            raise ValueError(f"action must be accept/pick/mark, got {action}")
        timestamp = decision.get("timestamp")
        if timestamp is not None and (not isinstance(timestamp, int) or isinstance(timestamp, bool)):
            raise ValueError("timestamp must be int if present")
        file_id = decision.get("file_id")
        if action == "pick":
            if not isinstance(file_id, int) or isinstance(file_id, bool):
                raise ValueError("pick requires int file_id")
        elif file_id is not None:
            raise ValueError(f"{action} must not have file_id")

        record = dict(decision)
        if member_fingerprint is not None:
            record["member_fingerprint"] = member_fingerprint

        with self._lock:
            self._data["groups"][str(group_id)] = record
            self._save()

    def clear_group(self, group_id: int) -> None:
        """Remove human decision for one group."""
        with self._lock:
            self._data["groups"].pop(str(group_id), None)
            self._save()

    def summary(self) -> Dict[str, Any]:
        """Return counts for display. Returns immutable copy."""
        with self._lock:
            groups = self._data["groups"]
            reviewed = sum(1 for g in groups.values() if g.get("action") in ("accept", "pick"))
            marked = sum(1 for g in groups.values() if g.get("action") == "mark")
            return {"total": len(groups), "reviewed": reviewed, "marked": marked}

    def current_page_states(self, group_ids: list[int]) -> Dict[int, Dict[str, Any]]:
        """Return state for specific groups (scoped to current page). Returns immutable copy."""
        with self._lock:
            return {gid: dict(self._data["groups"][str(gid)])
                    for gid in group_ids if str(gid) in self._data["groups"]}

    def snapshot(self) -> Dict[str, Any]:
        """Return full state snapshot for cross-page operations. Returns immutable copy."""
        with self._lock:
            return {
                "groups": {k: dict(v) for k, v in self._data["groups"].items()},
                "version": self._data["version"],
            }

    def validate_and_prune_stale(self, conn: sqlite3.Connection, scope: Any) -> int:
        """Remove decisions whose member_fingerprint doesn't match current group members.

        Fail closed: if old state has no fingerprint, ignore it with warning.
        Returns number of pruned entries.
        """
        pruned_count = 0
        with self._lock:
            stale_gids = []
            for gid_str, value in list(self._data["groups"].items()):
                gid = int(gid_str)
                stored_fp = value.get("member_fingerprint")

                # Fail closed: no fingerprint = legacy state, can't validate safety
                if stored_fp is None:
                    print(f"[review_state] WARNING: group {gid} decision has no member_fingerprint, ignoring (stage2 rerun may have reassigned this group_id)")
                    stale_gids.append(gid_str)
                    continue

                # Fetch current members from DB
                predicate, params = db.scope_sql(scope, "f")
                rows = conn.execute(
                    f"""
                    SELECT gm.file_id
                    FROM group_members gm
                    JOIN files f ON f.id = gm.file_id
                    WHERE gm.group_id = :gid AND {predicate}
                    ORDER BY gm.file_id
                    """,
                    {**params, "gid": gid}
                ).fetchall()

                if not rows:
                    # Group no longer exists in scope
                    stale_gids.append(gid_str)
                    continue

                current_fp = compute_member_fingerprint([r["file_id"] for r in rows])
                if current_fp != stored_fp:
                    print(f"[review_state] WARNING: group {gid} member_fingerprint mismatch (stored={stored_fp}, current={current_fp}), ignoring stale decision")
                    stale_gids.append(gid_str)

            for gid_str in stale_gids:
                self._data["groups"].pop(gid_str)
                pruned_count += 1

            if pruned_count > 0:
                self._backup_before_prune()

        return pruned_count

    def _backup_before_prune(self) -> None:
        """Snapshot the on-disk state before stale entries can be overwritten.

        Pruning is in-memory only; this keeps a one-time copy of the user's
        original decisions so a stage2 rerun can never destroy them.
        """
        if not self.path.is_file():
            return
        backup = self.output_dir / "review_state.pre-prune.json"
        if backup.exists():
            return
        try:
            backup.write_bytes(self.path.read_bytes())
            print(f"[review_state] original decisions backed up to {backup.name}")
        except OSError as exc:
            print(f"[review_state] WARNING: could not back up {self.path.name}: {exc}")


def validate_action(
    conn: sqlite3.Connection,
    group_id: Any,
    file_id: Any,
    action: str,
    scope: Any,
    _lock: Optional[threading.Lock] = None,
) -> Optional[str]:
    """Validate that group/file_id belong to scope and group.

    Returns error message or None. Never opens an original.
    """
    # Strict payload validation
    if not isinstance(group_id, int) or isinstance(group_id, bool):
        return "group_id must be int"
    if action not in ("accept", "pick", "mark", "clear"):
        return f"invalid action: {action}"

    # accept/mark/clear must not have file_id
    if action in ("accept", "mark", "clear"):
        if file_id is not None:
            return f"{action} action must not include file_id"

    # pick requires file_id that is a real int
    if action == "pick":
        if file_id is None:
            return "pick action requires file_id"
        if not isinstance(file_id, int) or isinstance(file_id, bool):
            return "file_id must be int"

    # Check group exists and is in scope (use lock if provided)
    predicate, params = db.scope_sql(scope, "f")
    if _lock:
        with _lock:
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
    else:
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
        if _lock:
            with _lock:
                member_row = conn.execute(
                    f"""
                    SELECT 1 FROM group_members gm
                    JOIN files f ON f.id = gm.file_id
                    WHERE gm.group_id = :group_id AND gm.file_id = :file_id AND {predicate}
                    """,
                    {**params, "group_id": int(group_id), "file_id": int(file_id)},
                ).fetchone()
        else:
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
