"""Human review state: never touches originals, writes only derived data.

All decisions are recorded in ``output/review_state.json``, atomically written
every time. Original photos stay read-only; the AI decisions/manifests remain
unchanged. This is a parallel, independent layer.

Two derived concepts live here as well:

* **Queues.** A group is ``PENDING`` while it carries no human decision,
  ``LATER`` once it is marked, and ``DONE`` after accept/pick. The review
  workbench walks the PENDING queue; the other two are re-entry lists.
* **Undo.** Every decision made through :meth:`ReviewState.apply_decision` /
  :meth:`ReviewState.revert_decision` also appends
  ``{group_id, before, after, focus_file_id}`` to a bounded history in the same
  atomic write, so ``undo`` restores the exact previous state of that group --
  including "no decision at all" -- and survives a page reload or a server
  restart. ``focus_file_id`` is the photo that was actually on screen when the
  decision was made, which is what undo has to come back to: ``accept`` and
  ``mark`` name no photo themselves, so without it undo would land on whichever
  member happens to sort first. History is capped at :data:`HISTORY_LIMIT`
  entries, and every step is re-validated against the group's current members at
  startup (:meth:`ReviewState.validate_and_prune_stale`) so a recycled
  ``group_id`` can never let undo resurrect a decision onto different photos.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import db

_SCHEMA_VERSION = 1
ACTIONS = ("accept", "pick", "mark")
QUEUES = ("PENDING", "LATER", "DONE")
HISTORY_LIMIT = 200


def compute_member_fingerprint(member_file_ids: list[int]) -> str:
    """Compute stable fingerprint of group members to detect group_id reassignment."""
    sorted_ids = sorted(member_file_ids)
    payload = ",".join(str(fid) for fid in sorted_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def queue_of(decision: Optional[Dict[str, Any]]) -> str:
    """Queue a group with ``decision`` belongs to (``None`` -> PENDING)."""
    action = (decision or {}).get("action")
    if action in ("accept", "pick"):
        return "DONE"
    if action == "mark":
        return "LATER"
    return "PENDING"


def _valid_decision(value: Any) -> bool:
    """Shape check for one stored decision (used on load and for history)."""
    if not isinstance(value, dict):
        return False
    if value.get("action") not in ACTIONS:
        return False
    timestamp = value.get("timestamp")
    if timestamp is not None and (not isinstance(timestamp, int)
                                  or isinstance(timestamp, bool)):
        return False
    file_id = value.get("file_id")
    if value.get("action") == "pick":
        if not isinstance(file_id, int) or isinstance(file_id, bool):
            return False
    elif file_id is not None:
        return False
    member_fp = value.get("member_fingerprint")
    return member_fp is None or isinstance(member_fp, str)


def _positive_int(value: Any) -> bool:
    """True for a real positive int (``bool`` is not an id)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validated_history(raw: Any) -> List[Dict[str, Any]]:
    """Keep only well-formed undo steps; a corrupt tail must not block undo."""
    if not isinstance(raw, list):
        return []
    history: List[Dict[str, Any]] = []
    for entry in raw[-HISTORY_LIMIT:]:
        if not isinstance(entry, dict):
            continue
        group_id = entry.get("group_id")
        if not _positive_int(group_id):
            continue
        before, after = entry.get("before"), entry.get("after")
        if before is not None and not _valid_decision(before):
            continue
        if after is not None and not _valid_decision(after):
            continue
        if before is None and after is None:
            continue
        timestamp = entry.get("timestamp")
        if timestamp is not None and (not isinstance(timestamp, int)
                                      or isinstance(timestamp, bool)):
            continue
        # A malformed focus is dropped on its own: the step is still a valid thing
        # to undo, we just no longer know which photo to return to.
        focus = entry.get("focus_file_id")
        history.append({"group_id": group_id, "before": before, "after": after,
                        "timestamp": timestamp,
                        "focus_file_id": focus if _positive_int(focus) else None})
    return history


class ReviewState:
    """Thread-safe review state backed by atomic JSON writes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.path = self.output_dir / "review_state.json"
        self._data: Dict[str, Any] = {"version": _SCHEMA_VERSION, "groups": {},
                                      "history": []}
        self._lock = threading.RLock()
        # Set when the file on disk could not be understood. The first write then
        # moves that file aside instead of replacing it -- see :meth:`_save`.
        self.load_error: Optional[str] = None
        self.quarantined_as: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load existing state. An unreadable file is kept and flagged, not lost.

        A file this build cannot understand -- wrong ``version``, malformed or
        truncated JSON, wrong top-level shape -- is no longer silently ignored:
        ``load_error`` is set, the first write renames the original to
        ``review_state.corrupt-<timestamp>.json``, and the server reports the
        same message through ``/api/status`` so the UI can show it. Losing a
        reviewer's decisions to a failed read is not an acceptable outcome.
        """
        with self._lock:
            if not self.path.is_file():
                return
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except (OSError, ValueError) as exc:
                self.load_error = (
                    f"{self.path.name} could not be read ({type(exc).__name__}); "
                    "the file is kept and this review starts with no decisions"
                )
                return
            if not isinstance(raw, dict):
                self.load_error = f"{self.path.name} is not a JSON object; file kept"
                return
            version = raw.get("version")
            if version != _SCHEMA_VERSION:
                self.load_error = (
                    f"{self.path.name} is version {version!r}, this build reads "
                    f"version {_SCHEMA_VERSION}; file kept"
                )
                return
            groups = raw.get("groups", {})
            if not isinstance(groups, dict):
                self.load_error = (f"{self.path.name} has a malformed 'groups' "
                                   "section; file kept")
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
                if action not in ACTIONS:
                    continue
                # Timestamp must be int, not bool
                timestamp = value.get("timestamp")
                if timestamp is not None and (not isinstance(timestamp, int)
                                              or isinstance(timestamp, bool)):
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
            self._data["history"] = _validated_history(raw.get("history"))

    def warning(self) -> Optional[str]:
        """Human-readable load problem for the UI, or None. Thread-safe."""
        with self._lock:
            if self.load_error is None:
                return None
            if self.quarantined_as:
                return f"{self.load_error} (saved as {self.quarantined_as})"
            return self.load_error

    def _quarantine_unreadable(self) -> None:
        """Move an unreadable state file aside once, before the first write.

        Caller holds the lock. A rename (not a copy) is used so the bytes are
        never read again. If the rename fails, this **raises**: the whole point
        of the quarantine is that the reviewer's unreadable (or newer-version)
        file is preserved, so carrying on and overwriting it would break exactly
        the promise being made. The decision is refused instead, the original
        keeps its bytes, and the next attempt tries the quarantine again.
        """
        if self.load_error is None or self.quarantined_as is not None:
            return
        stamp = int(time.time())
        target = self.output_dir / f"review_state.corrupt-{stamp}.json"
        index = 0
        while target.exists():
            index += 1
            target = self.output_dir / f"review_state.corrupt-{stamp}-{index}.json"
        try:
            self.path.replace(target)
        except OSError as exc:
            raise OSError(
                f"refusing to overwrite {self.path.name}: it could not be read "
                f"and could not be set aside as {target.name} ({exc}). The file "
                "is unchanged; fix the permissions or move it yourself, then "
                "retry."
            ) from exc
        self.quarantined_as = target.name
        print(f"[review_state] WARNING: {self.load_error}; kept as {target.name}")

    def _save(self) -> None:
        """Durable atomic write: temp file, fsync, rename, fsync directory.

        Caller must hold the lock. The fsync pair is what makes the rename
        meaningful after a power loss: without it the rename can land while the
        temp file's contents have not, leaving an empty or partial state file.
        Directory fsync is best-effort -- it is not supported everywhere (Windows)
        and its failure must not fail the write.
        """
        self._quarantine_unreadable()
        # Compact separators: this file is rewritten on every keystroke-driven
        # decision, and indentation roughly triples it for no reader benefit.
        payload = json.dumps(self._data, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.output_dir, prefix=".review_state_", suffix=".tmp",
                delete=False
            ) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            tmp_path.replace(self.path)
            self._fsync_dir()
        except Exception:
            # P0-8: Clean up temp file on failure
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _fsync_dir(self) -> None:
        """Best-effort durability for the rename itself."""
        try:
            fd = os.open(self.output_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def get_group(self, group_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve human decision for one group, or None. Returns immutable copy."""
        with self._lock:
            value = self._data["groups"].get(str(group_id))
            return dict(value) if value else None

    def set_group(self, group_id: int, decision: Dict[str, Any], member_fingerprint: Optional[str] = None) -> None:
        """Record one group's human decision and persist atomically (no undo entry).

        Args:
            group_id: Group ID to save decision for
            decision: Decision dict with action/timestamp/file_id
            member_fingerprint: SHA256 prefix of sorted member file_ids (prevents rebind after stage2 rerun)
        """
        record = self._record(group_id, decision, member_fingerprint)
        with self._lock, self._transaction(group_id):
            self._data["groups"][str(group_id)] = record
            self._save()

    @staticmethod
    def _record(group_id: int, decision: Dict[str, Any],
                member_fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """Validate one decision and return the record to store."""
        if not isinstance(group_id, int) or isinstance(group_id, bool) or group_id <= 0:
            raise ValueError("group_id must be positive int")
        if not isinstance(decision, dict):
            raise ValueError("decision must be dict")
        action = decision.get("action")
        if action not in ACTIONS:
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
        return record

    def clear_group(self, group_id: int) -> None:
        """Remove human decision for one group (no undo entry)."""
        with self._lock, self._transaction(group_id):
            self._data["groups"].pop(str(group_id), None)
            self._save()

    @contextmanager
    def _transaction(self, *group_ids: int):
        """Roll the in-memory state back when the write does not land.

        Every mutation edits ``_data`` and then calls :meth:`_save`. Without this,
        a failed fsync or rename leaves the *old* file on disk but the *new*
        decision in memory: the API would report a decision that was never
        stored, and the next successful write would smuggle it onto disk.

        Only what a mutation can touch is copied: the affected group entries (a
        handful of small dicts, replaced wholesale, never edited in place) and the
        history list, which is bounded to :data:`HISTORY_LIMIT`. Caller holds the
        lock.
        """
        groups = self._data["groups"]
        saved_groups = {str(gid): groups.get(str(gid)) for gid in group_ids}
        saved_history = list(self._data.get("history", []))
        try:
            yield
        except BaseException:
            for key, value in saved_groups.items():
                if value is None:
                    groups.pop(key, None)
                else:
                    groups[key] = value
            self._data["history"] = saved_history
            raise

    # --- undoable operations ------------------------------------------------

    def _append_history(self, group_id: int, before: Optional[Dict[str, Any]],
                        after: Optional[Dict[str, Any]],
                        focus_file_id: Optional[int] = None) -> None:
        """Record one reversible step. Caller holds the lock and saves.

        ``focus_file_id`` is the photo that was on screen, kept beside the step
        rather than inside the decision: ``accept``/``mark`` must not carry a
        ``file_id``, which would read as "the user chose this one".
        """
        history = self._data.setdefault("history", [])
        history.append({"group_id": int(group_id), "before": before,
                        "after": after, "timestamp": int(time.time()),
                        "focus_file_id": (int(focus_file_id)
                                          if _positive_int(focus_file_id) else None)})
        if len(history) > HISTORY_LIMIT:
            del history[:-HISTORY_LIMIT]

    def apply_decision(self, group_id: int, decision: Dict[str, Any],
                       member_fingerprint: Optional[str] = None,
                       focus_file_id: Optional[int] = None) -> Dict[str, Any]:
        """Record a decision *and* the step needed to undo it, in one write."""
        record = self._record(group_id, decision, member_fingerprint)
        if focus_file_id is not None and not _positive_int(focus_file_id):
            raise ValueError("focus_file_id must be a positive int if present")
        with self._lock, self._transaction(group_id):
            previous = self._data["groups"].get(str(group_id))
            before = dict(previous) if previous else None
            self._data["groups"][str(group_id)] = record
            self._append_history(group_id, before, dict(record), focus_file_id)
            self._save()
            return dict(record)

    def revert_decision(self, group_id: int,
                        focus_file_id: Optional[int] = None) -> bool:
        """Drop a group's decision as an undoable step. True if one existed.

        Clearing a group that had no decision changes nothing, so it records no
        step: a ``{before: None, after: None}`` entry is not something ``undo``
        can act on, and letting them accumulate would make U walk back through
        no-ops instead of the reviewer's actual decisions.
        """
        if not _positive_int(group_id):
            raise ValueError("group_id must be positive int")
        if focus_file_id is not None and not _positive_int(focus_file_id):
            raise ValueError("focus_file_id must be a positive int if present")
        with self._lock, self._transaction(group_id):
            previous = self._data["groups"].pop(str(group_id), None)
            if previous is None:
                return False
            self._append_history(group_id, dict(previous), None, focus_file_id)
            self._save()
            return True

    def undo_last(self, allowed: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Restore the group touched by the most recent recorded step.

        ``allowed(group_id)`` (optional) filters entries whose group has left the
        current scope: those are discarded, and the search continues, so undo can
        never resurrect another root's decision. Returns the reversal that was
        applied, or ``None`` when nothing is left to undo.

        A failed write rolls back completely: neither the restored decision nor
        the consumed history survives in memory, so U can simply be pressed again.
        """
        with self._lock:
            history = self._data.get("history", [])
            if not history:
                return None
            # The step that will be applied decides which group is copied; the
            # transaction covers the whole walk, including discarded entries.
            touched = {int(entry["group_id"]) for entry in history}
            with self._transaction(*touched):
                changed = False
                while self._data["history"]:
                    entry = self._data["history"].pop()
                    changed = True
                    group_id = int(entry["group_id"])
                    if allowed is not None and not allowed(group_id):
                        continue
                    before = entry.get("before")
                    if before is None:
                        self._data["groups"].pop(str(group_id), None)
                    else:
                        self._data["groups"][str(group_id)] = dict(before)
                    # Return to the photo that was on screen. A history entry
                    # written without one (older client) falls back to the
                    # decision's own file_id, which only ``pick`` has.
                    focus = entry.get("focus_file_id")
                    if not _positive_int(focus):
                        focus = (entry.get("after") or {}).get("file_id")
                    self._save()
                    return {
                        "group_id": group_id,
                        "restored": dict(before) if before else None,
                        "undone": dict(entry["after"]) if entry.get("after") else None,
                        "queue": queue_of(before),
                        "focus_file_id": focus if _positive_int(focus) else None,
                    }
                if changed:
                    # Every remaining step belonged to another scope; that pruning
                    # is itself a change worth persisting.
                    self._save()
                return None

    def history_depth(self) -> int:
        """How many steps ``undo_last`` could still walk back."""
        with self._lock:
            return len(self._data.get("history", []))

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

    def queue_tally(self, group_ids: Any) -> Dict[str, Any]:
        """Classify ``group_ids`` into queues in one pass, under one lock.

        Returns ``{"counts": {...}, "queues": {group_id: queue}}`` built from the
        ``action`` field alone. This is what the paging path needs, and it avoids
        copying every stored decision (:meth:`snapshot`) just to read one key off
        each -- which is the difference between cheap and quadratic-feeling on a
        library with tens of thousands of groups.
        """
        counts = {name: 0 for name in QUEUES}
        placement: Dict[int, str] = {}
        with self._lock:
            groups = self._data["groups"]
            for group_id in group_ids:
                action = (groups.get(str(group_id)) or {}).get("action")
                name = ("DONE" if action in ("accept", "pick")
                        else "LATER" if action == "mark" else "PENDING")
                counts[name] += 1
                placement[int(group_id)] = name
        return {"counts": counts, "queues": placement}

    def snapshot(self) -> Dict[str, Any]:
        """Return full state snapshot for cross-page operations. Returns immutable copy."""
        with self._lock:
            return {
                "groups": {k: dict(v) for k, v in self._data["groups"].items()},
                "version": self._data["version"],
                "history_depth": len(self._data.get("history", [])),
            }

    def validate_and_prune_stale(self, conn: sqlite3.Connection, scope: Any) -> int:
        """Drop decisions *and* undo steps that no longer describe real groups.

        Stage 2 can hand a recycled ``group_id`` to a completely different set of
        photos, so every recorded ``member_fingerprint`` is re-checked against the
        group's current in-scope members at startup. Two things are pruned:

        * **Decisions**, as before: a mismatch (or a legacy entry with no
          fingerprint at all) is ignored rather than shown against new photos.
        * **Undo steps**, checked *independently*. A step's ``before`` and
          ``after`` each carry the fingerprint of the decision they hold, and a
          step outlives its decision: ``pick`` then ``clear`` leaves no current
          decision but a history entry whose ``before`` still holds the old one.
          Validating only current decisions would leave that step in place, and
          undoing the ``clear`` after a regroup would resurrect a decision onto
          members it was never about.

        Fail closed in both cases: anything that cannot be proven to match the
        current members is discarded.

        Returns the number of pruned *decisions* (the count the server reports).
        """
        pruned_count = 0
        fingerprints: Dict[int, Optional[str]] = {}
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

                current_fp = self._current_fingerprint(conn, scope, gid, fingerprints)
                if current_fp is None:
                    # Group no longer exists in scope
                    stale_gids.append(gid_str)
                    continue

                if current_fp != stored_fp:
                    print(f"[review_state] WARNING: group {gid} member_fingerprint mismatch (stored={stored_fp}, current={current_fp}), ignoring stale decision")
                    stale_gids.append(gid_str)

            for gid_str in stale_gids:
                self._data["groups"].pop(gid_str)
                pruned_count += 1

            # Undo steps are validated on their own, not by whether their group
            # still holds a decision.
            dropped = {int(gid) for gid in stale_gids}
            kept: List[Dict[str, Any]] = []
            dropped_steps = 0
            for entry in self._data.get("history", []):
                if (int(entry["group_id"]) in dropped
                        or not self._step_is_current(conn, scope, entry, fingerprints)):
                    dropped_steps += 1
                    continue
                kept.append(entry)
            if dropped_steps:
                print(f"[review_state] WARNING: discarded {dropped_steps} undo "
                      "step(s) whose group members no longer match (stage2 rerun "
                      "may have reassigned these group_ids)")
            self._data["history"] = kept

            if pruned_count > 0 or dropped_steps > 0:
                self._backup_before_prune()

        return pruned_count

    @staticmethod
    def _current_fingerprint(conn: sqlite3.Connection, scope: Any, group_id: int,
                             cache: Dict[int, Optional[str]]) -> Optional[str]:
        """Fingerprint of ``group_id``'s current in-scope members, or None.

        Cached per call, so validating many steps of one group costs one query.
        """
        if group_id in cache:
            return cache[group_id]
        predicate, params = db.scope_sql(scope, "f")
        rows = conn.execute(
            f"""
            SELECT gm.file_id
            FROM group_members gm
            JOIN files f ON f.id = gm.file_id
            WHERE gm.group_id = :gid AND {predicate}
            ORDER BY gm.file_id
            """,
            {**params, "gid": int(group_id)},
        ).fetchall()
        value = (compute_member_fingerprint([row["file_id"] for row in rows])
                 if rows else None)
        cache[group_id] = value
        return value

    def _step_is_current(self, conn: sqlite3.Connection, scope: Any,
                         entry: Dict[str, Any],
                         cache: Dict[int, Optional[str]]) -> bool:
        """True when every decision this undo step holds still fits its group."""
        recorded = [side for side in (entry.get("before"), entry.get("after"))
                    if isinstance(side, dict)]
        if not recorded:
            return False
        current_fp = self._current_fingerprint(conn, scope,
                                               int(entry["group_id"]), cache)
        if current_fp is None:
            return False
        # Fail closed on a legacy step: no fingerprint means it cannot be proven
        # to describe these members, and undo must not gamble on it.
        return all(side.get("member_fingerprint") == current_fp for side in recorded)

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
    context_file_id: Any = None,
) -> Optional[str]:
    """Validate that group/file_id belong to scope and group.

    ``context_file_id`` is the photo the reviewer had on screen. It is validated
    exactly like a ``pick`` target -- real int, member of this group, in scope --
    but separately from ``file_id``, because it carries no decision meaning: it
    only tells undo where to come back to. Absent (older client) is fine.

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

    if context_file_id is not None and not _positive_int(context_file_id):
        return "context_file_id must be a positive int"

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
    if action == "pick" and not _is_member(conn, group_id, file_id, predicate,
                                           params, _lock):
        return "file_id not a member of this group"

    # The on-screen photo gets the same membership check, on its own, so a crafted
    # id can never be stored as "where the reviewer was".
    if context_file_id is not None and not _is_member(conn, group_id, context_file_id,
                                                     predicate, params, _lock):
        return "context_file_id not a member of this group"

    return None


def _is_member(conn: sqlite3.Connection, group_id: int, file_id: int,
               predicate: str, params: Dict[str, Any],
               _lock: Optional[threading.Lock] = None) -> bool:
    """True when ``file_id`` is an in-scope member of ``group_id``."""
    sql = f"""
        SELECT 1 FROM group_members gm
        JOIN files f ON f.id = gm.file_id
        WHERE gm.group_id = :group_id AND gm.file_id = :file_id AND {predicate}
        """
    query = {**params, "group_id": int(group_id), "file_id": int(file_id)}
    if _lock:
        with _lock:
            return conn.execute(sql, query).fetchone() is not None
    return conn.execute(sql, query).fetchone() is not None
