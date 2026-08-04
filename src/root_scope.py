r"""Durable photo-root identity + current-run scope for one output directory.

The bug this module exists to prevent
-------------------------------------
An output directory (``inventory.sqlite`` + ``thumbs/`` + the reports) used to
carry no record of *which* photo root produced it. Pointing a second run at the
same ``--output`` with a different ``--root`` therefore silently mixed
libraries: Stage 0 added the new files next to the old ones, Stages 1-3 queried
the whole table, and the review/report counts described E:\Photos\2015 +
E:\Photos\2017 + E:\Photos\2026 while the user believed they were looking at
2026 only.

The fix has two halves, and both are needed:

* **Durable identity.** Every ``files`` row stores a normalised ``root_key``
  and ``path_key``. The output directory records the root it was bound to. A
  run whose root disagrees with that binding **fails closed** with
  :class:`ParameterError` *before anything is written* -- no deletion, no
  rebinding, no "helpful" cleanup of the user's existing data.
* **Current-run scope.** Even inside one root, every downstream query filters
  on the normalised path prefix, and Stage 0 stamps ``last_run_id`` on the rows
  it saw this run, so reports can state what belongs to the current run.

Normalisation rules (Windows-first, because that is the target platform)
-----------------------------------------------------------------------
``E:\Photos\2026``, ``E:/Photos/2026/`` and ``e:\photos\2026`` are one root.
Separators collapse to ``/``; a trailing separator is dropped; on
case-insensitive filesystems (Windows, macOS) the whole key is casefolded, and
a Windows-looking path is always treated that way regardless of the host OS so
a DB written on Windows stays comparable. Prefix comparisons always append
``/``, so ``E:/Photos/2026`` is never a parent of ``E:/Photos/2026extra``.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

# meta keys holding the binding for this output directory.
META_ROOT_KEY = "scope_root_key"
META_ROOT_PATH = "scope_root_path"
META_BOUND_AT = "scope_bound_at"
META_RUN_ID = "scope_run_id"
META_RUN_STARTED_AT = "scope_run_started_at"
META_SCHEMA = "scope_schema"
SCOPE_SCHEMA = "1"

# How many example out-of-scope rows a diagnostic lists.
EXAMPLE_ROWS = 5


class ParameterError(ValueError):
    """Invalid combination of parameters. Raised before any mutation."""


# --- normalisation ---------------------------------------------------------

def _windows_like(text: str) -> bool:
    """True when the string looks like a Windows path (drive letter or UNC)."""
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return True
    return text.startswith("\\\\") or text.startswith("//")


def normalize(path: str | os.PathLike) -> str:
    """Normalised comparison key for a root or a file path.

    Never touches the filesystem (a key must be computable for a path recorded
    on another machine, or for a drive that is currently unplugged).
    """
    raw = str(path).strip().strip("\x00")
    if not raw:
        raise ParameterError("path must not be empty")
    windows = _windows_like(raw) or os.name == "nt"
    if windows:
        pure: Any = PureWindowsPath(raw)
        text = str(pure).replace("\\", "/")
    else:
        pure = PurePosixPath(raw)
        text = str(pure)
    # Collapse duplicate separators without eating a UNC/POSIX root prefix.
    lead = "//" if text.startswith("//") else ("/" if text.startswith("/") else "")
    body = "/".join(part for part in text.split("/") if part)
    text = lead + body
    if len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    if windows or _case_insensitive():
        return text.casefold()
    # POSIX: paths are case-sensitive, so the literal text is the key.
    return text


def _case_insensitive() -> bool:
    """True on filesystems that compare names case-insensitively by default."""
    return os.name == "nt" or sys.platform == "darwin"


def prefix_of(key: str) -> str:
    """Directory prefix used for containment tests (always ends with ``/``)."""
    return key if key.endswith("/") else key + "/"


def contains(root_key: str, candidate_key: str) -> bool:
    """True when ``candidate_key`` is ``root_key`` itself or below it."""
    return candidate_key == root_key or candidate_key.startswith(prefix_of(root_key))


def relation(requested_key: str, recorded_key: str) -> str:
    """How the requested root relates to the recorded one.

    ``same`` | ``child`` (requested is inside recorded) | ``parent``
    (requested contains recorded) | ``disjoint``.
    """
    if requested_key == recorded_key:
        return "same"
    if contains(recorded_key, requested_key):
        return "child"
    if contains(requested_key, recorded_key):
        return "parent"
    return "disjoint"


_RELATION_TEXT = {
    "child": "the recorded root is a PARENT of the requested root",
    "parent": "the requested root is a PARENT of the recorded root",
    "disjoint": "the two roots are unrelated",
}


def new_run_id() -> str:
    return uuid.uuid4().hex


# --- scope object ----------------------------------------------------------

@dataclass(frozen=True)
class RootScope:
    """The root this output directory is bound to, plus the current run id."""

    key: Optional[str]
    path: Optional[str]
    run_id: Optional[str] = None

    @property
    def bound(self) -> bool:
        return bool(self.key)

    def contains_path(self, path: str | os.PathLike) -> bool:
        if not self.bound:
            return True
        return contains(str(self.key), normalize(path))

    def clause(self, alias: str = "f", name: str = "scope") -> Tuple[str, Dict[str, Any]]:
        """SQL predicate + named params restricting rows to this root.

        Unbound scope yields the constant ``1`` so callers can always inline it.
        A row with a NULL ``path_key`` (impossible after migration, but cheap to
        be explicit about) is treated as out of scope rather than silently kept.
        """
        if not self.bound:
            return "1", {}
        prefix = prefix_of(str(self.key))
        return (
            f"(COALESCE({alias}.path_key, '') = :{name}_key"
            f" OR substr(COALESCE({alias}.path_key, ''), 1, :{name}_len) = :{name}_prefix)",
            {f"{name}_key": self.key, f"{name}_len": len(prefix), f"{name}_prefix": prefix},
        )

    def describe(self) -> str:
        if not self.bound:
            return "unbound (legacy output directory)"
        return f"{self.path} [key={self.key}]"


UNBOUND = RootScope(key=None, path=None, run_id=None)


# --- schema ----------------------------------------------------------------

def migrate(conn: sqlite3.Connection) -> Dict[str, int]:
    """Add identity columns and backfill them. Never deletes a row.

    Safe to call on every :func:`src.db.open_db`; the backfill only touches rows
    whose ``path_key`` is still NULL.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    for column in ("path_key", "root_key", "last_run_id"):
        if column not in existing:
            conn.execute(f"ALTER TABLE files ADD COLUMN {column} TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path_key ON files(path_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_root_key ON files(root_key)")
    pending = conn.execute(
        "SELECT id, path FROM files WHERE path_key IS NULL AND path IS NOT NULL"
    ).fetchall()
    updates = []
    for row in pending:
        try:
            updates.append((normalize(row[1]), int(row[0])))
        except ParameterError:
            continue
    if updates:
        conn.executemany("UPDATE files SET path_key = ? WHERE id = ?", updates)
    return {"backfilled": len(updates)}


# --- binding ---------------------------------------------------------------

def recorded(conn: sqlite3.Connection) -> RootScope:
    """The root binding stored in this output directory (unbound if absent)."""
    from . import db

    key = db.get_meta(conn, META_ROOT_KEY)
    if not key:
        # Pre-scope databases recorded only a display root; adopt it as a hint
        # but do not treat it as a binding (its normalisation was never checked).
        return RootScope(key=None, path=db.get_meta(conn, "stage0_root"), run_id=None)
    return RootScope(key=key, path=db.get_meta(conn, META_ROOT_PATH), run_id=None)


def out_of_scope_count(conn: sqlite3.Connection, root_key: str) -> int:
    scope = RootScope(key=root_key, path=None)
    predicate, params = scope.clause("f")
    return int(conn.execute(
        f"SELECT COUNT(*) AS n FROM files f WHERE NOT {predicate}", params
    ).fetchone()["n"])


def out_of_scope_examples(conn: sqlite3.Connection, root_key: str,
                          limit: int = EXAMPLE_ROWS) -> List[str]:
    scope = RootScope(key=root_key, path=None)
    predicate, params = scope.clause("f")
    rows = conn.execute(
        f"SELECT path FROM files f WHERE NOT {predicate} ORDER BY f.id LIMIT :limit",
        {**params, "limit": int(limit)},
    ).fetchall()
    return [str(row["path"]) for row in rows]


def _counts(conn: sqlite3.Connection) -> Tuple[int, int]:
    total = int(conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])
    return total, int(conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE file_kind IN ('jpg', 'jpg_motion')"
    ).fetchone()["n"])


def _mismatch_error(requested: RootScope, recorded_scope: RootScope,
                    conn: sqlite3.Connection, db_path: Optional[str]) -> ParameterError:
    total, stills = _counts(conn)
    kind = relation(str(requested.key), str(recorded_scope.key))
    lines = [
        "incompatible output directory: it is already bound to a different photo root.",
        f"  requested root : {requested.path}  [key={requested.key}]",
        f"  recorded root  : {recorded_scope.path}  [key={recorded_scope.key}]",
        f"  relation       : {_RELATION_TEXT.get(kind, kind)}",
        f"  existing data  : {total} inventory row(s), {stills} still image(s)"
        + (f" in {db_path}" if db_path else ""),
        "Reusing it would mix records from different roots into one report, which is how"
        " a 2026-only run previously reported 2015/2017 photos.",
        "Nothing was read, written, or deleted. Choose one:",
        "  * use a separate --output directory for this root (recommended), or",
        f"  * re-run with --root \"{recorded_scope.path}\" to resume that inventory, or",
        "  * move/rename the existing output directory if you no longer need it.",
    ]
    if kind in ("parent", "child"):
        lines.append(
            "Note: a parent/child root is still a different scope -- the reports,"
            " thumbnail cache and ETA would describe a library you did not ask for."
        )
    return ParameterError("\n".join(lines))


def _legacy_error(requested: RootScope, conn: sqlite3.Connection,
                  stale: int, db_path: Optional[str]) -> ParameterError:
    total, _stills = _counts(conn)
    examples = out_of_scope_examples(conn, str(requested.key))
    lines = [
        "incompatible output directory: it was created before root identity was"
        " recorded and still holds rows from outside the requested root.",
        f"  requested root : {requested.path}  [key={requested.key}]",
        f"  existing data  : {total} inventory row(s), {stale} of them outside that root"
        + (f" in {db_path}" if db_path else ""),
        "  examples       : " + ", ".join(examples) if examples else "  examples       : -",
        "Nothing was read, written, or deleted -- this older database is left exactly as"
        " it is, because deleting rows to make the numbers match would destroy work.",
        "Choose one:",
        "  * use a fresh --output directory for this root (recommended), or",
        "  * re-run with --root set to the common parent of the paths above to adopt"
        " the whole existing inventory.",
    ]
    return ParameterError("\n".join(lines))


def check(conn: sqlite3.Connection, root: str | os.PathLike,
          db_path: Optional[str] = None) -> RootScope:
    """Verify that ``root`` may use this output directory. Raises or returns.

    Read-only: this is the fail-closed gate, so it must never mutate anything.
    Returns the requested scope (without a run id) when compatible.
    """
    requested = RootScope(key=normalize(root), path=str(root))
    current = recorded(conn)
    if current.bound:
        if relation(str(requested.key), str(current.key)) != "same":
            raise _mismatch_error(requested, current, conn, db_path)
        return RootScope(key=current.key, path=current.path or requested.path)
    total, _stills = _counts(conn)
    if total:
        stale = out_of_scope_count(conn, str(requested.key))
        if stale:
            raise _legacy_error(requested, conn, stale, db_path)
    return requested


def preflight(db_path: str | os.PathLike, root: str | os.PathLike) -> RootScope:
    """Fail closed *before* the pipeline opens/creates anything else.

    A missing database is fine (first run for this output directory). An
    existing one is opened read-only so a rejected run cannot even upgrade the
    schema of a database that belongs to another root.
    """
    path = Path(db_path)
    requested = RootScope(key=normalize(root), path=str(root))
    if not path.is_file():
        return requested
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "files" not in tables or "meta" not in tables:
            return requested
        if "path_key" not in {row[1] for row in conn.execute("PRAGMA table_info(files)")}:
            return _preflight_legacy(conn, requested, str(path))
        return check(conn, root, db_path=str(path))
    finally:
        conn.close()


def _preflight_legacy(conn: sqlite3.Connection, requested: RootScope,
                      db_path: str) -> RootScope:
    """Read-only legacy check: normalise paths in Python, no schema change."""
    from . import db

    if db.get_meta(conn, META_ROOT_KEY):  # pragma: no cover - defensive
        return check(conn, str(requested.path), db_path=db_path)
    rows = conn.execute("SELECT path FROM files").fetchall()
    stale = 0
    examples: List[str] = []
    for row in rows:
        try:
            key = normalize(row["path"])
        except ParameterError:
            continue
        if not contains(str(requested.key), key):
            stale += 1
            if len(examples) < EXAMPLE_ROWS:
                examples.append(str(row["path"]))
    if not stale:
        return requested
    total = len(rows)
    raise ParameterError("\n".join([
        "incompatible output directory: it was created before root identity was"
        " recorded and still holds rows from outside the requested root.",
        f"  requested root : {requested.path}  [key={requested.key}]",
        f"  existing data  : {total} inventory row(s), {stale} of them outside that root"
        f" in {db_path}",
        "  examples       : " + (", ".join(examples) if examples else "-"),
        "Nothing was read, written, or deleted -- this older database is left exactly as"
        " it is, because deleting rows to make the numbers match would destroy work.",
        "Choose one:",
        "  * use a fresh --output directory for this root (recommended), or",
        "  * re-run with --root set to the common parent of the paths above to adopt"
        " the whole existing inventory.",
    ]))


def bind(conn: sqlite3.Connection, root: str | os.PathLike,
         run_id: Optional[str] = None, db_path: Optional[str] = None) -> RootScope:
    """Validate, then record the binding + a fresh run id. Stage 0 only.

    Same root -> resumable (the binding is simply confirmed). Different root ->
    :class:`ParameterError` from :func:`check`, before this writes anything.
    """
    from . import db

    verified = check(conn, root, db_path=db_path)
    scope = RootScope(key=verified.key, path=str(root), run_id=run_id or new_run_id())
    db.set_meta(conn, META_SCHEMA, SCOPE_SCHEMA)
    db.set_meta(conn, META_ROOT_KEY, str(scope.key))
    db.set_meta(conn, META_ROOT_PATH, str(scope.path))
    if not db.get_meta(conn, META_BOUND_AT):
        db.set_meta(conn, META_BOUND_AT, str(int(time.time())))
    db.set_meta(conn, META_RUN_ID, str(scope.run_id))
    db.set_meta(conn, META_RUN_STARTED_AT, str(int(time.time())))
    # Adopt rows that predate the binding but do belong to this root.
    predicate, params = scope.clause("files")
    conn.execute(
        f"UPDATE files SET root_key = :root_key WHERE root_key IS NULL AND {predicate}",
        {**params, "root_key": scope.key},
    )
    conn.commit()
    return scope


def resolve(conn: sqlite3.Connection, root: Optional[str | os.PathLike] = None,
            db_path: Optional[str] = None) -> RootScope:
    """Scope for a downstream stage: the recorded binding + the recorded run id.

    When ``root`` is given it is verified too, so ``stage1 --config`` pointed at
    a foreign output directory fails closed exactly like the one-command entry
    point does.
    """
    from . import db

    current = recorded(conn)
    if root is not None:
        verified = check(conn, root, db_path=db_path)
        current = RootScope(key=verified.key, path=verified.path)
    return RootScope(key=current.key, path=current.path,
                     run_id=db.get_meta(conn, META_RUN_ID))


# --- run bookkeeping -------------------------------------------------------

def mark_seen(conn: sqlite3.Connection, run_id: Optional[str],
              file_ids: Sequence[int]) -> int:
    """Stamp ``last_run_id`` on rows the current run actually visited."""
    if not run_id or not file_ids:
        return 0
    conn.executemany(
        "UPDATE files SET last_run_id = ? WHERE id = ?",
        [(str(run_id), int(value)) for value in file_ids],
    )
    return len(file_ids)


def summary(conn: sqlite3.Connection, scope: RootScope) -> Dict[str, Any]:
    """Counts that let a report state honestly what it is describing."""
    predicate, params = scope.clause("f")
    in_scope = int(conn.execute(
        f"SELECT COUNT(*) AS n FROM files f WHERE {predicate}", params).fetchone()["n"])
    total = int(conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])
    seen = 0
    if scope.run_id:
        seen = int(conn.execute(
            f"SELECT COUNT(*) AS n FROM files f WHERE {predicate} AND f.last_run_id = :run_id",
            {**params, "run_id": scope.run_id},
        ).fetchone()["n"])
    return {
        "root": scope.path,
        "root_key": scope.key,
        "run_id": scope.run_id,
        "files_in_scope": in_scope,
        "files_in_db": total,
        "files_out_of_scope": max(0, total - in_scope),
        "files_seen_this_run": seen,
    }


def scope_note(summary_payload: Dict[str, Any]) -> str:
    """One-line, honest description for logs and performance.txt."""
    root = summary_payload.get("root") or "unbound"
    note = (f"scope: root={root} run={summary_payload.get('run_id') or 'n/a'} "
            f"files_in_scope={summary_payload.get('files_in_scope', 0)} "
            f"seen_this_run={summary_payload.get('files_seen_this_run', 0)}")
    stale = int(summary_payload.get("files_out_of_scope", 0) or 0)
    if stale:
        note += f" (ignored_out_of_scope_rows={stale})"
    return note
