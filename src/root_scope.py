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

Where the pieces live
---------------------
* :mod:`src.root_identity` -- normalisation, :class:`RootScope`, the SQL clause.
  Pure: no database, no filesystem.
* :mod:`src.root_readonly` -- correctly escaped, side-effect-free read-only
  access (a rejected run must not even create ``-wal``/``-shm``).
* :mod:`src.root_diagnostics` -- the refusal messages and the read-only counting
  behind them.
* this module -- the binding stored in ``meta``, and the gates
  (:func:`preflight`, :func:`check`, :func:`resolve`, :func:`adopt_or_resolve`).

Binding is never implicit
-------------------------
Only Stage 0 (or an explicit ``--root``/declared ``paths.root``) may *create* a
binding. A defaulted config value must never bind an output directory: the
packaged default is ``E:/Photos``, so adopting it would silently claim a root the
user never named -- the same class of dishonesty as the original bug. Stages 1-3
therefore refuse to run against an unbound directory unless a root was actually
supplied (see :func:`adopt_or_resolve`).
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from . import root_diagnostics as diag
from . import root_readonly as ro
from .root_identity import (  # noqa: F401 (re-exported: this is the public entry point)
    UNBOUND,
    ParameterError,
    RootScope,
    contains,
    new_run_id,
    normalize,
    prefix_of,
    relation,
    requested_scope,
)
from .root_readonly import (  # noqa: F401 (re-exported for callers/tests)
    connect_readonly,
    readonly_uri,
)
from .root_diagnostics import (  # noqa: F401 (re-exported for callers/tests)
    EXAMPLE_ROWS,
    out_of_scope_count,
    out_of_scope_examples,
)

# meta keys holding the binding for this output directory.
META_ROOT_KEY = "scope_root_key"
META_ROOT_PATH = "scope_root_path"
META_BOUND_AT = "scope_bound_at"
META_RUN_ID = "scope_run_id"
META_RUN_STARTED_AT = "scope_run_started_at"
META_SCHEMA = "scope_schema"
SCOPE_SCHEMA = "1"


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


def check(conn: sqlite3.Connection, root: str | os.PathLike,
          db_path: Optional[str] = None) -> RootScope:
    """Verify that ``root`` may use this output directory. Raises or returns.

    Read-only: this is the fail-closed gate, so it must never mutate anything.
    Returns the requested scope (without a run id) when compatible.
    """
    requested = requested_scope(root)
    current = recorded(conn)
    if current.bound:
        if relation(str(requested.key), str(current.key)) != "same":
            raise diag.mismatch_error(requested, current, conn, db_path)
        return RootScope(key=current.key, path=current.path or requested.path)
    total, _stills = diag.counts(conn)
    if total:
        stale = diag.out_of_scope_count(conn, str(requested.key))
        if stale:
            raise diag.legacy_error(
                requested, total, stale,
                diag.out_of_scope_examples(conn, str(requested.key)), db_path,
            )
    return requested


def preflight(db_path: str | os.PathLike, root: str | os.PathLike) -> RootScope:
    """Fail closed *before* the pipeline opens/creates anything else.

    A missing database is fine (first run for this output directory). An existing
    one is opened read-only, without creating ``-wal``/``-shm``, so a rejected run
    cannot even upgrade the schema of a database that belongs to another root.
    """
    path = Path(db_path)
    requested = requested_scope(root)
    if not path.is_file():
        return requested
    conn = ro.connect_readonly(path)
    try:
        try:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "files" not in tables or "meta" not in tables:
                return requested
            legacy = "path_key" not in {
                row[1] for row in conn.execute("PRAGMA table_info(files)")
            }
        except sqlite3.Error as exc:
            raise ParameterError(ro.uninspectable_message(path, exc)) from exc
        if legacy:
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
    total, stale, examples = diag.legacy_scan(conn, str(requested.key))
    if not stale:
        return requested
    raise diag.legacy_error(requested, total, stale, examples, db_path)


def _binding_key(conn: sqlite3.Connection) -> Optional[str]:
    """``META_ROOT_KEY`` read defensively (the ``meta`` table may not exist)."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "meta" not in tables:
        return None
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (META_ROOT_KEY,)).fetchone()
    value = row[0] if row else None
    return str(value) if value else None


def preflight_stage(db_path: str | os.PathLike,
                    root: Optional[str | os.PathLike] = None,
                    declared_root: Optional[str | os.PathLike] = None) -> None:
    """Read-only gate for stages 1-3, run *before* the database is opened.

    :func:`adopt_or_resolve` enforces the same policy inside the transaction;
    this runs first so a refusal costs the output directory nothing at all --
    not a created ``inventory.sqlite``, not a schema upgrade, not a ``-wal``.

    * ``--root`` given -> verify it against the recorded binding (or the existing
      rows, for a pre-identity database).
    * no ``--root``, directory already bound -> nothing to verify here; the
      binding decides the scope, and a config that describes a sub-tree of it is
      allowed (see :func:`adopt_or_resolve`).
    * no ``--root``, not bound, but the config *declared* a root -> verify that
      root, because the stage is about to bind the directory to it.
    * no ``--root``, not bound, nothing declared -> refuse. A defaulted
      ``paths.root`` is not a declaration.
    """
    if root is not None:
        preflight(db_path, root)
        return
    path = Path(db_path)
    bound = False
    if path.is_file():
        conn = ro.connect_readonly(path)
        try:
            try:
                bound = _binding_key(conn) is not None
            except sqlite3.Error as exc:
                raise ParameterError(ro.uninspectable_message(path, exc)) from exc
        finally:
            conn.close()
    if bound:
        return
    if declared_root is None:
        raise diag.unbound_error(str(db_path))
    preflight(db_path, declared_root)


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
    # ``root_key`` is a denormalised convenience column; ``path_key`` is the
    # authority every scope filter uses. Keep the two consistent for *every* row
    # in scope (not just NULL ones), so :func:`root_key_mismatches` stays 0 after
    # any legacy adoption.
    predicate, params = scope.clause("files")
    conn.execute(
        f"UPDATE files SET root_key = :root_key "
        f"WHERE {predicate} AND COALESCE(root_key, '') != :root_key",
        {**params, "root_key": scope.key},
    )
    conn.commit()
    return scope


def root_key_mismatches(conn: sqlite3.Connection, scope: RootScope) -> int:
    """In-scope rows whose ``root_key`` disagrees with the binding (should be 0)."""
    if not scope.bound:
        return 0
    predicate, params = scope.clause("f")
    return int(conn.execute(
        f"SELECT COUNT(*) AS n FROM files f WHERE {predicate} "
        "AND COALESCE(f.root_key, '') != :expected",
        {**params, "expected": scope.key},
    ).fetchone()["n"])


def resolve(conn: sqlite3.Connection, root: Optional[str | os.PathLike] = None,
            db_path: Optional[str] = None, require_bound: bool = True) -> RootScope:
    """Scope for a downstream stage: the recorded binding + the recorded run id.

    When ``root`` is given it is verified too, so ``stage1 --config`` pointed at
    a foreign output directory fails closed exactly like the one-command entry
    point does.

    ``require_bound`` (the default) makes an unbound database a hard error: a
    stage that ran with ``scope.clause() == "1"`` would silently query the whole
    file while printing a root, which is exactly the dishonesty this module
    exists to remove. Stage 0 creates the binding, so the fix is to run it (or
    pass ``--root``). ``require_bound=False`` exists only for read-only viewers
    (the review server) that must still serve a pre-identity output directory.
    """
    from . import db

    current = recorded(conn)
    if root is not None:
        verified = check(conn, root, db_path=db_path)
        current = RootScope(key=verified.key, path=verified.path)
    if require_bound and not current.bound:
        raise diag.unbound_error(db_path)
    return RootScope(key=current.key, path=current.path,
                     run_id=db.get_meta(conn, META_RUN_ID))


def adopt_or_resolve(conn: sqlite3.Connection, root: Optional[str | os.PathLike],
                     db_path: Optional[str] = None,
                     declared_root: Optional[str | os.PathLike] = None) -> RootScope:
    """Scope for stages 1-3, which must never run unscoped.

    * **Already bound** -> use that binding. An explicit ``root`` (``--root``) is
      verified against it and a mismatch fails closed. ``declared_root`` is *not*
      re-verified here: a config may legitimately describe a sub-tree while the
      directory is bound to the tree Stage 0 scanned.
    * **Not bound, and a root was actually supplied** -> validate it and record
      the binding exactly as Stage 0 would, so the stage still runs *scoped*.
      "Supplied" means ``--root``, or a ``paths.root`` the user really wrote in
      the config (:attr:`src.config.Config.declared_root`, which is ``None`` when
      the value came from the packaged default).
    * **Not bound and no root supplied** -> :class:`ParameterError` from
      :func:`resolve`, before anything is written. Binding to the default
      ``E:/Photos`` instead would claim a library nobody named, and on a machine
      where that path happens to exist it would silently adopt it.
    """
    current = recorded(conn)
    if current.bound:
        return resolve(conn, root, db_path=db_path)
    candidate = root if root is not None else declared_root
    if candidate is None:
        return resolve(conn, None, db_path=db_path)     # raises: unbound + no root
    scope = bind(conn, candidate, db_path=db_path)
    source = "--root" if root is not None else "config paths.root"
    print(f"[scope] bound this output directory to {scope.path} (from {source})")
    return scope


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
