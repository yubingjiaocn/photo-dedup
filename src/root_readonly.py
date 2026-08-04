r"""Read-only access to an output directory's database, with no side effects.

This is the *rejection* path's I/O layer, split out of :mod:`src.root_scope` so
the two hard requirements below live in one small, testable place:

1. **Correct URI escaping.** ``f"file:{path}?mode=ro"`` is wrong. ``#`` starts a
   URI fragment, so ``C:\out#1\inventory.sqlite`` silently becomes ``C:\out``
   (which does not exist, so SQLite opens/creates the wrong thing or errors
   confusingly), and a literal ``%`` is read as the start of a percent-escape.
   Both would make a scope check read an *empty* database and therefore fail
   *open* -- the exact failure mode this project cannot have.
   ``Path.as_uri`` performs precisely the escaping SQLite's URI parser expects.

2. **No files created while rejecting a run.** A run refused for the wrong root
   must leave the directory byte-for-byte as it found it, and plain ``mode=ro``
   on a WAL database creates ``-wal``/``-shm`` merely by reading it.

Everything here raises :class:`src.root_identity.ParameterError` instead of
leaking a raw ``sqlite3`` error, because the caller's job is to tell the user
which parameter to change.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

from .root_identity import ParameterError


def readonly_uri(path: str | os.PathLike) -> str:
    """Correctly escaped read-only SQLite URI for ``path`` (see module docstring)."""
    resolved = Path(path).resolve()
    if not resolved.is_absolute():  # pragma: no cover - resolve() makes it absolute
        raise ParameterError(f"database path must be absolute: {path}")
    return f"{resolved.as_uri()}?mode=ro"


def immutable_uri(path: str | os.PathLike) -> str:
    """Escaped ``immutable=1`` URI: reads without creating or locking anything."""
    resolved = Path(path).resolve()
    return f"{resolved.as_uri()}?immutable=1"


def wal_sidecars(path: str | os.PathLike) -> List[Path]:
    """Existing ``-wal``/``-shm`` companions of ``path``."""
    text = str(Path(path))
    return [candidate for candidate in (Path(text + "-wal"), Path(text + "-shm"))
            if candidate.exists()]


def connect_readonly(path: str | os.PathLike) -> sqlite3.Connection:
    """Open an existing database read-only, or fail with :class:`ParameterError`.

    Creates no side files when the database was closed cleanly, so the mode is
    chosen by what is already on disk:

    * no ``-wal``/``-shm`` present -> the last writer committed and checkpointed,
      the main file is self-contained, and ``immutable=1`` reads it without
      creating or locking anything;
    * a companion present -> a WAL may hold committed data, so ``mode=ro`` is
      used, because reading stale content would be worse than touching a file
      that already exists (``immutable`` deliberately ignores the WAL).

    A missing file is a programming error here (callers check first); anything
    else (permissions, corruption, a directory in the way) is reported as a
    parameter problem, and the connection is closed before raising so no handle
    is leaked into the failure path.
    """
    resolved = Path(path).resolve()
    hot = bool(wal_sidecars(resolved))
    uri = readonly_uri(resolved) if hot else immutable_uri(resolved)
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        # Force a real read: sqlite3.connect is lazy, so a corrupt file or a
        # permission problem would otherwise surface later, mid-pipeline.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn
    except sqlite3.Error as exc:
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        raise ParameterError(unreadable_message(path, exc)) from exc


def unreadable_message(path: str | os.PathLike, exc: BaseException) -> str:
    """Diagnostic for a database that cannot be read at all."""
    return "\n".join([
        f"cannot read the existing database in this output directory: {path}",
        f"  sqlite error : {type(exc).__name__}: {exc}",
        "The run stopped before touching anything. Either the file is not a"
        " readable SQLite database (corrupt/locked/permissions), or it is not"
        " a photo-dedup output directory.",
        "Choose one:",
        "  * fix the file permissions, or",
        "  * move/rename that file and let this run create a fresh one, or",
        "  * use a different --output directory.",
    ])


def uninspectable_message(path: str | os.PathLike, exc: BaseException) -> str:
    """Diagnostic for a database that opens but cannot be inspected."""
    return (f"cannot inspect the existing database {path}: "
            f"{type(exc).__name__}: {exc}. Nothing was written; move that file "
            "aside or use a different --output directory.")
