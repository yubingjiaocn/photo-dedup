"""The refusal messages for an incompatible output directory.

Split out of :mod:`src.root_scope` because these strings are the product: when a
run is rejected the user has to be able to act on it without reading the code.
Every message therefore states what was requested, what was recorded, how the
two relate, how much data is at stake, that **nothing was changed**, and the
concrete alternatives.

Read-only by construction: this module only ever runs ``SELECT``s, because it is
called on the rejection path, where the whole promise is that the directory is
left byte-for-byte as it was found.
"""

from __future__ import annotations

import sqlite3
from typing import List, Optional, Tuple

from .root_identity import ParameterError, RootScope, contains, normalize, relation

# How many example out-of-scope rows a diagnostic lists.
EXAMPLE_ROWS = 5

_RELATION_TEXT = {
    "child": "the recorded root is a PARENT of the requested root",
    "parent": "the requested root is a PARENT of the recorded root",
    "disjoint": "the two roots are unrelated",
}

_LEGACY_HEAD = ("incompatible output directory: it was created before root identity was"
                " recorded and still holds rows from outside the requested root.")
_LEGACY_TAIL = [
    "Nothing was read, written, or deleted -- this older database is left exactly as"
    " it is, because deleting rows to make the numbers match would destroy work.",
    "Choose one:",
    "  * use a fresh --output directory for this root (recommended), or",
    "  * re-run with --root set to the common parent of the paths above to adopt"
    " the whole existing inventory.",
]


# --- counting (read-only) --------------------------------------------------

def out_of_scope_count(conn: sqlite3.Connection, root_key: str) -> int:
    """Rows whose ``path_key`` is outside ``root_key``."""
    predicate, params = RootScope(key=root_key, path=None).clause("f")
    return int(conn.execute(
        f"SELECT COUNT(*) AS n FROM files f WHERE NOT {predicate}", params
    ).fetchone()["n"])


def out_of_scope_examples(conn: sqlite3.Connection, root_key: str,
                         limit: int = EXAMPLE_ROWS) -> List[str]:
    """A few example paths from outside ``root_key``, for the error message."""
    predicate, params = RootScope(key=root_key, path=None).clause("f")
    rows = conn.execute(
        f"SELECT path FROM files f WHERE NOT {predicate} ORDER BY f.id LIMIT :limit",
        {**params, "limit": int(limit)},
    ).fetchall()
    return [str(row["path"]) for row in rows]


def counts(conn: sqlite3.Connection) -> Tuple[int, int]:
    """``(inventory rows, still images)`` -- what the user would be risking."""
    total = int(conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])
    return total, int(conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE file_kind IN ('jpg', 'jpg_motion')"
    ).fetchone()["n"])


def legacy_scan(conn: sqlite3.Connection, root_key: str) -> Tuple[int, int, List[str]]:
    """Pre-identity scan: ``(total, stale, examples)`` normalised in Python.

    A legacy database has no ``path_key`` column, so containment cannot be a SQL
    prefix test; normalising each row here keeps the check read-only (no schema
    upgrade on a database this run may be about to refuse).
    """
    rows = conn.execute("SELECT path FROM files").fetchall()
    stale = 0
    examples: List[str] = []
    for row in rows:
        try:
            key = normalize(row["path"])
        except ParameterError:
            continue
        if not contains(root_key, key):
            stale += 1
            if len(examples) < EXAMPLE_ROWS:
                examples.append(str(row["path"]))
    return len(rows), stale, examples


# --- messages --------------------------------------------------------------

def mismatch_error(requested: RootScope, recorded: RootScope,
                   conn: sqlite3.Connection, db_path: Optional[str]) -> ParameterError:
    """This output directory is bound to a different root."""
    total, stills = counts(conn)
    kind = relation(str(requested.key), str(recorded.key))
    lines = [
        "incompatible output directory: it is already bound to a different photo root.",
        f"  requested root : {requested.path}  [key={requested.key}]",
        f"  recorded root  : {recorded.path}  [key={recorded.key}]",
        f"  relation       : {_RELATION_TEXT.get(kind, kind)}",
        f"  existing data  : {total} inventory row(s), {stills} still image(s)"
        + (f" in {db_path}" if db_path else ""),
        "Reusing it would mix records from different roots into one report, which is how"
        " a 2026-only run previously reported 2015/2017 photos.",
        "Nothing was read, written, or deleted. Choose one:",
        "  * use a separate --output directory for this root (recommended), or",
        f"  * re-run with --root \"{recorded.path}\" to resume that inventory, or",
        "  * move/rename the existing output directory if you no longer need it.",
    ]
    if kind in ("parent", "child"):
        lines.append(
            "Note: a parent/child root is still a different scope -- the reports,"
            " thumbnail cache and ETA would describe a library you did not ask for."
        )
    return ParameterError("\n".join(lines))


def legacy_error(requested: RootScope, total: int, stale: int,
                 examples: List[str], db_path: Optional[str]) -> ParameterError:
    """A pre-identity database holding rows from outside the requested root."""
    where = f" in {db_path}" if db_path else ""
    return ParameterError("\n".join([
        _LEGACY_HEAD,
        f"  requested root : {requested.path}  [key={requested.key}]",
        f"  existing data  : {total} inventory row(s), {stale} of them outside that root"
        + where,
        "  examples       : " + (", ".join(examples) if examples else "-"),
        *_LEGACY_TAIL,
    ]))


def unbound_error(db_path: Optional[str]) -> ParameterError:
    """A scoped stage was asked to run on a directory with no recorded root."""
    return ParameterError("\n".join([
        "this output directory has no recorded photo root, so a scoped stage"
        " cannot know which photos it is responsible for.",
        f"  database : {db_path or 'unknown'}",
        "Running unscoped would query every row in the file while reporting a"
        " single root -- the exact mix-up that made a 2026 run report 2015/2017"
        " photos. Nothing was written.",
        "Choose one:",
        "  * run stage 0 first (it records the binding), or",
        "  * pass --root <photo root> to this stage to verify and adopt it, or",
        "  * use the one-command entry point: python -m src.run_pipeline"
        " --root ... --output ...",
    ]))
