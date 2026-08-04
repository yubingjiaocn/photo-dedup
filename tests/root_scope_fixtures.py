"""Shared fixtures for the cross-root safety tests.

Kept in one place because the two adversarial suites
(``test_root_scope_gates.py`` -- can the gate be made to fail open or to cause
damage; ``test_root_scope_leaks.py`` -- can a foreign row reach a consumer) need
the same libraries, configs and byte-level directory snapshots, and drift between
two copies of a fixture is how a safety test quietly stops testing anything.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image

from src import db, root_scope

# Output directory names that break naive URI building. '#' truncates a ``file:``
# URI at the fragment; '%' begins a percent-escape. Both are legal in a Windows
# folder name, so both are things a user can actually type.
HOSTILE_DIR = "out#1 %2f %ZZ end"


def photos(root: Path, count: int = 2, tag: int = 0) -> List[Path]:
    """``count`` tiny, distinct JPEGs under ``root`` (``tag`` keeps names unique)."""
    root.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(count):
        path = root / f"IMG_2026010{index}_1200{tag:02d}.jpg"
        Image.new("RGB", (64, 48), (10 + index * 20, 60 + tag, 120)).save(path, "JPEG")
        made.append(path)
    return made


def write_config(tmp_path: Path, name: str, db_path: Path, output: Path,
                 root: Optional[Path] = None) -> str:
    """Write a stage config; **omit** ``paths.root`` entirely when ``root`` is None.

    Omitting the key is the point of several tests: the packaged default then
    supplies ``E:/Photos``, and no stage may treat that as a declared root.
    """
    paths: Dict[str, str] = {
        "db": str(db_path), "output_dir": str(output),
        "models_dir": str(tmp_path / "models"), "trash": str(tmp_path / "trash"),
    }
    if root is not None:
        paths["root"] = str(root)
    path = tmp_path / f"config-{name}.yaml"
    path.write_text(yaml.safe_dump({
        "paths": paths,
        "features": {"backend": "stub", "thumbnails": {"enabled": False}},
    }), encoding="utf-8")
    return str(path)


def inventory_rows(conn: sqlite3.Connection, paths: List[Path]) -> List[int]:
    """Insert plain inventory rows (no features) and return their ids."""
    ids = []
    for index, path in enumerate(paths):
        stat = path.stat()
        ids.append(db.insert_file(conn, {
            "path": str(path), "basename": path.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "width": 64, "height": 48,
            "file_kind": "jpg", "scan_status": "done", "exif_timestamp": 1000 + index,
        }))
    conn.commit()
    return ids


def tree(path: Path) -> Dict[str, int]:
    """Every entry under ``path`` with its size (``-1`` for directories).

    Comparing two of these is how a test asserts "the refused run changed nothing"
    without enumerating the specific side files it must not have created.
    """
    if not path.exists():
        return {}
    return {str(item.relative_to(path)): (item.stat().st_size if item.is_file() else -1)
            for item in path.rglob("*")}


def unbound_output(tmp_path: Path) -> tuple[Path, Path, List[Path]]:
    """A database with rows but **no recorded root** (the pre-identity case)."""
    root = tmp_path / "Photos" / "2026"
    paths = photos(root, 2)
    db_path = tmp_path / "state" / "inventory.sqlite"
    conn = db.open_db(db_path)
    inventory_rows(conn, paths)
    try:
        assert root_scope.recorded(conn).bound is False
    finally:
        conn.close()
    return root, db_path, paths


def straddling_library(tmp_path: Path) -> Dict[str, Any]:
    """A bound directory that nevertheless contains a cross-root group.

    Order matters, and it is the realistic one: the directory is bound to
    ``Photos/2026`` while it holds only 2026 rows (so :func:`root_scope.bind`
    accepts it), and the foreign row + straddling group appear *afterwards* --
    left over from a wider run, written by external tooling, or produced by a bug
    elsewhere. ``bind`` cannot be the defence here, because it already happened;
    the query-level scope has to be.

    The group is built to be maximally dangerous: the foreign member is
    ``AUTO_REMOVE`` (so it would enter the delete manifest), and it is the
    bidirectional motion partner of an in-scope ``AUTO_REMOVE`` file (so partner
    expansion would drag it in even if the member join alone were fixed).
    """
    root = tmp_path / "Photos" / "2026"
    foreign_root = tmp_path / "Photos" / "2015"
    keeper, mine = photos(root, 2)
    theirs = photos(foreign_root, 1, tag=1)[0]
    output = tmp_path / "out"
    db_path = output / "inventory.sqlite"

    conn = db.open_db(db_path)
    keeper_id, mine_id = inventory_rows(conn, [keeper, mine])
    scope = root_scope.bind(conn, root, db_path=str(db_path))   # legitimate: 2026 only
    theirs_id = inventory_rows(conn, [theirs])[0]               # appears afterwards
    conn.execute("UPDATE files SET motion_partner_id = ?, file_kind = 'jpg_motion' "
                 "WHERE id = ?", (theirs_id, mine_id))
    conn.execute("UPDATE files SET motion_partner_id = ?, file_kind = 'jpg_motion' "
                 "WHERE id = ?", (mine_id, theirs_id))
    group_id = db.insert_group(
        conn, "burst", keeper_id,
        [(keeper_id, True, "keep"), (mine_id, False, "dup"), (theirs_id, False, "dup")], 1,
    )
    db.update_member_decisions(conn, group_id, [
        {"file_id": keeper_id, "decision": "KEEP", "confidence": 1.0,
         "reason": "keep", "evidence_json": "{}"},
        {"file_id": mine_id, "decision": "AUTO_REMOVE", "confidence": 1.0,
         "reason": "byte-identical", "evidence_json": "{}"},
        {"file_id": theirs_id, "decision": "AUTO_REMOVE", "confidence": 1.0,
         "reason": "byte-identical", "evidence_json": "{}"},
    ])
    conn.commit()
    conn.close()
    return {"root": root, "output": output, "db_path": db_path, "scope": scope,
            "group_id": group_id, "keeper": keeper_id, "mine": mine_id,
            "theirs": theirs_id, "theirs_path": theirs, "mine_path": mine}
