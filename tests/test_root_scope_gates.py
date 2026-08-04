"""Can the cross-root gate be made to fail open, or to cause damage?

``test_root_scope.py`` covers "does it reject a wrong root". This file covers the
ways a rejection could stop being one:

* **Fail open through a path.** A ``#`` or ``%`` in the output path used to be
  pasted straight into a ``file:...`` URI. ``#`` starts a fragment and ``%``
  starts an escape, so the check read a *different* (usually empty) database, saw
  no recorded root, and let a second library in.
* **Fail open through corruption.** A truncated or non-SQLite ``inventory.sqlite``
  must stop the run, not be read as "no binding recorded yet".
* **Damage while refusing.** A refused run must leave the directory byte-for-byte
  as it found it: no ``-wal``/``-shm``, no diagnostic log, no created output
  directory, no schema upgrade, no rebinding.
* **Running unscoped.** Stages 1-3 on an unbound directory must refuse unless a
  root was really supplied. The packaged default ``paths.root: E:/Photos`` is not
  a supplied root -- adopting it would claim a library nobody named.

Leak paths (a foreign row reaching a consumer) live in
``test_root_scope_leaks.py``; the shared fixtures live in
``root_scope_fixtures.py``.
"""

from __future__ import annotations

import sqlite3

import pytest

from src import (
    db,
    root_readonly,
    root_scope,
    run_pipeline,
    stage1_features,
    stage2_cluster,
    stage3_report,
)
from src.config import load_config
from src.root_scope import ParameterError
from tests import root_scope_fixtures as fixtures

# --- fail open through the path itself -------------------------------------

def test_hash_and_percent_in_the_output_path_still_bind_and_resume(tmp_path):
    root = tmp_path / "Photos" / "2026"
    fixtures.photos(root, 2)
    output = tmp_path / fixtures.HOSTILE_DIR

    first = run_pipeline.run(str(root), str(output), backend="stub")
    again = run_pipeline.run(str(root), str(output), backend="stub")

    assert first["inventory"]["files"] == 2
    assert again["features"]["processed"] == 0          # resumed, not re-read
    conn = db.open_db(output / "inventory.sqlite")
    try:
        assert root_scope.recorded(conn).key == root_scope.normalize(root)
    finally:
        conn.close()


def test_hash_and_percent_in_the_output_path_cannot_fail_open_on_mismatch(tmp_path):
    """The regression: a mis-escaped URI read an empty DB and accepted any root.

    Checked through ``main`` and through the filesystem, not only through the
    exception: with the naive URI the *preflight* fails open, Stage 0's own check
    still refuses -- but by then the output directory has been created, the
    diagnostic log written, and a stray database created next to it. So the
    evidence has to be "nothing on disk moved", not just "an error was raised".
    """
    first_root = tmp_path / "Photos" / "2015"
    second_root = tmp_path / "Photos" / "2026"
    fixtures.photos(first_root, 2, tag=1)
    fixtures.photos(second_root, 2, tag=2)
    output = tmp_path / fixtures.HOSTILE_DIR

    run_pipeline.run(str(first_root), str(output), backend="stub")
    before = fixtures.tree(output)
    siblings_before = fixtures.tree(tmp_path)
    db_bytes = (output / "inventory.sqlite").read_bytes()

    with pytest.raises(ParameterError) as excinfo:
        run_pipeline.run(str(second_root), str(output), backend="stub")
    assert "already bound to a different photo root" in str(excinfo.value)

    code = run_pipeline.main([
        "--root", str(second_root), "--output", str(output), "--backend", "stub",
        "--no-serve",
    ])

    assert code == 2
    assert (output / "inventory.sqlite").read_bytes() == db_bytes
    assert fixtures.tree(output) == before                  # no log, no -wal, no -shm
    assert not (output / "photo-dedup.log").exists()
    # A mis-escaped URI truncates at '#', so it would create a stray database
    # named after the truncated prefix somewhere beside the real one.
    assert fixtures.tree(tmp_path) == siblings_before


def test_readonly_uri_escapes_hash_and_percent(tmp_path):
    directory = tmp_path / fixtures.HOSTILE_DIR
    directory.mkdir()
    db_path = directory / "inventory.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE probe(x)")
    conn.execute("INSERT INTO probe VALUES (1)")
    conn.commit()
    conn.close()

    for uri in (root_readonly.readonly_uri(db_path), root_readonly.immutable_uri(db_path)):
        assert "#" not in uri.split("?")[0]                     # no live fragment
        opened = sqlite3.connect(uri, uri=True)
        try:
            assert opened.execute("SELECT COUNT(*) FROM probe").fetchone()[0] == 1
        finally:
            opened.close()

    # The naive form is what used to be used, and it silently opens nothing.
    naive = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.Error):
            naive.execute("SELECT COUNT(*) FROM probe").fetchone()
    finally:
        naive.close()

# --- fail open through corruption ------------------------------------------

@pytest.mark.parametrize("payload", [
    b"this is not a database at all",
    b"SQLite format 3\x00truncated header, nothing else",
    b"",
])
def test_corrupt_or_empty_database_is_a_parameter_error_not_a_fresh_start(tmp_path, payload):
    root = tmp_path / "Photos" / "2026"
    fixtures.photos(root, 1)
    output = tmp_path / "out"
    output.mkdir()
    db_path = output / "inventory.sqlite"
    db_path.write_bytes(payload)
    before = fixtures.tree(output)

    if payload == b"":
        # A zero-byte file is a legitimately empty SQLite database, so it must be
        # adopted rather than rejected -- but it must not be *mistaken* for a
        # bound directory either.
        assert root_scope.preflight(db_path, root).key == root_scope.normalize(root)
        return

    with pytest.raises(ParameterError) as excinfo:
        root_scope.preflight(db_path, root)
    message = str(excinfo.value)
    assert "cannot read the existing database" in message
    assert "corrupt/locked/permissions" in message
    assert "different --output directory" in message
    assert fixtures.tree(output) == before          # no -wal/-shm, nothing rewritten


def test_corrupt_database_stops_the_cli_before_it_writes_a_log(tmp_path):
    root = tmp_path / "Photos" / "2026"
    fixtures.photos(root, 1)
    output = tmp_path / "out"
    output.mkdir()
    (output / "inventory.sqlite").write_bytes(b"not a database")
    before = fixtures.tree(output)

    code = run_pipeline.main([
        "--root", str(root), "--output", str(output), "--backend", "stub", "--no-serve",
    ])

    assert code == 2
    assert fixtures.tree(output) == before          # no photo-dedup.log, no -wal, no thumbs


def test_a_directory_in_place_of_the_database_is_rejected(tmp_path):
    root = tmp_path / "Photos" / "2026"
    fixtures.photos(root, 1)
    output = tmp_path / "out"
    (output / "inventory.sqlite").mkdir(parents=True)

    # ``is_file()`` is False for a directory, so preflight treats it as absent;
    # the failure must then come from opening it, not from a corrupted scope.
    with pytest.raises((ParameterError, OSError, sqlite3.Error)):
        run_pipeline.run(str(root), str(output), backend="stub")

# --- refusing must not damage anything -------------------------------------

def test_rejected_bound_mismatch_creates_no_log_wal_or_shm(tmp_path):
    first_root = tmp_path / "Photos" / "2015"
    second_root = tmp_path / "Photos" / "2026"
    fixtures.photos(first_root, 2, tag=1)
    fixtures.photos(second_root, 2, tag=2)
    output = tmp_path / "out"
    run_pipeline.run(str(first_root), str(output), backend="stub")
    before = fixtures.tree(output)
    db_bytes = (output / "inventory.sqlite").read_bytes()

    code = run_pipeline.main([
        "--root", str(second_root), "--output", str(output), "--backend", "stub",
        "--no-serve",
    ])

    assert code == 2
    assert fixtures.tree(output) == before
    assert (output / "inventory.sqlite").read_bytes() == db_bytes
    assert not (output / "photo-dedup.log").exists()
    assert not (output / "inventory.sqlite-wal").exists()
    assert not (output / "inventory.sqlite-shm").exists()


def test_rejected_legacy_mixed_root_creates_no_side_files(tmp_path):
    """A pre-identity database must not even be migrated by the run that refuses."""
    root_2015 = tmp_path / "Photos" / "2015"
    root_2026 = tmp_path / "Photos" / "2026"
    output = tmp_path / "out"
    output.mkdir(parents=True)
    legacy = output / "inventory.sqlite"
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, "
                 "basename TEXT, size_bytes INTEGER, file_kind TEXT, scan_status TEXT)")
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    for path in fixtures.photos(root_2015, 2, tag=1) + fixtures.photos(root_2026, 2, tag=2):
        conn.execute("INSERT INTO files(path, basename, size_bytes, file_kind, scan_status) "
                     "VALUES (?, ?, 1, 'jpg', 'done')", (str(path), path.name))
    conn.commit()
    conn.close()
    before = fixtures.tree(output)
    db_bytes = legacy.read_bytes()

    code = run_pipeline.main([
        "--root", str(root_2026), "--output", str(output), "--backend", "stub",
        "--no-serve",
    ])

    assert code == 2
    assert fixtures.tree(output) == before
    assert legacy.read_bytes() == db_bytes
    assert not (output / "photo-dedup.log").exists()
    assert not (output / "inventory.sqlite-wal").exists()
    assert not (output / "inventory.sqlite-shm").exists()
    # Still legacy: no identity column was added by the refused run.
    conn = sqlite3.connect(legacy)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    finally:
        conn.close()
    assert "path_key" not in columns


def test_refusal_does_not_create_a_fresh_output_directory(tmp_path):
    """The bound DB lives elsewhere; the report directory must stay uncreated."""
    root = tmp_path / "Photos" / "2026"
    other = tmp_path / "Photos" / "2015"
    fixtures.photos(root, 1)
    fixtures.photos(other, 1, tag=1)
    db_dir = tmp_path / "state"
    db_dir.mkdir()
    db_path = db_dir / "inventory.sqlite"
    conn = db.open_db(db_path)
    fixtures.inventory_rows(conn, fixtures.photos(root, 1))
    root_scope.bind(conn, root, db_path=str(db_path))
    conn.close()
    report_dir = tmp_path / "never-created"

    with pytest.raises(ParameterError):
        stage3_report.run(
            config_path=fixtures.write_config(tmp_path, "mismatch", db_path, report_dir, root=other),
            root_override=str(other),
        )

    assert not report_dir.exists()

# --- unbound stages fail closed --------------------------------------------

@pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
def test_unbound_stage_without_a_root_refuses_and_changes_nothing(tmp_path, stage):
    _root, db_path, _paths = fixtures.unbound_output(tmp_path)
    output = tmp_path / "reports"
    config = fixtures.write_config(tmp_path, f"noroot-{stage}", db_path, output)   # no paths.root
    before = fixtures.tree(db_path.parent)

    runner = {"stage1": lambda: stage1_features.run(config_path=config,
                                                    backend_override="stub"),
              "stage2": lambda: stage2_cluster.run(config_path=config),
              "stage3": lambda: stage3_report.run(config_path=config)}[stage]
    with pytest.raises(ParameterError) as excinfo:
        runner()

    message = str(excinfo.value)
    assert "no recorded photo root" in message
    assert "Nothing was written" in message
    assert "run stage 0 first" in message
    assert fixtures.tree(db_path.parent) == before
    assert not output.exists()
    conn = db.open_db(db_path)
    try:
        assert root_scope.recorded(conn).bound is False        # still unbound
        assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0
    finally:
        conn.close()


def test_default_config_root_is_never_adopted_as_a_binding(tmp_path):
    """``paths.root`` defaulting to E:/Photos must not bind anything.

    The refusal must be the *unbound* one. Adopting the default would raise too
    (the rows here are not under E:/Photos), but with a message about legacy rows
    that sends the user chasing a root they never named -- and on a machine where
    ``E:/Photos`` does exist it would bind silently instead of raising at all.
    """
    _root, db_path, _paths = fixtures.unbound_output(tmp_path)
    config = fixtures.write_config(tmp_path, "defaulted", db_path, tmp_path / "reports")

    cfg = load_config(config)
    assert cfg.declared_root is None                       # nothing was declared
    assert str(cfg.root_path) == "E:/Photos"               # ...but a default exists

    with pytest.raises(ParameterError) as excinfo:
        stage1_features.run(config_path=config, backend_override="stub")
    message = str(excinfo.value)
    assert "no recorded photo root" in message
    assert "E:/Photos" not in message                      # never names the default

    conn = db.open_db(db_path)
    try:
        assert root_scope.recorded(conn).key is None
    finally:
        conn.close()


@pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
def test_empty_output_directory_is_not_bound_to_the_default_root(tmp_path, stage):
    """The dangerous case: nothing in the DB, so a defaulted root would bind.

    With no rows there is no legacy check to accidentally save us -- adopting
    ``paths.root`` would record ``E:/Photos`` as this directory's photo root, and
    every later report would claim a library the user never asked for.
    """
    db_path = tmp_path / "state" / "inventory.sqlite"
    db.open_db(db_path).close()                            # empty, unbound
    output = tmp_path / "reports"
    config = fixtures.write_config(tmp_path, f"empty-{stage}", db_path, output)

    runner = {"stage1": lambda: stage1_features.run(config_path=config,
                                                    backend_override="stub"),
              "stage2": lambda: stage2_cluster.run(config_path=config),
              "stage3": lambda: stage3_report.run(config_path=config)}[stage]
    with pytest.raises(ParameterError) as excinfo:
        runner()

    assert "no recorded photo root" in str(excinfo.value)
    assert not output.exists()
    conn = db.open_db(db_path)
    try:
        assert root_scope.recorded(conn).key is None
        assert db.get_meta(conn, root_scope.META_ROOT_PATH) is None
    finally:
        conn.close()


def test_declared_config_root_is_adopted_and_scopes_the_stage(tmp_path):
    root, db_path, _paths = fixtures.unbound_output(tmp_path)
    other = tmp_path / "Photos" / "2015"
    other_paths = fixtures.photos(other, 3, tag=1)
    conn = db.open_db(db_path)
    fixtures.inventory_rows(conn, other_paths)      # foreign rows in the same database
    conn.close()

    config = fixtures.write_config(tmp_path, "declared", db_path, tmp_path / "reports", root=root)
    assert load_config(config).declared_root == root

    with pytest.raises(ParameterError) as excinfo:
        stage1_features.run(config_path=config, backend_override="stub")
    # The declared root does not cover the foreign rows, so adoption is refused
    # by the legacy check rather than silently narrowing the library.
    assert "outside the requested root" in str(excinfo.value)


def test_explicit_root_adoption_binds_an_unbound_directory(tmp_path):
    root, db_path, _paths = fixtures.unbound_output(tmp_path)
    config = fixtures.write_config(tmp_path, "explicit", db_path, tmp_path / "reports")

    stats = stage1_features.run(config_path=config, backend_override="stub",
                                root_override=str(root))

    assert stats["processed"] == 2
    assert stats["scope"]["root_key"] == root_scope.normalize(root)
    conn = db.open_db(db_path)
    try:
        assert root_scope.recorded(conn).key == root_scope.normalize(root)
        assert root_scope.root_key_mismatches(conn, root_scope.recorded(conn)) == 0
    finally:
        conn.close()


def test_explicit_root_adoption_is_refused_when_legacy_rows_are_mixed(tmp_path):
    root, db_path, _paths = fixtures.unbound_output(tmp_path)
    conn = db.open_db(db_path)
    fixtures.inventory_rows(conn, fixtures.photos(tmp_path / "Photos" / "2015", 2, tag=1))
    conn.close()
    config = fixtures.write_config(tmp_path, "mixed", db_path, tmp_path / "reports")

    with pytest.raises(ParameterError) as excinfo:
        stage1_features.run(config_path=config, backend_override="stub",
                            root_override=str(root))

    message = str(excinfo.value)
    assert "outside the requested root" in message
    assert "left exactly as it is" in message
    conn = db.open_db(db_path)
    try:
        assert root_scope.recorded(conn).bound is False
        assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0
    finally:
        conn.close()

# --- each gate layer, on its own --------------------------------------------

def test_adopt_or_resolve_refuses_without_a_root_and_binds_with_one(tmp_path):
    """The in-transaction gate, tested directly (not via a stage's preflight)."""
    root, db_path, _paths = fixtures.unbound_output(tmp_path)
    conn = db.open_db(db_path)
    try:
        with pytest.raises(ParameterError) as excinfo:
            root_scope.adopt_or_resolve(conn, None, db_path=str(db_path))
        assert "no recorded photo root" in str(excinfo.value)
        assert root_scope.recorded(conn).bound is False

        # An explicit root binds; a declared one would too, a defaulted one is
        # simply never passed in (see src.config.Config.declared_root).
        scope = root_scope.adopt_or_resolve(conn, root, db_path=str(db_path))
        assert scope.key == root_scope.normalize(root)
        assert root_scope.recorded(conn).key == scope.key
    finally:
        conn.close()


def test_resolve_requires_a_binding_unless_the_caller_is_a_viewer(tmp_path):
    _root, db_path, _paths = fixtures.unbound_output(tmp_path)
    conn = db.open_db(db_path)
    try:
        with pytest.raises(ParameterError):
            root_scope.resolve(conn, None, db_path=str(db_path))
        # The review server is read-only and must still serve a legacy directory,
        # which is the only reason this escape hatch exists.
        viewer = root_scope.resolve(conn, None, db_path=str(db_path),
                                    require_bound=False)
        assert viewer.bound is False
    finally:
        conn.close()


def test_preflight_stage_is_read_only_and_creates_nothing(tmp_path):
    """The pre-open gate, tested directly: it must not create the database."""
    db_path = tmp_path / "state" / "inventory.sqlite"

    # Missing database + no root supplied -> refuse, and do not create the file.
    with pytest.raises(ParameterError):
        root_scope.preflight_stage(db_path, None, None)
    assert not db_path.exists()
    assert not db_path.parent.exists()

    # Missing database + a supplied root -> fine (a first run may create it).
    root = tmp_path / "Photos" / "2026"
    fixtures.photos(root, 1)
    root_scope.preflight_stage(db_path, None, root)
    root_scope.preflight_stage(db_path, root, None)
    assert not db_path.exists()

    # Bound database + no root at all -> fine, the binding decides.
    conn = db.open_db(db_path)
    fixtures.inventory_rows(conn, fixtures.photos(root, 1))
    root_scope.bind(conn, root, db_path=str(db_path))
    conn.close()
    before = fixtures.tree(db_path.parent)
    root_scope.preflight_stage(db_path, None, None)
    assert fixtures.tree(db_path.parent) == before
