"""Cross-root safety: one output directory belongs to exactly one photo root.

The Windows bug being regression-tested: ``--root E:\\Photos\\2026`` reused the
output directory of an earlier ``E:\\Photos\\2015`` / ``2017`` run, so Stage 0
appended to the same inventory, Stages 1-3 queried everything, and the review
reported photos the user never asked about.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src import db, root_scope, run_pipeline, stage0_inventory, stage1_features
from src.root_scope import ParameterError

LEGACY_SCHEMA = """
CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, basename TEXT,
  size_bytes INTEGER, mtime_ns INTEGER, exif_datetime TEXT, exif_timestamp INTEGER,
  width INTEGER, height INTEGER, file_kind TEXT, motion_partner_id INTEGER,
  scan_status TEXT, scan_error TEXT);
CREATE TABLE features(file_id INTEGER PRIMARY KEY, phash BLOB, dinov2_embedding BLOB,
  quality_score REAL, quality_meta TEXT, face_count INTEGER, faces_json TEXT, status TEXT);
CREATE TABLE groups(id INTEGER PRIMARY KEY, group_type TEXT, keep_file_id INTEGER,
  member_count INTEGER, created_at INTEGER);
CREATE TABLE group_members(group_id INTEGER, file_id INTEGER, is_keep INTEGER,
  reason TEXT, PRIMARY KEY(group_id, file_id));
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""


def _photos(root: Path, count: int = 2, tag: int = 0) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(count):
        path = root / f"IMG_2026010{index}_1200{tag:02d}.jpg"
        Image.new("RGB", (64, 48), (10 + index * 20, 60 + tag, 120)).save(path, "JPEG")
        made.append(path)
    return made


def _config(tmp_path: Path, root: Path, db_path: Path, output: Path) -> str:
    path = tmp_path / f"config-{root.name}-{output.name}.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(db_path), "output_dir": str(output),
                  "models_dir": str(tmp_path / "models"), "trash": str(tmp_path / "trash")},
        "features": {"backend": "stub", "thumbnails": {"enabled": False}},
    }), encoding="utf-8")
    return str(path)


# --- normalisation ---------------------------------------------------------

@pytest.mark.parametrize("variant", [
    r"E:\Photos\2026", "E:/Photos/2026", "E:/Photos/2026/", r"e:\photos\2026",
    r"E:\Photos\\2026", r"E:/Photos/./2026",
])
def test_windows_case_and_separator_variants_are_one_root(variant):
    assert root_scope.normalize(variant) == root_scope.normalize(r"E:\Photos\2026")


def test_sibling_prefix_is_not_a_child_root():
    parent = root_scope.normalize(r"E:\Photos\2026")
    sibling = root_scope.normalize(r"E:\Photos\2026extra")
    assert not root_scope.contains(parent, sibling)
    assert root_scope.relation(sibling, parent) == "disjoint"


def test_parent_and_child_relations_are_named():
    parent = root_scope.normalize(r"E:\Photos")
    child = root_scope.normalize(r"E:\Photos\2026")
    assert root_scope.relation(child, parent) == "child"
    assert root_scope.relation(parent, child) == "parent"
    assert root_scope.relation(child, child) == "same"


def test_unc_paths_normalise_and_nest():
    share = root_scope.normalize(r"\\NAS\Photos")
    below = root_scope.normalize(r"\\nas\photos\2026")
    assert root_scope.contains(share, below)


def test_empty_path_is_rejected():
    with pytest.raises(ParameterError):
        root_scope.normalize("   ")


# --- fail closed -----------------------------------------------------------

def test_second_root_is_rejected_before_any_mutation(tmp_path):
    first_root = tmp_path / "Photos" / "2015"
    second_root = tmp_path / "Photos" / "2026"
    _photos(first_root, 2, tag=1)
    _photos(second_root, 3, tag=2)
    output = tmp_path / "out"

    first = run_pipeline.run(str(first_root), str(output), backend="stub")
    assert first["inventory"]["files"] == 2

    db_path = output / "inventory.sqlite"
    before = db_path.read_bytes()
    with pytest.raises(ParameterError) as excinfo:
        run_pipeline.run(str(second_root), str(output), backend="stub")

    message = str(excinfo.value)
    assert "incompatible output directory" in message
    assert str(second_root) in message and str(first_root) in message
    assert "Nothing was read, written, or deleted" in message
    assert "separate --output directory" in message
    # Byte-identical DB: the rejected run did not even touch the inventory.
    assert db_path.read_bytes() == before


def test_parent_and_child_roots_are_rejected_with_the_relation_named(tmp_path):
    root = tmp_path / "Photos" / "2026"
    _photos(root, 2)
    output = tmp_path / "out"
    run_pipeline.run(str(root), str(output), backend="stub")

    with pytest.raises(ParameterError) as parent:
        run_pipeline.run(str(tmp_path / "Photos"), str(output), backend="stub")
    assert "PARENT of the recorded root" in str(parent.value)

    child = root / "sub"
    _photos(child, 1, tag=3)
    with pytest.raises(ParameterError) as child_error:
        run_pipeline.run(str(child), str(output), backend="stub")
    assert "PARENT of the requested root" in str(child_error.value)


def test_same_root_stays_resumable_including_case_and_separator_variants(tmp_path):
    root = tmp_path / "Photos" / "2026"
    _photos(root, 3)
    output = tmp_path / "out"

    first = run_pipeline.run(str(root), str(output), backend="stub")
    again = run_pipeline.run(str(root) + "/", str(output), backend="stub")

    assert first["features"]["processed"] == 3
    assert again["features"]["processed"] == 0     # resumed, not redone
    assert again["inventory"]["files"] == 3
    assert again["report"]["all_items"] == 3


def test_each_root_reports_only_its_own_photos(tmp_path):
    root_2015 = tmp_path / "Photos" / "2015"
    root_2026 = tmp_path / "Photos" / "2026"
    _photos(root_2015, 2, tag=1)
    _photos(root_2026, 5, tag=2)

    old = run_pipeline.run(str(root_2015), str(tmp_path / "out2015"), backend="stub")
    new = run_pipeline.run(str(root_2026), str(tmp_path / "out2026"), backend="stub")

    assert old["report"]["all_items"] == 2
    assert new["report"]["all_items"] == 5
    assert new["report"]["scope"]["files_in_scope"] == 5
    assert new["report"]["scope"]["files_out_of_scope"] == 0
    assert new["performance"]["inventory_still_images"] == 5


# --- scoped queries --------------------------------------------------------

def test_stage_queries_ignore_another_roots_rows_in_one_database(tmp_path):
    """A shared DB (legacy adoption) must still scope stage/report queries."""
    root_a = tmp_path / "Photos" / "2015"
    root_b = tmp_path / "Photos" / "2026"
    paths_a = _photos(root_a, 2, tag=1)
    paths_b = _photos(root_b, 3, tag=2)
    db_path = tmp_path / "shared.sqlite"

    conn = db.open_db(db_path)
    for path in paths_a + paths_b:
        stat = path.stat()
        db.insert_file(conn, {
            "path": str(path), "basename": path.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "width": 64, "height": 48,
            "file_kind": "jpg", "scan_status": "done", "exif_timestamp": 1000,
        })
    conn.commit()

    scope_a = root_scope.RootScope(key=root_scope.normalize(root_a), path=str(root_a))
    scope_b = root_scope.RootScope(key=root_scope.normalize(root_b), path=str(root_b))

    assert db.count_files(conn, scope=scope_a) == 2
    assert db.count_files(conn, scope=scope_b) == 3
    assert db.count_files(conn) == 5
    assert db.count_still_images(conn, scope=scope_b) == 3
    assert len(list(db.iter_files_for_features(conn, scope=scope_a))) == 2
    assert db.build_all_view_index(conn, scope=scope_b) == 3
    summary = root_scope.summary(conn, scope_b)
    assert summary["files_in_scope"] == 3
    assert summary["files_out_of_scope"] == 2
    assert "ignored_out_of_scope_rows=2" in root_scope.scope_note(summary)
    conn.close()


def test_stage1_only_processes_the_bound_root(tmp_path):
    root_a = tmp_path / "Photos" / "2015"
    root_b = tmp_path / "Photos" / "2026"
    _photos(root_a, 2, tag=1)
    _photos(root_b, 3, tag=2)
    db_path = tmp_path / "inventory.sqlite"
    output = tmp_path / "out"

    config_a = _config(tmp_path, root_a, db_path, output)
    stage0_inventory.run(config_path=config_a)
    # Same DB, other root's rows inserted directly (simulates the old mixing).
    conn = db.open_db(db_path)
    for path in _photos(root_b, 3, tag=2):
        stat = path.stat()
        db.insert_file(conn, {
            "path": str(path), "basename": path.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "width": 64, "height": 48,
            "file_kind": "jpg", "scan_status": "done", "exif_timestamp": 1000,
        })
    conn.commit()
    conn.close()

    stats = stage1_features.run(config_path=config_a, backend_override="stub")
    assert stats["processed"] == 2                       # not 5
    assert stats["scope"]["files_in_scope"] == 2
    assert stats["scope"]["files_out_of_scope"] == 3


def test_stage_cli_root_override_fails_closed(tmp_path):
    root = tmp_path / "Photos" / "2026"
    other = tmp_path / "Photos" / "2015"
    _photos(root, 2)
    _photos(other, 1, tag=1)
    db_path = tmp_path / "inventory.sqlite"
    output = tmp_path / "out"
    stage0_inventory.run(config_path=_config(tmp_path, root, db_path, output))

    with pytest.raises(ParameterError):
        stage1_features.run(
            config_path=_config(tmp_path, other, db_path, output),
            backend_override="stub", root_override=str(other),
        )


# --- legacy databases ------------------------------------------------------

def _legacy_db(db_path: Path, paths: list[Path]) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(LEGACY_SCHEMA)
    for path in paths:
        stat = path.stat()
        conn.execute(
            "INSERT INTO files(path, basename, size_bytes, mtime_ns, width, height, "
            "file_kind, scan_status) VALUES(?, ?, ?, ?, 64, 48, 'jpg', 'done')",
            (str(path), path.name, stat.st_size, stat.st_mtime_ns),
        )
    conn.commit()
    conn.close()


def test_legacy_mixed_database_is_rejected_without_deleting_anything(tmp_path):
    root_2015 = tmp_path / "Photos" / "2015"
    root_2026 = tmp_path / "Photos" / "2026"
    output = tmp_path / "out"
    output.mkdir(parents=True)
    _legacy_db(output / "inventory.sqlite",
               _photos(root_2015, 2, tag=1) + _photos(root_2026, 2, tag=2))
    db_path = output / "inventory.sqlite"
    before = db_path.read_bytes()

    with pytest.raises(ParameterError) as excinfo:
        run_pipeline.run(str(root_2026), str(output), backend="stub")

    message = str(excinfo.value)
    assert "before root identity was recorded" in message
    assert "2 of them outside that root" in message
    assert "left exactly as it is" in message
    assert "common parent" in message
    assert db_path.read_bytes() == before          # no migration, no deletion


def test_legacy_single_root_database_is_adopted_and_migrated(tmp_path):
    root = tmp_path / "Photos" / "2026"
    paths = _photos(root, 3)
    output = tmp_path / "out"
    output.mkdir(parents=True)
    _legacy_db(output / "inventory.sqlite", paths)

    result = run_pipeline.run(str(root), str(output), backend="stub")

    assert result["inventory"]["files"] == 3
    conn = db.open_db(output / "inventory.sqlite")
    keys = {row["path_key"] for row in conn.execute("SELECT path_key FROM files")}
    assert keys == {root_scope.normalize(path) for path in paths}
    assert root_scope.recorded(conn).key == root_scope.normalize(root)
    conn.close()


def test_legacy_parent_root_adoption_keeps_every_row(tmp_path):
    root_2015 = tmp_path / "Photos" / "2015"
    root_2026 = tmp_path / "Photos" / "2026"
    output = tmp_path / "out"
    output.mkdir(parents=True)
    paths = _photos(root_2015, 2, tag=1) + _photos(root_2026, 2, tag=2)
    _legacy_db(output / "inventory.sqlite", paths)

    result = run_pipeline.run(str(tmp_path / "Photos"), str(output), backend="stub")

    assert result["inventory"]["files"] == 4
    assert result["report"]["all_items"] == 4
    assert result["report"]["scope"]["files_out_of_scope"] == 0


# --- run scope -------------------------------------------------------------

def test_current_run_is_recorded_per_file(tmp_path):
    root = tmp_path / "Photos" / "2026"
    _photos(root, 2)
    output = tmp_path / "out"
    first = run_pipeline.run(str(root), str(output), backend="stub")
    first_run = first["report"]["scope"]["run_id"]

    _photos(root, 3)                                # one new file appears
    second = run_pipeline.run(str(root), str(output), backend="stub")

    assert first_run and second["report"]["scope"]["run_id"] != first_run
    assert second["report"]["scope"]["files_in_scope"] == 3
    assert second["report"]["scope"]["files_seen_this_run"] == 3


def test_review_server_scopes_counts_to_the_bound_root(tmp_path):
    root = tmp_path / "Photos" / "2026"
    _photos(root, 3)
    output = tmp_path / "out"
    baseline = run_pipeline.run(str(root), str(output), backend="stub")
    own_groups = baseline["report"]["views"]["GROUPS"]

    # Inject a foreign row + group after the fact; the server must ignore them.
    conn = db.open_db(output / "inventory.sqlite")
    foreign = db.insert_file(conn, {
        "path": r"E:\OtherLibrary\x.jpg", "basename": "x.jpg", "size_bytes": 1,
        "mtime_ns": 1, "width": 10, "height": 10, "file_kind": "jpg",
        "scan_status": "done", "exif_timestamp": 5,
    })
    foreign_group = db.insert_group(conn, "burst", foreign, [(foreign, True, "keep")], 1)
    conn.commit()
    assert db.count_groups(conn) == own_groups + 1          # both exist in the DB
    conn.close()

    server, _url = run_pipeline.start_review_server(output)
    try:
        data = server.review_data
        assert data.scope.key == root_scope.normalize(root)
        assert data.counts()["GROUPS"] == own_groups        # foreign group excluded
        assert data.group(foreign_group) is None
        assert data.original_record(foreign) is None
        assert data.status()["scope"]["root"] == str(root)
    finally:
        server.server_close()
        server.review_data.close()


def test_review_index_rows_outside_the_scope_are_not_counted_or_paged(tmp_path):
    """Defence in depth: a stale index row cannot inflate the ALL view."""
    root = tmp_path / "Photos" / "2026"
    _photos(root, 2)
    output = tmp_path / "out"
    result = run_pipeline.run(str(root), str(output), backend="stub")
    assert result["report"]["all_items"] == 2

    conn = db.open_db(output / "inventory.sqlite")
    foreign = db.insert_file(conn, {
        "path": r"E:\OtherLibrary\y.jpg", "basename": "y.jpg", "size_bytes": 1,
        "mtime_ns": 1, "width": 10, "height": 10, "file_kind": "jpg",
        "scan_status": "done", "exif_timestamp": 7,
    })
    conn.execute(
        "INSERT INTO review_index (view, position, file_id, group_id, decision, risk) "
        "VALUES ('ALL', 99, ?, NULL, NULL, NULL)", (foreign,)
    )
    conn.commit()
    scope = root_scope.recorded(conn)
    assert db.review_index_count(conn, "ALL") == 3                    # raw index
    assert db.review_index_count(conn, "ALL", scope=scope) == 2        # scoped truth
    paged = db.review_page(conn, "ALL", 0, 50, scope=scope)
    assert foreign not in [int(row["file_id"]) for row in paged]
    conn.close()
