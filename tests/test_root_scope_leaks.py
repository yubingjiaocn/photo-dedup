"""Can a row outside the bound root still reach a consumer?

The binding gate (``test_root_scope_gates.py``) stops the *run*. This file
assumes the gate was passed legitimately and asks the next question: given a
database that nonetheless contains a foreign row -- left over from a wider run,
written by external tooling, or produced by a bug elsewhere -- can that row
surface anywhere a user or a manifest would see it?

The dangerous shape is a **straddling group**: one cluster with members in two
roots, where the foreign member is marked ``AUTO_REMOVE`` and is the motion
partner of an in-scope file. Every consumer is checked: the ``db`` join, the
delete manifest and its motion-partner expansion, the review API (pages, group
lookup, thumbnails, originals), and the files Stage 3 writes.

Fixtures live in ``root_scope_fixtures.py``.
"""

from __future__ import annotations

from src import db, review_server, stage3_report, thumbnails
from tests import root_scope_fixtures as fixtures

# --- a group straddling two roots ------------------------------------------

def test_straddling_group_members_and_partners_are_invisible_in_the_db_layer(tmp_path):
    case = fixtures.straddling_library(tmp_path)
    conn = db.open_db(case["db_path"])
    try:
        scope = case["scope"]
        members = db.group_members(conn, case["group_id"], scope=scope)
        assert [int(row["file_id"]) for row in members] == [case["keeper"], case["mine"]]
        assert db.get_file(conn, case["theirs"], scope=scope) is None
        assert db.get_file(conn, case["mine"], scope=scope) is not None
        # Unscoped access still sees all three -- the filter is the scope, not
        # deletion: this database is left intact.
        assert len(db.group_members(conn, case["group_id"])) == 3
    finally:
        conn.close()


def test_straddling_group_cannot_leak_into_the_delete_manifest(tmp_path):
    case = fixtures.straddling_library(tmp_path)
    conn = db.open_db(case["db_path"])
    try:
        scoped = stage3_report.collect_deletions(conn, scope=case["scope"])
        unscoped = stage3_report.collect_deletions(conn)
    finally:
        conn.close()

    foreign = str(case["theirs_path"])
    # The danger is real: without the scope the foreign file *is* listed, both as
    # a member decision and through motion-partner expansion.
    assert foreign in unscoped["delete_paths"]

    assert foreign not in scoped["delete_paths"]
    assert str(case["mine_path"]) in scoped["delete_paths"]      # in-scope work kept
    assert case["theirs_path"].name not in [
        item["filename"] for item in scoped["cloud_items"]
    ]
    for group in scoped["groups"]:
        for record in group["deletes"]:
            assert int(record["file_id"]) != case["theirs"]


def test_straddling_group_cannot_leak_through_the_review_server(tmp_path):
    case = fixtures.straddling_library(tmp_path)
    conn = db.open_db(case["db_path"])
    try:
        stage3_report.build_review_indexes(
            conn, stage3_report.collect_deletions(conn, scope=case["scope"]),
            scope=case["scope"],
        )
        conn.commit()
    finally:
        conn.close()
    # A thumbnail cached for the foreign file (a previous, wider run) must not be
    # reachable either, even though the JPEG is sitting in the cache directory.
    thumbs = thumbnails.thumbs_dir(case["output"])
    thumbs.mkdir(parents=True, exist_ok=True)
    for file_id in (case["keeper"], case["theirs"]):
        thumbnails.thumb_path(thumbs, file_id).write_bytes(b"\xff\xd8\xff fake jpeg")

    data = review_server.ReviewData(case["output"])
    try:
        group = data.group(case["group_id"])
        assert group is not None
        assert [item["file_id"] for item in group["members"]] == [
            case["keeper"], case["mine"]
        ]
        assert group["member_count"] == 2               # header matches the tiles
        page = data.page("GROUPS", 1, 50)
        for entry in page["items"]:
            assert case["theirs"] not in [item["file_id"] for item in entry["members"]]
        assert data.in_scope(case["keeper"]) is True
        assert data.in_scope(case["theirs"]) is False
        assert data.thumb_bytes(case["keeper"]) is not None
        assert data.thumb_bytes(case["theirs"]) is None      # cached but out of scope
        assert data.original_record(case["theirs"]) is None
        assert data.original_record(case["keeper"]) is not None
        for view in ("ALL", "MAYBE", "UNKNOWN"):
            ids = [item["file_id"] for item in data.page(view, 1, 50)["items"]]
            assert case["theirs"] not in ids
    finally:
        data.close()


def test_straddling_group_is_absent_from_the_stage3_manifest_files(tmp_path):
    case = fixtures.straddling_library(tmp_path)
    report_dir = tmp_path / "reports"
    config = fixtures.write_config(tmp_path, "straddle", case["db_path"], report_dir,
                     root=case["root"])

    stage3_report.run(config_path=config)

    local = (report_dir / "delete_local.txt").read_text(encoding="utf-8")
    cloud = (report_dir / "delete_cloud.json").read_text(encoding="utf-8")
    summary = (report_dir / "summary.txt").read_text(encoding="utf-8")
    foreign_name = case["theirs_path"].name
    assert str(case["theirs_path"]) not in local
    assert foreign_name not in local
    assert foreign_name not in cloud
    assert "2015" not in local
    scope_line = [line for line in summary.splitlines() if "scope:" in line]
    assert scope_line and str(case["root"]) in scope_line[0]
