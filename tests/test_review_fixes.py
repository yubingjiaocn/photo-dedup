"""Tests for keyboard review workbench fixes (issues 2-10)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from src import db, review_server, review_state, thumbnails


def _build_output_with_groups(tmp_path: Path, count: int = 20) -> tuple[Path, Path]:
    """Output dir with inventory, groups, and thumbnails."""
    output = tmp_path / "output"
    photos = tmp_path / "photos"
    output.mkdir(parents=True)
    photos.mkdir(parents=True)
    thumbs = thumbnails.thumbs_dir(output)
    thumbs.mkdir(parents=True, exist_ok=True)
    conn = db.open_db(output / "inventory.sqlite")

    file_ids = []
    for index in range(count):
        source = photos / f"IMG_{index:04d}.jpg"
        Image.new("RGB", (3, 2), (index % 255, 20, 30)).save(source, "JPEG")
        stat = source.stat()
        file_id = db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "exif_datetime": f"2026-01-01 12:00:{index:02d}",
            "exif_timestamp": 1000 + index, "width": 4000, "height": 3000,
            "file_kind": "jpg", "scan_status": "done",
        })
        file_ids.append(file_id)
        db.batch_insert_features(conn, [{
            "file_id": file_id, "phash": None, "dinov2_embedding": None,
            "quality_score": 50.0 + index, "quality_meta": "{}", "face_count": 1,
            "faces_json": "[]", "status": "done",
        }])
        Image.new("RGB", (32, 24), (index % 255, 40, 90)).save(
            thumbnails.thumb_path(thumbs, file_id), "JPEG")
        db.batch_upsert_thumbnails(conn, [{
            "file_id": file_id, "status": "ok", "max_px": 320, "bytes": 900,
            "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "error": None, "created_at": 1,
        }])

    # Create groups: 0=(1,2), 1=(3,4), 2=(5,6)...
    for group_num in range(count // 2):
        keeper = file_ids[group_num * 2]
        candidate = file_ids[group_num * 2 + 1]
        group_id = db.insert_group(conn, "burst", keeper,
                                   [(keeper, True, "keep"), (candidate, False, "dup")], 1)
        db.update_member_decisions(conn, group_id, [
            {"file_id": keeper, "decision": "KEEP", "confidence": 1.0,
             "reason": "best", "evidence_json": "{}"},
            {"file_id": candidate, "decision": "MAYBE", "confidence": 0.5,
             "reason": "near duplicate", "evidence_json": "{}"},
        ])

    db.build_all_view_index(conn)
    conn.commit()
    conn.close()

    (output / "review.html").write_text('<h1>Review</h1>', encoding="utf-8")
    (output / "review_summary.json").write_text(json.dumps({"views": {}}), encoding="utf-8")
    return output, photos


class _Served:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.server, self.url = review_server.start_server(output)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = self.url.rsplit("/", 1)[0]

    def __enter__(self) -> str:
        return self.base

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.review_data.close()
        self.thread.join()


def _post(base: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(f"{base}/api/action", data=body, method="POST",
                  headers={"Content-Type": "application/json"})
    return json.loads(urlopen(req).read())


def _get(base: str, path: str) -> dict:
    return json.loads(urlopen(base + path).read())


# Issue 2: state overlay should be visible with current page groups
def test_groups_page_returns_current_page_state(tmp_path):
    """Page response includes review_state for visible groups."""
    output, _photos = _build_output_with_groups(tmp_path, 10)
    fp1 = review_state.compute_member_fingerprint([1, 2])
    fp2 = review_state.compute_member_fingerprint([3, 4])
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1234567890}, fp1)
    state.set_group(2, {"action": "mark", "timestamp": 1234567891}, fp2)

    with _Served(output) as base:
        # Group 1 was accepted and group 2 marked, so they live in the DONE and
        # LATER queues; each page carries the state of its own groups.
        done = _get(base, "/api/page?view=GROUPS&queue=DONE&page=1&page_size=100")
        later = _get(base, "/api/page?view=GROUPS&queue=LATER&page=1&page_size=100")
        assert "review_state" in done and "review_state" in later
        assert {str(k) for k in done["review_state"]} == {"1"}
        assert {str(k) for k in later["review_state"]} == {"2"}
        assert list(done["review_state"].values())[0]["action"] == "accept"
        assert list(later["review_state"].values())[0]["action"] == "mark"


# Issue 7: ReviewState concurrent access safety
def test_review_state_concurrent_set_does_not_lose_writes(tmp_path):
    """Multiple threads setting state concurrently do not lose updates."""
    output, _photos = _build_output_with_groups(tmp_path, 20)
    state = review_state.ReviewState(output)

    def writer(group_id: int):
        for _ in range(5):
            state.set_group(group_id, {"action": "accept", "timestamp": int(time.time())})
            time.sleep(0.001)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(1, 6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All groups should be recorded
    for gid in range(1, 6):
        assert state.get_group(gid) is not None
        assert state.get_group(gid)["action"] == "accept"


# Issue 8: strict payload validation
def test_action_endpoint_rejects_file_id_in_accept_mark_clear(tmp_path):
    """accept/mark/clear must not include file_id."""
    output, _photos = _build_output_with_groups(tmp_path, 4)
    with _Served(output) as base:
        for action in ("accept", "mark", "clear"):
            try:
                _post(base, {"group_id": 1, "file_id": 1, "action": action})
                assert False, f"{action} should reject file_id"
            except HTTPError as e:
                assert e.code == 400


def test_action_endpoint_rejects_bool_as_int(tmp_path):
    """True/False masquerading as int are rejected."""
    output, _photos = _build_output_with_groups(tmp_path, 4)
    with _Served(output) as base:
        # group_id must not accept bool
        try:
            _post(base, {"group_id": True, "action": "accept"})
            assert False, "should reject bool group_id"
        except HTTPError as e:
            assert e.code == 400

        # file_id must not accept bool in pick
        try:
            _post(base, {"group_id": 1, "file_id": False, "action": "pick"})
            assert False, "should reject bool file_id"
        except HTTPError as e:
            assert e.code == 400


def test_action_endpoint_rejects_malformed_utf8_json(tmp_path):
    """Malformed UTF-8 or invalid JSON return stable 400."""
    output, _photos = _build_output_with_groups(tmp_path, 2)
    with _Served(output) as base:
        # Invalid UTF-8
        req = Request(f"{base}/api/action", data=b"\xff\xfe invalid",
                      method="POST", headers={"Content-Type": "application/json"})
        try:
            urlopen(req)
            assert False
        except HTTPError as e:
            assert e.code == 400

        # Malformed JSON
        req = Request(f"{base}/api/action", data=b"{not valid json}",
                      method="POST", headers={"Content-Type": "application/json"})
        try:
            urlopen(req)
            assert False
        except HTTPError as e:
            assert e.code == 400


# Issue 9: stale out-of-scope groups not counted
def test_status_filters_stale_groups_from_summary(tmp_path):
    """After restart with different scope, stale state not counted."""
    output, _photos = _build_output_with_groups(tmp_path, 10)

    # Record decisions for groups 1-3 with correct fingerprints
    fp1 = review_state.compute_member_fingerprint([1, 2])
    fp2 = review_state.compute_member_fingerprint([3, 4])
    fp3 = review_state.compute_member_fingerprint([5, 6])
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1234567890}, fp1)
    state.set_group(2, {"action": "mark", "timestamp": 1234567891}, fp2)
    state.set_group(3, {"action": "pick", "file_id": 5, "timestamp": 1234567892}, fp3)

    # Now simulate group 3 is out of scope (delete its members from DB)
    conn = db.open_db(output / "inventory.sqlite")
    conn.execute("DELETE FROM group_members WHERE group_id = 3")
    conn.commit()
    conn.close()

    # Status should only count groups 1 and 2
    with _Served(output) as base:
        status = _get(base, "/api/status")
        assert status["review_state"]["total"] == 2
        assert status["review_state"]["reviewed"] == 1  # only group 1
        assert status["review_state"]["marked"] == 1  # only group 2


# Issue 10: validate state at load
def test_review_state_rejects_invalid_schema_at_load(tmp_path):
    """State file with bool file_id or wrong action/file_id combo is filtered."""
    output, _photos = _build_output_with_groups(tmp_path, 4)
    state_path = output / "review_state.json"

    # Write state with violations
    bad_state = {
        "version": 1,
        "groups": {
            "1": {"action": "pick", "file_id": True},  # bool not int
            "2": {"action": "accept", "file_id": 5},  # accept should not have file_id
            "3": {"action": "mark", "timestamp": 123},  # valid
        },
    }
    state_path.write_text(json.dumps(bad_state), encoding="utf-8")

    # Load should filter out groups 1 and 2
    state = review_state.ReviewState(output)
    assert state.get_group(1) is None
    assert state.get_group(2) is None
    assert state.get_group(3) is not None
    assert state.get_group(3)["action"] == "mark"
