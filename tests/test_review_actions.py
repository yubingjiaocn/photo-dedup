"""Human review actions: state persistence, API validation, no original writes."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from src import db, review_server, review_state, thumbnails


def _build_output_with_groups(tmp_path: Path, count: int = 10) -> tuple[Path, Path]:
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


def _served(output: Path):
    server, url = review_server.start_server(output)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, url.rsplit("/", 1)[0]


class _Served:
    def __init__(self, output: Path) -> None:
        self.output = output

    def __enter__(self) -> str:
        self.server, self.thread, self.base = _served(self.output)
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


# --- state persistence ------------------------------------------------------

def test_review_state_persists_and_survives_restart(tmp_path):
    """State is written atomically to JSON and reloaded."""
    output, _photos = _build_output_with_groups(tmp_path, 6)
    state = review_state.ReviewState(output)

    state.set_group(1, {"action": "accept", "timestamp": 1234567890})
    state.set_group(2, {"action": "pick", "file_id": 5, "timestamp": 1234567891})
    state.set_group(3, {"action": "mark", "timestamp": 1234567892})

    assert state.path.is_file()
    with state.path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == 1
    assert data["groups"]["1"]["action"] == "accept"
    assert data["groups"]["2"]["file_id"] == 5
    assert data["groups"]["3"]["action"] == "mark"

    # Reload
    state2 = review_state.ReviewState(output)
    assert state2.get_group(1)["action"] == "accept"
    assert state2.get_group(2)["file_id"] == 5
    assert state2.get_group(3)["action"] == "mark"
    assert state2.get_group(999) is None

    summary = state2.summary()
    assert summary["reviewed"] == 2  # accept + pick
    assert summary["marked"] == 1
    assert summary["total"] == 3


def test_review_state_clear_removes_and_persists(tmp_path):
    output, _photos = _build_output_with_groups(tmp_path, 4)
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1234567890})
    assert state.get_group(1) is not None

    state.clear_group(1)
    assert state.get_group(1) is None

    state2 = review_state.ReviewState(output)
    assert state2.get_group(1) is None


def test_review_state_ignores_corrupt_file(tmp_path):
    """Corrupt state file is left intact and not loaded."""
    output, _photos = _build_output_with_groups(tmp_path, 2)
    state_path = output / "review_state.json"
    state_path.write_text("not valid json", encoding="utf-8")

    state = review_state.ReviewState(output)
    assert state.summary()["total"] == 0
    assert state_path.read_text(encoding="utf-8") == "not valid json"


# --- API validation ---------------------------------------------------------

def test_action_endpoint_validates_group_in_scope(tmp_path):
    output, _photos = _build_output_with_groups(tmp_path, 4)
    with _Served(output) as base:
        # Valid group
        result = _post(base, {"group_id": 1, "action": "accept"})
        assert result.get("ok") is True

        # Invalid group
        try:
            _post(base, {"group_id": 999, "action": "accept"})
            assert False, "should have failed"
        except HTTPError as e:
            assert e.code == 400


def test_action_endpoint_validates_file_id_is_group_member(tmp_path):
    output, _photos = _build_output_with_groups(tmp_path, 6)
    with _Served(output) as base:
        # Valid pick
        result = _post(base, {"group_id": 1, "file_id": 1, "action": "pick"})
        assert result.get("ok") is True

        # file_id not in group
        try:
            _post(base, {"group_id": 1, "file_id": 5, "action": "pick"})
            assert False, "should have failed"
        except HTTPError as e:
            assert e.code == 400


def test_action_endpoint_rejects_invalid_actions(tmp_path):
    output, _photos = _build_output_with_groups(tmp_path, 2)
    with _Served(output) as base:
        for bad in ({"group_id": 1, "action": "delete"},
                    {"group_id": 1, "action": ""},
                    {"action": "accept"},  # missing group_id
                    {"group_id": "not_int", "action": "accept"}):
            try:
                _post(base, bad)
                assert False, f"should have rejected: {bad}"
            except HTTPError as e:
                assert e.code == 400


def test_action_endpoint_rejects_oversized_payloads(tmp_path):
    output, _photos = _build_output_with_groups(tmp_path, 2)
    with _Served(output) as base:
        big = json.dumps({"group_id": 1, "action": "accept", "junk": "x" * 10000}).encode("utf-8")
        req = Request(f"{base}/api/action", data=big, method="POST",
                      headers={"Content-Type": "application/json"})
        try:
            urlopen(req)
            assert False, "should have rejected"
        except HTTPError as e:
            assert e.code == 400


# --- integration ------------------------------------------------------------

def test_review_actions_round_trip_through_api(tmp_path):
    output, _photos = _build_output_with_groups(tmp_path, 8)
    with _Served(output) as base:
        status0 = _get(base, "/api/status")
        assert status0["review_state"]["total"] == 0

        # Accept AI keeper
        r1 = _post(base, {"group_id": 1, "action": "accept"})
        assert r1["ok"] is True
        assert r1["summary"]["reviewed"] == 1

        # Pick custom keeper
        r2 = _post(base, {"group_id": 2, "file_id": 4, "action": "pick"})
        assert r2["ok"] is True
        assert r2["summary"]["reviewed"] == 2

        # Mark for later
        r3 = _post(base, {"group_id": 3, "action": "mark"})
        assert r3["ok"] is True
        assert r3["summary"]["marked"] == 1

        # Clear
        r4 = _post(base, {"group_id": 1, "action": "clear"})
        assert r4["ok"] is True
        assert r4["summary"]["reviewed"] == 1  # only group 2 left
        assert r4["summary"]["total"] == 2  # groups 2 and 3

        status = _get(base, "/api/status")
        assert status["review_state"]["reviewed"] == 1
        assert status["review_state"]["marked"] == 1


def test_state_file_never_contains_original_paths(tmp_path, monkeypatch):
    """State records group_id and file_id only, never source paths."""
    output, photos = _build_output_with_groups(tmp_path, 4)
    originals = {str(p) for p in photos.iterdir()}

    with _Served(output) as base:
        _post(base, {"group_id": 1, "file_id": 2, "action": "pick"})
        _post(base, {"group_id": 2, "action": "accept"})

    state_path = output / "review_state.json"
    assert state_path.is_file()
    state_text = state_path.read_text(encoding="utf-8")
    for original in originals:
        assert original not in state_text
        assert Path(original).name not in state_text  # basename also not exposed


# --- no original writes -----------------------------------------------------

def test_review_actions_never_write_to_original_photos(tmp_path, monkeypatch):
    """Actions only write to output dir, never touch source photos."""
    output, photos = _build_output_with_groups(tmp_path, 6)
    originals = {str(p) for p in photos.iterdir()}
    writes: list[str] = []

    real_open = Path.open

    def tracked_open(self, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode:
            if str(self) in originals:
                writes.append(str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        _post(base, {"group_id": 2, "file_id": 4, "action": "pick"})
        _post(base, {"group_id": 3, "action": "mark"})
        _post(base, {"group_id": 1, "action": "clear"})

    assert writes == [], f"wrote to originals: {writes}"


def test_review_state_module_never_opens_original_photos(tmp_path, monkeypatch):
    """validate_action uses DB only, never accesses source files."""
    output, photos = _build_output_with_groups(tmp_path, 4)
    originals = {str(p) for p in photos.iterdir()}
    opened: list[str] = []

    real_open = Path.open

    def tracked_open(self, *args, **kwargs):
        if str(self) in originals:
            opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    conn = db.open_db(output / "inventory.sqlite")
    try:
        assert review_state.validate_action(conn, 1, None, "accept", None) is None
        assert review_state.validate_action(conn, 1, 1, "pick", None) is None
        assert review_state.validate_action(conn, 999, None, "accept", None) is not None
    finally:
        conn.close()

    assert opened == []


# --- P0 fixes ---------------------------------------------------------------

def test_review_state_concurrent_writes_all_preserved(tmp_path):
    """P0-4: Concurrent writes must not lose updates."""
    output, _photos = _build_output_with_groups(tmp_path, 20)
    state = review_state.ReviewState(output)

    from concurrent.futures import ThreadPoolExecutor
    results = []

    def write_group(group_id):
        try:
            state.set_group(group_id, {"action": "accept", "timestamp": group_id * 1000})
            return (group_id, "ok")
        except Exception as e:
            return (group_id, str(e))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_group, gid) for gid in range(1, 11)]
        results = [f.result() for f in futures]

    # All writes succeeded
    assert all(status == "ok" for _, status in results)

    # Reload and verify all are present
    state2 = review_state.ReviewState(output)
    for gid in range(1, 11):
        decision = state2.get_group(gid)
        assert decision is not None, f"group {gid} lost"
        assert decision["action"] == "accept"
        assert decision["timestamp"] == gid * 1000


def test_review_state_concurrent_reads_while_writing(tmp_path):
    """P0-4: Reads during writes must not crash or see corrupt data."""
    output, _photos = _build_output_with_groups(tmp_path, 10)
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1234})

    from concurrent.futures import ThreadPoolExecutor
    import time

    def reader():
        for _ in range(50):
            summary = state.summary()
            assert isinstance(summary["total"], int)
            time.sleep(0.001)

    def writer():
        for gid in range(2, 6):
            state.set_group(gid, {"action": "accept", "timestamp": gid * 100})
            time.sleep(0.005)

    with ThreadPoolExecutor(max_workers=3) as executor:
        r_futures = [executor.submit(reader) for _ in range(2)]
        w_future = executor.submit(writer)
        w_future.result()
        for f in r_futures:
            f.result()

    summary = state.summary()
    assert summary["total"] == 5
    assert summary["reviewed"] == 5


def test_page_endpoint_includes_review_state_on_load(tmp_path):
    """P0-1: /api/page must include review_state so UI can render on first load."""
    output, _photos = _build_output_with_groups(tmp_path, 6)
    conn = db.open_db(output / "inventory.sqlite")
    fp1 = review_state.compute_member_fingerprint([1, 2])
    fp2 = review_state.compute_member_fingerprint([3, 4])
    conn.close()
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1234}, fp1)
    state.set_group(2, {"action": "pick", "file_id": 4, "timestamp": 1235}, fp2)

    with _Served(output) as base:
        page_data = _get(base, "/api/page?view=GROUPS&page=1&page_size=100")
        assert "review_state" in page_data
        # review_state should be a dict keyed by group_id
        rs = page_data["review_state"]
        assert isinstance(rs, dict)
        assert 1 in rs or "1" in str(rs)  # Can be int or str key from JSON
        # Check it has the right shape
        for gid in [1, 2]:
            gid_key = gid if gid in rs else str(gid)
            if gid_key in rs:
                assert rs[gid_key]["action"] in ("accept", "pick")


def test_review_state_validates_loaded_data(tmp_path):
    """P0-8: Load must reject invalid action/file_id combinations."""
    output, _photos = _build_output_with_groups(tmp_path, 4)
    state_path = output / "review_state.json"

    # Write state with invalid data
    bad_data = {
        "version": 1,
        "groups": {
            "1": {"action": "accept", "file_id": 999},  # accept must not have file_id
            "2": {"action": "pick"},  # pick requires file_id
            "3": {"action": "invalid_action", "timestamp": 123},
            "4": {"action": "accept", "timestamp": 456},  # valid
        }
    }
    state_path.write_text(json.dumps(bad_data), encoding="utf-8")

    # Load should only keep valid entry
    state = review_state.ReviewState(output)
    summary = state.summary()
    assert summary["total"] == 1
    assert state.get_group(4) is not None
    assert state.get_group(1) is None
    assert state.get_group(2) is None
    assert state.get_group(3) is None


def test_review_state_get_returns_immutable_copy(tmp_path):
    """P0-4: get_group must return copy so caller mutations don't affect internal state."""
    output, _photos = _build_output_with_groups(tmp_path, 2)
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1234})

    decision1 = state.get_group(1)
    decision1["action"] = "tampered"
    decision1["extra"] = "injected"

    # Internal state must be unchanged
    decision2 = state.get_group(1)
    assert decision2["action"] == "accept"
    assert "extra" not in decision2
