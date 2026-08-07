"""Server-side gates for the review queues, their totals and undo.

What these pin down, all through the public API or the ``ReviewData`` object the
handler uses:

* the group queues (``PENDING`` / ``LATER`` / ``DONE``) are computed and paged
  **on the server**, with each queue reporting its own ``total`` and ``pages``;
* the state machine: undecided -> accept/pick leaves the pending queue for DONE,
  mark moves it to LATER, deciding from LATER leaves for DONE, and clearing
  brings a group back to PENDING;
* ``undo`` restores the exact previous state of the most recently decided group
  -- including "no decision at all" -- survives a restart, and refuses to reach
  outside the current root scope;
* ``/api/locate`` answers "which page holds this group" and degrades to the next
  survivor when the group has left the queue;
* none of it ever opens, moves or writes an original photo, and the AI decisions
  and delete manifests are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from src import db, review_server, review_state, root_scope, thumbnails


def _build(tmp_path: Path, groups: int = 6, photos_root: str = "photos") -> Path:
    """Output dir bound to one photo root, holding ``groups`` groups of two."""
    output = tmp_path / "output"
    photos = tmp_path / photos_root
    output.mkdir(parents=True, exist_ok=True)
    photos.mkdir(parents=True, exist_ok=True)
    thumbs = thumbnails.thumbs_dir(output)
    thumbs.mkdir(parents=True, exist_ok=True)
    conn = db.open_db(output / "inventory.sqlite")
    root_scope.bind(conn, photos)

    file_ids = []
    for index in range(groups * 2):
        source = photos / f"IMG_{index:04d}.jpg"
        Image.new("RGB", (4, 3), (index % 255, 20, 30)).save(source, "JPEG")
        stat = source.stat()
        file_id = db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "exif_datetime": f"2026-01-01 12:00:{index:02d}",
            "exif_timestamp": 1000 + index, "width": 40, "height": 30,
            "file_kind": "jpg", "scan_status": "done"})
        file_ids.append(file_id)
        db.batch_insert_features(conn, [{
            "file_id": file_id, "phash": None, "dinov2_embedding": None,
            "quality_score": 50.0 + index, "quality_meta": "{}", "face_count": 1,
            "faces_json": "[]", "status": "done"}])
        Image.new("RGB", (8, 6), (index % 255, 40, 90)).save(
            thumbnails.thumb_path(thumbs, file_id), "JPEG")
        db.batch_upsert_thumbnails(conn, [{
            "file_id": file_id, "status": "ok", "max_px": 320, "bytes": 90,
            "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "error": None, "created_at": 1}])

    for number in range(groups):
        keeper, dup = file_ids[number * 2], file_ids[number * 2 + 1]
        group_id = db.insert_group(conn, "burst", keeper,
                                   [(keeper, True, "keep"), (dup, False, "dup")], 1)
        db.update_member_decisions(conn, group_id, [
            {"file_id": keeper, "decision": "KEEP", "confidence": 1.0,
             "reason": "GROUP_KEEPER", "evidence_json": "{}"},
            {"file_id": dup, "decision": "MAYBE", "confidence": 0.5,
             "reason": "LOW_MARGIN", "evidence_json": "{}"}])
    db.build_all_view_index(conn, scope=None)
    conn.commit()
    conn.close()

    (output / "review.html").write_text("<h1>Review</h1>", encoding="utf-8")
    (output / "review_summary.json").write_text(json.dumps({"views": {}}),
                                                encoding="utf-8")
    return output


class _Served:
    def __init__(self, output: Path) -> None:
        self.output = output

    def __enter__(self) -> str:
        self.server, url = review_server.start_server(self.output)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return url.rsplit("/", 1)[0]

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.review_data.close()
        self.thread.join(timeout=5)


def _get(base: str, path: str) -> dict:
    return json.loads(urlopen(base + path).read())


def _post(base: str, payload: dict) -> dict:
    request = Request(f"{base}/api/action", data=json.dumps(payload).encode("utf-8"),
                      method="POST", headers={"Content-Type": "application/json"})
    return json.loads(urlopen(request).read())


def _queue_members_of(output: Path, gid: int) -> list[int]:
    """Current member file ids of ``gid``, read straight from the DB."""
    conn = db.open_db(output / "inventory.sqlite")
    try:
        return [int(row["file_id"]) for row in conn.execute(
            "SELECT file_id FROM group_members WHERE group_id = ? ORDER BY file_id",
            (gid,))]
    finally:
        conn.close()

def _queue(base: str, name: str, page: int = 1, size: int = 100) -> dict:
    return _get(base, f"/api/page?view=GROUPS&queue={name}&page={page}&page_size={size}")


# --- queue membership and totals -------------------------------------------

def test_everything_starts_in_the_pending_queue(tmp_path):
    output = _build(tmp_path, groups=6)
    with _Served(output) as base:
        pending = _queue(base, "PENDING")
        assert pending["queue"] == "PENDING"
        assert pending["total"] == 6 and pending["shown"] == 6
        assert pending["queue_counts"] == {"PENDING": 6, "LATER": 0, "DONE": 0}
        assert _queue(base, "LATER")["total"] == 0
        assert _queue(base, "DONE")["total"] == 0
        assert _get(base, "/api/status")["queues"] == {"PENDING": 6, "LATER": 0,
                                                      "DONE": 0}


def test_accept_and_pick_move_a_group_from_pending_to_done(tmp_path):
    output = _build(tmp_path, groups=4)
    with _Served(output) as base:
        first = _queue(base, "PENDING")["items"][0]
        gid = first["group_id"]
        keeper = next(m["file_id"] for m in first["members"] if m["is_keep"])

        response = _post(base, {"group_id": gid, "action": "accept"})
        assert response["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 1}
        pending = _queue(base, "PENDING")
        assert pending["total"] == 3
        assert gid not in [g["group_id"] for g in pending["items"]]
        done = _queue(base, "DONE")
        assert [g["group_id"] for g in done["items"]] == [gid]

        second = pending["items"][0]["group_id"]
        response = _post(base, {"group_id": second, "file_id": keeper + 2,
                                "action": "pick"})
        assert response["queues"] == {"PENDING": 2, "LATER": 0, "DONE": 2}
        assert _queue(base, "DONE")["total"] == 2


def test_mark_moves_pending_to_later_and_deciding_there_moves_it_to_done(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        gid = _queue(base, "PENDING")["items"][0]["group_id"]
        assert _post(base, {"group_id": gid, "action": "mark"})["queues"] == {
            "PENDING": 2, "LATER": 1, "DONE": 0}
        later = _queue(base, "LATER")
        assert [g["group_id"] for g in later["items"]] == [gid]
        assert str(gid) in {str(k) for k in later["review_state"]}

        assert _post(base, {"group_id": gid, "action": "accept"})["queues"] == {
            "PENDING": 2, "LATER": 0, "DONE": 1}
        assert _queue(base, "LATER")["total"] == 0
        assert [g["group_id"] for g in _queue(base, "DONE")["items"]] == [gid]


def test_clear_returns_a_group_to_the_pending_queue(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        gid = _queue(base, "PENDING")["items"][0]["group_id"]
        _post(base, {"group_id": gid, "action": "accept"})
        assert _post(base, {"group_id": gid, "action": "clear"})["queues"] == {
            "PENDING": 3, "LATER": 0, "DONE": 0}
        assert gid in [g["group_id"] for g in _queue(base, "PENDING")["items"]]


def test_a_done_group_is_still_viewable_with_its_members_and_state(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        first = _queue(base, "PENDING")["items"][0]
        gid = first["group_id"]
        chosen = first["members"][1]["file_id"]
        _post(base, {"group_id": gid, "file_id": chosen, "action": "pick"})
        done = _queue(base, "DONE")
        group = done["items"][0]
        assert group["group_id"] == gid
        assert len(group["members"]) == 2
        state = done["review_state"][str(gid)]
        assert state["action"] == "pick" and state["file_id"] == chosen


# --- paging the queues -----------------------------------------------------

@pytest.mark.parametrize("size,expected_pages", [(50, 2), (100, 1), (200, 1)])
def test_queue_paging_uses_the_queue_total_not_the_group_total(tmp_path, size,
                                                               expected_pages):
    output = _build(tmp_path, groups=60)
    with _Served(output) as base:
        for gid in range(1, 6):
            _post(base, {"group_id": gid, "action": "accept"})
        pending = _queue(base, "PENDING", size=size)
        assert pending["total"] == 55
        assert pending["pages"] == expected_pages
        assert pending["group_total"] == 60           # the whole library, for context
        assert pending["shown"] == min(size, 55)
        done = _queue(base, "DONE", size=size)
        assert done["total"] == 5 and done["pages"] == 1


def test_queue_pages_are_dense_complete_and_never_overlap(tmp_path):
    output = _build(tmp_path, groups=120)
    with _Served(output) as base:
        for gid in (5, 17, 90):
            _post(base, {"group_id": gid, "action": "mark"})
        seen: list[int] = []
        page = 1
        while True:
            data = _queue(base, "PENDING", page=page, size=50)
            seen.extend(g["group_id"] for g in data["items"])
            if page >= data["pages"]:
                break
            page += 1
        assert len(seen) == len(set(seen)) == 117
        assert set(seen).isdisjoint({5, 17, 90})
        assert seen == sorted(seen)


def test_a_page_beyond_the_end_is_clamped_and_an_empty_queue_still_answers(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        clamped = _queue(base, "PENDING", page=99)
        assert clamped["page"] == 1 and clamped["pages"] == 1
        empty = _queue(base, "DONE", page=7)
        assert empty["page"] == 1 and empty["pages"] == 1
        assert empty["total"] == 0 and empty["items"] == []


def test_an_unknown_queue_or_page_size_is_rejected(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        for path in ("/api/page?view=GROUPS&queue=EVIL",
                     "/api/page?view=GROUPS&queue=PENDING&page_size=7",
                     "/api/page?view=GROUPS&queue=PENDING&page=0",
                     "/api/locate?queue=EVIL&group_id=1",
                     "/api/locate?queue=PENDING&group_id=1&page_size=7"):
            with pytest.raises(HTTPError) as excinfo:
                _get(base, path)
            assert excinfo.value.code == 400


def test_queue_filtering_does_not_join_the_groups_it_excludes(tmp_path):
    """The filter must be applied before the member join, not after it."""
    output = _build(tmp_path, groups=8)
    data = review_server.ReviewData(output)
    try:
        for gid in range(1, 7):
            data.apply_action({"group_id": gid, "action": "accept"})
        calls: list[list[int]] = []
        original = db.group_page_by_ids

        def spy(conn, group_ids, scope=None):
            calls.append([int(gid) for gid in group_ids])
            return original(conn, group_ids, scope=scope)

        db.group_page_by_ids = spy
        try:
            page = data.page("GROUPS", 1, 50, "PENDING")
        finally:
            db.group_page_by_ids = original
        assert page["total"] == 2
        # Exactly the two pending groups were ever expanded into member rows.
        assert calls == [[7, 8]]
    finally:
        data.close()


# --- locate ----------------------------------------------------------------

def test_locate_reports_the_page_holding_a_group(tmp_path):
    output = _build(tmp_path, groups=130)
    with _Served(output) as base:
        found = _get(base, "/api/locate?queue=PENDING&group_id=120&page_size=50")
        assert found["found"] is True
        assert found["group_id"] == 120 and found["index"] == 119
        assert found["page"] == 3 and found["total"] == 130


def test_locate_falls_through_to_the_next_survivor_when_the_group_left(tmp_path):
    output = _build(tmp_path, groups=6)
    with _Served(output) as base:
        _post(base, {"group_id": 3, "action": "accept"})
        found = _get(base, "/api/locate?queue=PENDING&group_id=3")
        assert found["found"] is False
        # Group 3 is gone from PENDING, so the next one takes its place.
        assert found["group_id"] == 4 and found["page"] == 1
        # Past the end, the answer is clamped to the last surviving entry.
        for gid in (4, 5, 6):
            _post(base, {"group_id": gid, "action": "accept"})
        clamped = _get(base, "/api/locate?queue=PENDING&group_id=6")
        assert clamped["found"] is False and clamped["group_id"] == 2


def test_locate_on_an_empty_queue_reports_nothing_to_land_on(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        empty = _get(base, "/api/locate?queue=DONE&group_id=1")
        assert empty == {"queue": "DONE", "total": 0, "found": False, "page": 1,
                         "index": None, "group_id": None, "page_size": 100,
                         "queue_counts": {"PENDING": 2, "LATER": 0, "DONE": 0}}


# --- undo ------------------------------------------------------------------

def test_undo_restores_the_state_before_the_last_decision(tmp_path):
    output = _build(tmp_path, groups=4)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        response = _post(base, {"group_id": 2, "action": "mark"})
        assert response["undo_depth"] == 2

        undone = _post(base, {"action": "undo"})
        assert undone["undo"]["group_id"] == 2
        # Group 2 had no decision before the mark, so it goes back to pending.
        assert undone["undo"]["restored"] is None
        assert undone["undo"]["queue"] == "PENDING"
        assert undone["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 1}

        undone = _post(base, {"action": "undo"})
        assert undone["undo"]["group_id"] == 1
        assert undone["queues"] == {"PENDING": 4, "LATER": 0, "DONE": 0}
        assert undone["undo_depth"] == 0

        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"action": "undo"})
        assert excinfo.value.code == 400


def test_undo_restores_an_earlier_decision_not_just_the_pending_state(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        member = _queue(base, "PENDING")["items"][0]["members"][1]["file_id"]
        _post(base, {"group_id": 1, "file_id": member, "action": "pick"})
        _post(base, {"group_id": 1, "action": "mark"})
        assert _queue(base, "LATER")["total"] == 1

        undone = _post(base, {"action": "undo"})
        restored = undone["undo"]["restored"]
        assert restored["action"] == "pick" and restored["file_id"] == member
        assert undone["undo"]["queue"] == "DONE"
        assert undone["queues"] == {"PENDING": 2, "LATER": 0, "DONE": 1}
        state = _queue(base, "DONE")["review_state"]["1"]
        assert state["action"] == "pick" and state["file_id"] == member


def test_undo_reports_which_group_and_photo_to_return_to(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        group = _queue(base, "PENDING")["items"][2]
        gid = group["group_id"]
        chosen = group["members"][1]["file_id"]
        _post(base, {"group_id": gid, "file_id": chosen, "action": "pick",
                     "context_file_id": chosen})
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["group_id"] == gid
        assert undone["focus_file_id"] == chosen   # the photo that was acted on
        assert undone["undone"]["action"] == "pick"


# --- the photo undo comes back to ------------------------------------------

@pytest.mark.parametrize("action", ["accept", "mark"])
def test_undo_returns_to_the_photo_that_was_on_screen(tmp_path, action):
    """accept/mark name no photo, so the on-screen one has to be recorded.

    Without it, undo lands on whichever member sorts first -- which happens to be
    the AI keeper -- instead of the photo the reviewer was actually looking at.
    """
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        group = _queue(base, "PENDING")["items"][1]
        gid = group["group_id"]
        first, watched = (member["file_id"] for member in group["members"])
        assert watched != first

        _post(base, {"group_id": gid, "action": action, "context_file_id": watched})
        # The decision itself still names no photo; the focus sits beside the step.
        stored = json.loads((output / "review_state.json").read_text(encoding="utf-8"))
        assert "file_id" not in stored["groups"][str(gid)]
        assert stored["history"][-1]["focus_file_id"] == watched

    # A fresh server process reads that history back off disk.
    with _Served(output) as base:
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["group_id"] == gid
        assert undone["focus_file_id"] == watched


def test_a_decision_without_a_context_still_undoes(tmp_path):
    """An older client sends no context: undo works, it just has no photo."""
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["group_id"] == 1
        assert undone["focus_file_id"] is None


def test_undoing_a_pick_without_a_context_falls_back_to_the_chosen_photo(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        chosen = _queue(base, "PENDING")["items"][0]["members"][1]["file_id"]
        _post(base, {"group_id": 1, "file_id": chosen, "action": "pick"})
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["focus_file_id"] == chosen


def test_a_context_that_is_not_a_member_is_refused_and_changes_nothing(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        groups = _queue(base, "PENDING")["items"]
        gid = groups[0]["group_id"]
        foreign = groups[2]["members"][0]["file_id"]      # a real photo, wrong group
        assert foreign not in [m["file_id"] for m in groups[0]["members"]]
        for bad in (foreign, 999999, 0, -1, "3", True, 1.5, [3], {}):
            with pytest.raises(HTTPError) as excinfo:
                _post(base, {"group_id": gid, "action": "accept",
                             "context_file_id": bad})
            assert excinfo.value.code == 400
        # Every rejected call left the queues and the history untouched.
        status = _get(base, "/api/status")
        assert status["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 0}
        assert status["undo_depth"] == 0
        assert not (output / "review_state.json").exists()


def test_a_context_for_a_pick_is_validated_separately_from_the_keeper(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        groups = _queue(base, "PENDING")["items"]
        gid = groups[0]["group_id"]
        keeper = groups[0]["members"][0]["file_id"]
        foreign = groups[1]["members"][0]["file_id"]
        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"group_id": gid, "file_id": keeper, "action": "pick",
                         "context_file_id": foreign})
        assert excinfo.value.code == 400
        assert _get(base, "/api/status")["queues"]["DONE"] == 0
        # The same call with a member context is accepted.
        assert _post(base, {"group_id": gid, "file_id": keeper, "action": "pick",
                            "context_file_id": keeper})["ok"] is True


def test_a_context_is_never_written_into_the_decision(tmp_path):
    """It must not turn accept/mark into "the user picked this photo"."""
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        watched = _queue(base, "PENDING")["items"][0]["members"][1]["file_id"]
        _post(base, {"group_id": 1, "action": "mark", "context_file_id": watched})
        state = _queue(base, "LATER")["review_state"]["1"]
        assert state["action"] == "mark"
        assert "file_id" not in state


def test_undo_refuses_a_context_of_its_own(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"action": "undo", "context_file_id": 1})
        assert excinfo.value.code == 400
        assert _get(base, "/api/status")["undo_depth"] == 1


def test_review_state_rejects_a_malformed_focus_outright(tmp_path):
    output = _build(tmp_path, groups=2)
    state = review_state.ReviewState(output)
    for bad in (0, -1, True, "5", 1.5):
        with pytest.raises(ValueError, match="focus_file_id"):
            state.apply_decision(1, {"action": "accept", "timestamp": 1},
                                 focus_file_id=bad)
        with pytest.raises(ValueError, match="focus_file_id"):
            state.revert_decision(1, focus_file_id=bad)
    assert state.history_depth() == 0


def test_a_corrupt_focus_on_disk_is_dropped_without_losing_the_step(tmp_path):
    output = _build(tmp_path, groups=2)
    (output / "review_state.json").write_text(json.dumps({
        "version": 1,
        "groups": {"1": {"action": "accept", "timestamp": 5,
                         "member_fingerprint": "x"}},
        "history": [{"group_id": 1, "before": None,
                     "after": {"action": "accept", "timestamp": 5},
                     "focus_file_id": "not-an-id"}],
    }), encoding="utf-8")
    state = review_state.ReviewState(output)
    assert state.history_depth() == 1
    undone = state.undo_last()
    assert undone["group_id"] == 1 and undone["focus_file_id"] is None


def test_the_focus_survives_persistence_up_to_the_history_bound(tmp_path):
    output = _build(tmp_path, groups=2)
    state = review_state.ReviewState(output)
    for step in range(review_state.HISTORY_LIMIT + 10):
        state.apply_decision(1, {"action": "accept", "timestamp": step},
                             focus_file_id=1 + step % 2)
    assert state.history_depth() == review_state.HISTORY_LIMIT
    reloaded = review_state.ReviewState(output)
    assert reloaded.history_depth() == review_state.HISTORY_LIMIT
    # The newest step is still the one undo replays, focus included.
    last_focus = 1 + (review_state.HISTORY_LIMIT + 9) % 2
    assert reloaded.undo_last()["focus_file_id"] == last_focus

def test_undo_survives_a_restart(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        _post(base, {"group_id": 2, "action": "accept"})
    # Fresh server process, same output directory: history is on disk.
    with _Served(output) as base:
        assert _get(base, "/api/status")["undo_depth"] == 1
        undone = _post(base, {"action": "undo"})
        assert undone["undo"]["group_id"] == 2
        assert undone["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 0}


def test_undo_history_is_bounded(tmp_path):
    output = _build(tmp_path, groups=2)
    state = review_state.ReviewState(output)
    for step in range(review_state.HISTORY_LIMIT + 25):
        state.apply_decision(1 + step % 2, {"action": "accept", "timestamp": step})
    assert state.history_depth() == review_state.HISTORY_LIMIT
    reloaded = review_state.ReviewState(output)
    assert reloaded.history_depth() == review_state.HISTORY_LIMIT


def test_undo_rejects_a_payload_that_names_a_group(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        for bad in ({"action": "undo", "group_id": 1},
                    {"action": "undo", "file_id": 1}):
            with pytest.raises(HTTPError) as excinfo:
                _post(base, bad)
            assert excinfo.value.code == 400
        # The rejected calls changed nothing.
        assert _get(base, "/api/status")["undo_depth"] == 1


def test_undo_skips_history_for_groups_outside_the_current_scope(tmp_path):
    """A step recorded for a group this root does not own must not be replayed."""
    output = _build(tmp_path, groups=3)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    state.apply_decision(9999, {"action": "accept", "timestamp": 2})
    assert state.history_depth() == 2

    mine = {1, 2, 3}
    undone = state.undo_last(allowed=lambda gid: gid in mine)
    # The foreign step was discarded on the way and the real one applied.
    assert undone["group_id"] == 1
    assert state.history_depth() == 0
    assert state.get_group(1) is None
    # The foreign decision is left exactly as it was: undo did not touch it.
    assert state.get_group(9999)["action"] == "accept"
    assert state.undo_last(allowed=lambda gid: gid in mine) is None


def test_the_server_only_offers_undo_for_groups_it_can_review(tmp_path):
    output = _build(tmp_path, groups=3)
    data = review_server.ReviewData(output)
    try:
        assert data.group_in_scope(2) is True
        assert data.group_in_scope(9999) is False
    finally:
        data.close()


def test_a_regrouped_id_loses_both_its_decision_and_its_undo_step(tmp_path):
    output = _build(tmp_path, groups=3)
    state = review_state.ReviewState(output)
    # A fingerprint from a member set this group no longer has.
    stale = review_state.compute_member_fingerprint([777, 888])
    state.apply_decision(1, {"action": "accept", "timestamp": 1}, stale)
    assert state.history_depth() == 1
    with _Served(output) as base:
        status = _get(base, "/api/status")
        assert status["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 0}
        assert status["undo_depth"] == 0
        with pytest.raises(HTTPError):
            _post(base, {"action": "undo"})


def test_a_corrupt_history_entry_is_ignored_without_losing_the_decisions(tmp_path):
    output = _build(tmp_path, groups=2)
    state_path = output / "review_state.json"
    state_path.write_text(json.dumps({
        "version": 1,
        "groups": {"1": {"action": "accept", "timestamp": 5,
                         "member_fingerprint": "x"}},
        "history": ["not a dict", {"group_id": 0}, {"group_id": 2, "before": 7},
                    {"group_id": 1, "before": None,
                     "after": {"action": "accept", "timestamp": 5}}],
    }), encoding="utf-8")
    state = review_state.ReviewState(output)
    assert state.get_group(1)["action"] == "accept"
    assert state.history_depth() == 1


# --- a recycled group_id must not let undo resurrect an old decision --------

def _regroup(output: Path, group_id: int, members: list[int]) -> None:
    """Give ``group_id`` a different member set, the way a Stage 2 rerun can.

    The id survives; the photos behind it do not. Everything recorded about the
    old members -- decisions *and* undo steps -- must stop applying.
    """
    conn = db.open_db(output / "inventory.sqlite")
    conn.execute("DELETE FROM group_members WHERE group_id = ?", (group_id,))
    conn.executemany(
        "INSERT INTO group_members (group_id, file_id, is_keep, reason) "
        "VALUES (?, ?, ?, ?)",
        [(group_id, fid, index == 0, "keep" if index == 0 else "dup")
         for index, fid in enumerate(members)],
    )
    conn.execute("UPDATE groups SET keep_file_id = ?, member_count = ? WHERE id = ?",
                 (members[0], len(members), group_id))
    conn.commit()
    conn.close()


@pytest.mark.parametrize("action", ["pick", "accept"])
def test_a_cleared_decision_leaves_no_undo_step_a_regroup_could_revive(tmp_path,
                                                                      action):
    """decision -> clear -> regroup: the step is gone, so undo cannot restore it.

    ``clear`` removes the current decision, so validating only current decisions
    would never look at this group again -- while the history still holds the old
    decision as the ``before`` that undoing the ``clear`` would put back.
    """
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        group = _queue(base, "PENDING")["items"][0]
        gid = group["group_id"]
        members = [member["file_id"] for member in group["members"]]
        payload = {"group_id": gid, "action": action, "context_file_id": members[1]}
        if action == "pick":
            payload["file_id"] = members[1]
        _post(base, payload)
        _post(base, {"group_id": gid, "action": "clear"})
        assert _get(base, "/api/status")["undo_depth"] == 2

    # On disk the cleared group holds no decision, but the history still carries
    # the old one -- that is exactly what must not survive a regroup.
    stored = json.loads((output / "review_state.json").read_text(encoding="utf-8"))
    assert str(gid) not in stored["groups"]
    assert stored["history"][0]["after"]["action"] == action
    assert stored["history"][-1]["before"]["action"] == action

    # Same group_id, different photos.
    other = _queue_members_of(output, gid=2)
    _regroup(output, gid, other)

    with _Served(output) as base:
        status = _get(base, "/api/status")
        assert status["undo_depth"] == 0
        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"action": "undo"})
        assert excinfo.value.code == 400
        # The regrouped id is undecided, and no decision was resurrected onto it.
        assert status["queues"]["DONE"] == 0
        assert status["review_state"]["reviewed"] == 0
        assert _queue(base, "DONE")["total"] == 0
        assert _queue(base, "LATER")["total"] == 0


def test_a_legacy_history_step_without_a_fingerprint_is_dropped(tmp_path):
    """It cannot be proven to describe these members, so undo must not use it."""
    output = _build(tmp_path, groups=2)
    (output / "review_state.json").write_text(json.dumps({
        "version": 1,
        "groups": {},
        "history": [{"group_id": 1, "before": {"action": "accept", "timestamp": 4},
                     "after": None, "timestamp": 4, "focus_file_id": 1}],
    }), encoding="utf-8")
    with _Served(output) as base:
        assert _get(base, "/api/status")["undo_depth"] == 0
        with pytest.raises(HTTPError):
            _post(base, {"action": "undo"})


def test_history_for_a_group_that_left_the_scope_is_dropped(tmp_path):
    output = _build(tmp_path, groups=2)
    state = review_state.ReviewState(output)
    state.apply_decision(9999, {"action": "accept", "timestamp": 1},
                         review_state.compute_member_fingerprint([1, 2]))
    with _Served(output) as base:
        assert _get(base, "/api/status")["undo_depth"] == 0
        with pytest.raises(HTTPError):
            _post(base, {"action": "undo"})


def test_valid_history_survives_the_startup_check(tmp_path):
    """The pruning must not be greedy: untouched groups keep their undo steps."""
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        groups = _queue(base, "PENDING")["items"]
        first, second = groups[0], groups[1]
        chosen = second["members"][1]["file_id"]
        _post(base, {"group_id": first["group_id"], "action": "mark",
                     "context_file_id": first["members"][0]["file_id"]})
        _post(base, {"group_id": second["group_id"], "file_id": chosen,
                     "action": "pick", "context_file_id": chosen})
        _post(base, {"group_id": second["group_id"], "action": "clear"})
        assert _get(base, "/api/status")["undo_depth"] == 3

    # Nothing was regrouped, so every step and the surviving decision stay.
    with _Served(output) as base:
        status = _get(base, "/api/status")
        assert status["undo_depth"] == 3
        assert status["queues"] == {"PENDING": 2, "LATER": 1, "DONE": 0}
        # Undoing the clear puts the pick back, on the same photo.
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["group_id"] == second["group_id"]
        assert undone["restored"]["action"] == "pick"
        assert undone["restored"]["file_id"] == chosen
        assert _queue(base, "DONE")["total"] == 1


def test_only_the_regrouped_id_loses_its_history(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        groups = _queue(base, "PENDING")["items"]
        stale_gid, safe_gid = groups[0]["group_id"], groups[1]["group_id"]
        _post(base, {"group_id": stale_gid, "action": "accept"})
        _post(base, {"group_id": safe_gid, "action": "mark",
                     "context_file_id": groups[1]["members"][0]["file_id"]})
        assert _get(base, "/api/status")["undo_depth"] == 2

    _regroup(output, stale_gid, _queue_members_of(output, gid=3))

    with _Served(output) as base:
        status = _get(base, "/api/status")
        assert status["undo_depth"] == 1
        # The regrouped group lost its decision; the untouched one kept it.
        assert status["queues"]["LATER"] == 1
        assert status["queues"]["DONE"] == 0
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["group_id"] == safe_gid


def test_pruning_history_backs_up_the_original_state_file(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        gid = _queue(base, "PENDING")["items"][0]["group_id"]
        _post(base, {"group_id": gid, "action": "accept"})
        _post(base, {"group_id": gid, "action": "clear"})
    original = (output / "review_state.json").read_text(encoding="utf-8")

    _regroup(output, gid, _queue_members_of(output, gid=2))
    with _Served(output) as base:
        assert _get(base, "/api/status")["undo_depth"] == 0
    # Pruning is in-memory; the user's original record is kept verbatim.
    backup = output / "review_state.pre-prune.json"
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == original

# --- advancing across a page boundary ---------------------------------------

def test_next_in_queue_returns_the_group_after_the_decided_one(tmp_path):
    """Deciding the first group of page 2 must not send the UI back to page 1.

    Removing one group can drop the queue's page count, and asking for the old
    page number then gets clamped -- landing the reviewer on a group they already
    finished. The server is asked what comes *next* instead.
    """
    output = _build(tmp_path, groups=51)
    with _Served(output) as base:
        # 51 groups at 50 per page: page 2 holds only group 51.
        assert _queue(base, "PENDING", size=50)["pages"] == 2
        page_two = _queue(base, "PENDING", page=2, size=50)
        assert [g["group_id"] for g in page_two["items"]] == [51]

        # Decide group 50 (last of page 1), then ask what follows it.
        _post(base, {"group_id": 50, "action": "accept"})
        following = _get(base, "/api/next?queue=PENDING&after=50&page_size=50")
        assert following["found"] is True
        assert following["group_id"] == 51        # forward, never back to 1
        assert following["wrapped"] is False
        # The queue is now 50 long, so group 51 sits on page 1 -- and the server
        # says so, rather than the client asking for a page that no longer exists.
        assert following["total"] == 50 and following["page"] == 1
        assert 51 in [g["group_id"] for g in
                      _queue(base, "PENDING", page=following["page"], size=50)["items"]]


def test_deciding_the_first_group_of_page_two_advances_to_the_next_one(tmp_path):
    output = _build(tmp_path, groups=120)
    with _Served(output) as base:
        _post(base, {"group_id": 51, "action": "mark"})
        following = _get(base, "/api/next?queue=PENDING&after=51&page_size=50")
        assert following["group_id"] == 52       # not 1, and not 51 again
        assert following["page"] == 2
        assert 52 in [g["group_id"] for g in
                      _queue(base, "PENDING", page=2, size=50)["items"]]


def test_the_end_of_the_queue_reports_the_last_survivor_without_wrapping(tmp_path):
    output = _build(tmp_path, groups=52)
    with _Served(output) as base:
        _post(base, {"group_id": 52, "action": "accept"})
        following = _get(base, "/api/next?queue=PENDING&after=52&page_size=50")
        assert following["wrapped"] is True
        # Nothing is ahead, so it stays on the last remaining group -- which is
        # on the previous page now, and the page number says so.
        assert following["group_id"] == 51
        assert following["total"] == 51 and following["page"] == 2

        # Emptying the queue answers "nothing to go to" rather than a stale id.
        for gid in range(1, 52):
            _post(base, {"group_id": gid, "action": "accept"})
        empty = _get(base, "/api/next?queue=PENDING&after=1&page_size=50")
        assert empty["found"] is False and empty["group_id"] is None
        assert empty["total"] == 0 and empty["page"] == 1


def test_next_in_queue_skips_groups_that_left_the_queue(tmp_path):
    output = _build(tmp_path, groups=8)
    with _Served(output) as base:
        for gid in (4, 5, 6):
            _post(base, {"group_id": gid, "action": "mark"})
        following = _get(base, "/api/next?queue=PENDING&after=3")
        assert following["group_id"] == 7        # 4/5/6 are in LATER now
        # And within LATER, the same walk works on that queue's own membership.
        assert _get(base, "/api/next?queue=LATER&after=4")["group_id"] == 5


def test_next_in_queue_validates_its_arguments(tmp_path):
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        for path in ("/api/next?queue=EVIL&after=1",
                     "/api/next?queue=PENDING&after=1&page_size=7",
                     "/api/next?queue=PENDING&after=not-an-int"):
            with pytest.raises(HTTPError) as excinfo:
                _get(base, path)
            assert excinfo.value.code == 400


# --- clearing an undecided group is not an undoable step --------------------

def test_clearing_an_undecided_group_records_no_undo_step(tmp_path):
    """A no-op must not become a step U has to walk back through."""
    output = _build(tmp_path, groups=3)
    with _Served(output) as base:
        assert _post(base, {"group_id": 1, "action": "clear"})["ok"] is True
        assert _post(base, {"group_id": 2, "action": "clear"})["ok"] is True
        status = _get(base, "/api/status")
        assert status["undo_depth"] == 0
        with pytest.raises(HTTPError):
            _post(base, {"action": "undo"})
        # A real decision then clears to exactly two steps, not four.
        _post(base, {"group_id": 1, "action": "accept"})
        _post(base, {"group_id": 1, "action": "clear"})
        assert _get(base, "/api/status")["undo_depth"] == 2

    # And the depth is the same after a restart: no phantom entries were written.
    with _Served(output) as base:
        assert _get(base, "/api/status")["undo_depth"] == 2
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["restored"]["action"] == "accept"


def test_a_repeated_clear_does_not_grow_the_history(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "mark"})
        for _ in range(5):
            _post(base, {"group_id": 1, "action": "clear"})
        # One mark, one effective clear.
        assert _get(base, "/api/status")["undo_depth"] == 2


def test_revert_decision_reports_whether_anything_changed(tmp_path):
    output = _build(tmp_path, groups=2)
    state = review_state.ReviewState(output)
    assert state.revert_decision(1) is False
    assert state.history_depth() == 0
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    assert state.revert_decision(1) is True
    assert state.history_depth() == 2


# --- accept needs an AI keeper ----------------------------------------------

def test_accept_is_refused_when_this_scope_has_no_ai_keeper(tmp_path):
    """A group straddling two roots can have its keeper outside this one."""
    output = _build(tmp_path, groups=2)
    photos = tmp_path / "photos"
    other = tmp_path / "other-library"
    other.mkdir()
    conn = db.open_db(output / "inventory.sqlite")
    # The keeper lives in another root; only the candidate is in scope.
    outside = other / "OTHER.jpg"
    Image.new("RGB", (4, 3), (9, 9, 9)).save(outside)
    stat = outside.stat()
    foreign_id = db.insert_file(conn, {
        "path": str(outside), "basename": outside.name, "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "exif_datetime": None, "exif_timestamp": 1,
        "width": 4, "height": 3, "file_kind": "jpg", "scan_status": "done"})
    inside = photos / "IMG_0000.jpg"
    inside_id = int(conn.execute("SELECT id FROM files WHERE path = ?",
                                 (str(inside),)).fetchone()["id"])
    straddling = db.insert_group(
        conn, "burst", foreign_id,
        [(foreign_id, True, "keep"), (inside_id, False, "dup")], 1)
    conn.commit()
    conn.close()

    with _Served(output) as base:
        members = _get(base, f"/api/group/{straddling}")["members"]
        assert [m["file_id"] for m in members] == [inside_id]
        assert not any(m["is_keep"] for m in members)

        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"group_id": straddling, "action": "accept",
                         "context_file_id": inside_id})
        assert excinfo.value.code == 400
        # Nothing was recorded, and no undo step was created for it.
        status = _get(base, "/api/status")
        assert status["queues"]["DONE"] == 0
        assert status["undo_depth"] == 0

        # Choosing explicitly still works: there is a photo, just no machine pick.
        assert _post(base, {"group_id": straddling, "file_id": inside_id,
                            "action": "pick", "context_file_id": inside_id}
                     )["ok"] is True
        # As does deferring it.
        _post(base, {"group_id": straddling, "action": "clear"})
        assert _post(base, {"group_id": straddling, "action": "mark"})["ok"] is True


def test_accept_still_works_for_an_ordinary_group(tmp_path):
    output = _build(tmp_path, groups=2)
    with _Served(output) as base:
        assert _post(base, {"group_id": 1, "action": "accept"})["ok"] is True
        assert _get(base, "/api/status")["queues"]["DONE"] == 1

# --- the queues respect the root scope -------------------------------------

def test_queues_only_count_groups_inside_the_bound_root(tmp_path):
    output = _build(tmp_path, groups=3)
    # Add a second library's group to the same database, as a legacy DB would have.
    conn = db.open_db(output / "inventory.sqlite")
    other = tmp_path / "other-library"
    other.mkdir()
    outside = []
    for index in range(2):
        source = other / f"OTHER_{index}.jpg"
        Image.new("RGB", (4, 3), (9, 9, 9)).save(source, "JPEG")
        stat = source.stat()
        outside.append(db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "exif_datetime": None,
            "exif_timestamp": 1, "width": 4, "height": 3,
            "file_kind": "jpg", "scan_status": "done"}))
    foreign_group = db.insert_group(
        conn, "burst", outside[0],
        [(outside[0], True, "keep"), (outside[1], False, "dup")], 1)
    conn.commit()
    conn.close()

    with _Served(output) as base:
        pending = _queue(base, "PENDING")
        assert pending["total"] == 3
        assert foreign_group not in [g["group_id"] for g in pending["items"]]
        assert pending["queue_counts"] == {"PENDING": 3, "LATER": 0, "DONE": 0}
        # Acting on the other root's group is refused outright.
        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"group_id": foreign_group, "action": "accept"})
        assert excinfo.value.code == 400
        located = _get(base, f"/api/locate?queue=PENDING&group_id={foreign_group}")
        assert located["found"] is False
        assert located["group_id"] in (1, 2, 3)


# --- safety ----------------------------------------------------------------

def test_no_queue_or_undo_operation_touches_an_original(tmp_path):
    output = _build(tmp_path, groups=4)
    photos = tmp_path / "photos"
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in photos.iterdir()}
    opened: list[str] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        if str(self).startswith(str(photos)):
            opened.append(str(self))
        return real_open(self, *args, **kwargs)

    Path.open = spy
    try:
        with _Served(output) as base:
            _queue(base, "PENDING")
            _post(base, {"group_id": 1, "action": "accept"})
            _post(base, {"group_id": 2, "action": "mark"})
            _post(base, {"action": "undo"})
            _queue(base, "LATER")
            _queue(base, "DONE")
            _get(base, "/api/locate?queue=PENDING&group_id=2")
    finally:
        Path.open = real_open

    assert opened == []
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in photos.iterdir()}
    assert before == after


def test_the_human_layer_stays_separate_from_the_ai_decisions(tmp_path):
    output = _build(tmp_path, groups=3)
    conn = db.open_db(output / "inventory.sqlite")
    before = [tuple(row) for row in conn.execute(
        "SELECT group_id, file_id, decision, is_keep FROM group_members "
        "ORDER BY group_id, file_id")]
    conn.close()
    manifest = output / "delete_local.txt"
    manifest.write_text("original manifest\n", encoding="utf-8")

    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        _post(base, {"group_id": 2, "file_id": 4, "action": "pick"})
        _post(base, {"action": "undo"})

    conn = db.open_db(output / "inventory.sqlite")
    after = [tuple(row) for row in conn.execute(
        "SELECT group_id, file_id, decision, is_keep FROM group_members "
        "ORDER BY group_id, file_id")]
    conn.close()
    assert before == after
    assert manifest.read_text(encoding="utf-8") == "original manifest\n"
    # Human state lives in its own file, and names no path.
    state_text = (output / "review_state.json").read_text(encoding="utf-8")
    assert str(tmp_path / "photos") not in state_text
    assert "IMG_0000.jpg" not in state_text
