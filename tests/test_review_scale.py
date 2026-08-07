"""A loose ceiling on the review server's per-interaction cost.

The queues live in a JSON file, not in SQLite, so every page and every decision
classifies the whole group list in memory. That is fine at the size this tool
targets, but it is exactly the kind of thing that silently becomes quadratic, so
this pins it with a synthetic library an order of magnitude past normal use.

The budgets are deliberately generous -- this is a regression fence, not a
benchmark, and it has to hold on a slow shared CI box. What it actually catches is
a change that reintroduces a full deep copy per lookup, a second scan per
response, or a per-group SQL query.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src import db, review_server, review_state, root_scope, thumbnails

GROUPS = 30_000
MEMBERS_PER_GROUP = 2

# Wall-clock ceilings for one interaction on the synthetic library above.
PAGE_BUDGET_S = 1.5
ACTION_BUDGET_S = 1.5
STATUS_BUDGET_S = 1.5
STARTUP_BUDGET_S = 30.0


@pytest.fixture(scope="module")
def big_output(tmp_path_factory) -> Path:
    """A 30k-group library, built once for the whole module.

    Rows only: no image is created or decoded, because nothing under test reads
    one. Thumbnail bookkeeping is inserted so the page join is realistic.
    """
    tmp_path = tmp_path_factory.mktemp("perf")
    output = tmp_path / "output"
    photos = tmp_path / "photos"
    output.mkdir(parents=True)
    photos.mkdir(parents=True)
    thumbnails.thumbs_dir(output).mkdir(parents=True, exist_ok=True)
    conn = db.open_db(output / "inventory.sqlite")
    root_scope.bind(conn, photos)

    total = GROUPS * MEMBERS_PER_GROUP
    conn.executemany(
        "INSERT INTO files (path, basename, size_bytes, mtime_ns, exif_datetime, "
        "exif_timestamp, width, height, file_kind, scan_status, path_key, root_key) "
        "VALUES (:path, :basename, :size_bytes, :mtime_ns, :exif_datetime, "
        ":exif_timestamp, :width, :height, :file_kind, :scan_status, :path_key, "
        ":root_key)",
        [{"path": str(photos / f"IMG_{i:06d}.jpg"), "basename": f"IMG_{i:06d}.jpg",
          "size_bytes": 1000 + i, "mtime_ns": 1, "exif_datetime": None,
          "exif_timestamp": 1000 + i, "width": 40, "height": 30,
          "file_kind": "jpg", "scan_status": "done",
          "path_key": root_scope.normalize(photos / f"IMG_{i:06d}.jpg"),
          "root_key": root_scope.normalize(photos)}
         for i in range(total)],
    )
    conn.executemany(
        "INSERT INTO groups (id, group_type, keep_file_id, member_count, created_at) "
        "VALUES (:id, :group_type, :keep_file_id, :member_count, :created_at)",
        [{"id": g + 1, "group_type": "burst", "keep_file_id": g * 2 + 1,
          "member_count": MEMBERS_PER_GROUP, "created_at": 1} for g in range(GROUPS)],
    )
    conn.executemany(
        "INSERT INTO group_members (group_id, file_id, is_keep, decision, reason) "
        "VALUES (:group_id, :file_id, :is_keep, :decision, :reason)",
        [{"group_id": g + 1, "file_id": g * 2 + 1 + m, "is_keep": m == 0,
          "decision": "KEEP" if m == 0 else "MAYBE",
          "reason": "GROUP_KEEPER" if m == 0 else "LOW_MARGIN"}
         for g in range(GROUPS) for m in range(MEMBERS_PER_GROUP)],
    )
    conn.executemany(
        "INSERT INTO thumbnails (file_id, status, max_px, bytes, created_at) "
        "VALUES (:file_id, 'ok', 320, 900, 1)",
        [{"file_id": i + 1} for i in range(total)],
    )
    db.build_all_view_index(conn, scope=None)
    conn.commit()
    conn.close()
    (output / "review.html").write_text("<h1>Review</h1>", encoding="utf-8")
    (output / "review_summary.json").write_text(json.dumps({"views": {}}),
                                                encoding="utf-8")
    return output


def _elapsed(call) -> float:
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def test_opening_a_large_library_is_not_slow(big_output):
    duration = _elapsed(lambda: review_server.ReviewData(big_output).close())
    assert duration < STARTUP_BUDGET_S, f"startup took {duration:.2f}s"


@pytest.fixture(scope="module")
def big_data(big_output):
    data = review_server.ReviewData(big_output)
    yield data
    data.close()


def test_a_queue_page_stays_fast_with_thirty_thousand_groups(big_data):
    # First and last page: the last one is where an accidental full scan per row
    # would show up.
    first = _elapsed(lambda: big_data.page("GROUPS", 1, 100, "PENDING"))
    last_page = big_data.page("GROUPS", 1, 100, "PENDING")["pages"]
    last = _elapsed(lambda: big_data.page("GROUPS", last_page, 100, "PENDING"))
    assert first < PAGE_BUDGET_S, f"first page took {first:.2f}s"
    assert last < PAGE_BUDGET_S, f"last page took {last:.2f}s"


def test_a_decision_stays_fast_with_thirty_thousand_groups(big_data, monkeypatch):
    # Decisions are recorded against a throwaway state so the module fixture is
    # not mutated for the other tests.
    durations = []
    for group_id in range(1, 21):
        durations.append(_elapsed(
            lambda gid=group_id: big_data.apply_action(
                {"group_id": gid, "action": "mark"})))
    worst = max(durations)
    assert worst < ACTION_BUDGET_S, f"slowest action took {worst:.2f}s"
    # And it does not get worse as the state file grows.
    assert durations[-1] < durations[0] * 5 + ACTION_BUDGET_S
    for group_id in range(1, 21):
        big_data.apply_action({"group_id": group_id, "action": "clear"})


def test_status_and_locate_stay_fast(big_data):
    status = _elapsed(big_data.status)
    locate = _elapsed(lambda: big_data.locate(29_000, "PENDING"))
    following = _elapsed(lambda: big_data.next_in_queue(29_000, "PENDING"))
    assert status < STATUS_BUDGET_S, f"status took {status:.2f}s"
    assert locate < STATUS_BUDGET_S, f"locate took {locate:.2f}s"
    assert following < STATUS_BUDGET_S, f"next took {following:.2f}s"


def test_the_group_id_list_is_read_once_not_per_request(big_data):
    """A per-request scan of every group would be the first thing to regress."""
    calls = {"n": 0}
    original = db.group_ids_in_scope

    def spy(conn, scope=None):
        calls["n"] += 1
        return original(conn, scope=scope)

    db.group_ids_in_scope = spy
    try:
        big_data.group_ids()
        for _ in range(5):
            big_data.page("GROUPS", 1, 100, "PENDING")
            big_data.status()
    finally:
        db.group_ids_in_scope = original
    assert calls["n"] == 0, "the cached scoped id list was rebuilt per request"


def test_queue_classification_does_not_copy_every_decision(big_data):
    """queue_tally must read the actions, not deep-copy the whole state."""
    state = big_data.state
    copies = {"n": 0}
    original = review_state.ReviewState.snapshot

    def spy(self):
        copies["n"] += 1
        return original(self)

    review_state.ReviewState.snapshot = spy
    try:
        state.queue_tally(big_data.group_ids())
        big_data.page("GROUPS", 1, 100, "PENDING")
        big_data.status()
        big_data.apply_action({"group_id": 1, "action": "mark"})
        big_data.apply_action({"group_id": 1, "action": "clear"})
    finally:
        review_state.ReviewState.snapshot = original
    assert copies["n"] == 0, "a full state snapshot is taken on the hot path"


def test_the_state_file_stays_compact_as_decisions_accumulate(big_data):
    """Compact separators keep the file the reviewer's disk actually carries small."""
    for group_id in range(1, 501):
        big_data.apply_action({"group_id": group_id, "action": "accept"})
    try:
        path = big_data.state.path
        size = path.stat().st_size
        # ~500 decisions plus their undo steps; indentation used to roughly
        # triple this. The ceiling is loose, it only has to catch a regression to
        # a pretty-printed file.
        assert size < 400_000, f"state file is {size} bytes for 500 decisions"
        assert "\n" not in path.read_text(encoding="utf-8").strip()
    finally:
        for group_id in range(1, 501):
            big_data.apply_action({"group_id": group_id, "action": "clear"})
