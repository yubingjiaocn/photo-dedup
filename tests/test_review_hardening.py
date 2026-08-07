"""Security and durability gates for the review server's write surface.

Three separate concerns, all about what a *local* HTTP server owes its user:

* **Cross-site writes.** The server binds loopback, but every page in the same
  browser can still reach it. ``/api/action`` therefore requires a JSON
  content type and rejects a cross-site ``Origin``/``Referer``, while a header-less
  local script (curl, a test) keeps working.
* **Static exposure.** The output directory is a working directory, not a web
  root: it holds the deletion manifests, the human review state, the summary and
  the run log. Only the review page is served statically.
* **State durability.** A ``review_state.json`` this build cannot read is kept,
  not overwritten, and the reviewer is told; a successful write is fsynced before
  the rename so a power loss cannot leave a truncated file behind.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from src import db, review_server, review_state, root_scope, thumbnails


def _build(tmp_path: Path, groups: int = 3) -> Path:
    """Output dir bound to one photo root, with ``groups`` groups of two."""
    output = tmp_path / "output"
    photos = tmp_path / "photos"
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
        file_ids.append(db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "exif_datetime": None,
            "exif_timestamp": 1000 + index, "width": 40, "height": 30,
            "file_kind": "jpg", "scan_status": "done"}))
    for number in range(groups):
        keeper, dup = file_ids[number * 2], file_ids[number * 2 + 1]
        db.insert_group(conn, "burst", keeper,
                        [(keeper, True, "keep"), (dup, False, "dup")], 1)
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
        self.base = url.rsplit("/", 1)[0]
        return self.base

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.review_data.close()
        self.thread.join(timeout=5)


def _get(base: str, path: str) -> dict:
    return json.loads(urlopen(base + path).read())


def _post_raw(base: str, payload: dict, headers: dict) -> dict:
    request = Request(f"{base}/api/action", data=json.dumps(payload).encode("utf-8"),
                      method="POST", headers=headers)
    return json.loads(urlopen(request).read())


def _post(base: str, payload: dict) -> dict:
    return _post_raw(base, payload, {"Content-Type": "application/json"})


# --- cross-site writes ------------------------------------------------------

def test_a_write_requires_a_json_content_type(tmp_path):
    """A cross-origin form can only send the safelisted types, so JSON is required."""
    output = _build(tmp_path)
    with _Served(output) as base:
        for media_type in ("text/plain", "text/plain;charset=UTF-8",
                           "application/x-www-form-urlencoded",
                           "multipart/form-data", ""):
            with pytest.raises(HTTPError) as excinfo:
                _post_raw(base, {"group_id": 1, "action": "accept"},
                          {"Content-Type": media_type} if media_type else {})
            assert excinfo.value.code == 403, media_type
        # Nothing was recorded by any of them.
        assert _get(base, "/api/status")["queues"]["DONE"] == 0
        # The charset parameter is fine on the correct type.
        assert _post_raw(base, {"group_id": 1, "action": "accept"},
                         {"Content-Type": "application/json; charset=utf-8"}
                         )["ok"] is True


def test_a_write_from_another_origin_is_refused(tmp_path):
    output = _build(tmp_path)
    with _Served(output) as base:
        port = base.rsplit(":", 1)[1]
        for header in ("Origin", "Referer"):
            for value in ("http://evil.example",
                          "https://evil.example/attack.html",
                          f"http://evil.example:{port}",
                          # A host that merely *contains* loopback is not loopback.
                          "http://127.0.0.1.evil.example",
                          "http://localhost.evil.example/x",
                          # Right host, wrong port: a different local server.
                          "http://127.0.0.1:1/attack.html",
                          "file:///tmp/attack.html"):
                with pytest.raises(HTTPError) as excinfo:
                    _post_raw(base, {"group_id": 1, "action": "accept"},
                              {"Content-Type": "application/json", header: value})
                assert excinfo.value.code == 403, f"{header}: {value}"
        assert _get(base, "/api/status")["queues"]["DONE"] == 0


def test_a_write_from_the_review_page_itself_is_allowed(tmp_path):
    output = _build(tmp_path)
    with _Served(output) as base:
        for header in ("Origin", "Referer"):
            value = base if header == "Origin" else f"{base}/review.html"
            assert _post_raw(base, {"group_id": 1, "action": "clear"},
                             {"Content-Type": "application/json", header: value}
                             )["ok"] is True


def test_a_local_script_without_an_origin_header_still_works(tmp_path):
    """curl and the pipeline's own tooling send no Origin; they are not the threat."""
    output = _build(tmp_path)
    with _Served(output) as base:
        assert _post(base, {"group_id": 1, "action": "mark"})["ok"] is True
        # An explicit opaque origin (a sandboxed iframe) is also not a site.
        assert _post_raw(base, {"group_id": 1, "action": "clear"},
                         {"Content-Type": "application/json", "Origin": "null"}
                         )["ok"] is True


def test_reads_are_not_gated_by_the_write_checks(tmp_path):
    output = _build(tmp_path)
    with _Served(output) as base:
        request = Request(f"{base}/api/status", headers={"Origin": "http://evil.example"})
        assert json.loads(urlopen(request).read())["queues"]["PENDING"] == 3


# --- static exposure --------------------------------------------------------

def _status_of(base: str, path: str) -> int:
    try:
        return urlopen(base + path).status
    except HTTPError as exc:
        return exc.code


def test_only_the_review_page_is_served_statically(tmp_path):
    """Everything else in the output directory stays unreachable, by default."""
    output = _build(tmp_path)
    # Files a real run writes beside the review page.
    (output / "delete_local.txt").write_text("C:/Photos/IMG_0001.jpg\n", encoding="utf-8")
    (output / "delete_cloud.json").write_text('{"items": []}', encoding="utf-8")
    (output / "summary.txt").write_text("secret totals\n", encoding="utf-8")
    (output / "performance.txt").write_text("stage1 12.5s\n", encoding="utf-8")
    (output / "photo-dedup.log").write_text("C:/Photos scanned\n", encoding="utf-8")
    (output / "review_state.json").write_text('{"version": 1, "groups": {}}',
                                              encoding="utf-8")
    (output / "review_state.pre-prune.json").write_text("{}", encoding="utf-8")
    (output / "execute_plan.txt").write_text("plan\n", encoding="utf-8")
    (output / "review_summary.json").write_text('{"views": {}}', encoding="utf-8")
    (output / "brand-new-artifact.txt").write_text("added later\n", encoding="utf-8")

    with _Served(output) as base:
        assert _status_of(base, "/review.html") == 200
        for name in ("delete_local.txt", "delete_cloud.json", "summary.txt",
                     "performance.txt", "photo-dedup.log", "review_state.json",
                     "review_state.pre-prune.json", "execute_plan.txt",
                     "review_summary.json", "brand-new-artifact.txt",
                     "inventory.sqlite"):
            assert _status_of(base, "/" + name) == 404, name


def test_the_thumbnail_cache_and_db_stay_api_only(tmp_path):
    output = _build(tmp_path)
    thumbs = thumbnails.thumbs_dir(output)
    thumbnails.thumb_path(thumbs, 1).write_bytes(b"\xff\xd8\xffnot-a-real-jpeg")
    with _Served(output) as base:
        for path in ("/thumbs/1.jpg", "/thumbs/", "/inventory.sqlite",
                     "/inventory.sqlite-wal", "/inventory.sqlite-shm"):
            assert _status_of(base, path) == 404, path


@pytest.mark.parametrize("path", [
    "/%2e%2e/%2e%2e/etc/passwd",
    "/..%2f..%2fetc/passwd",
    "/%2E%2E/review.html",
    "/subdir/../summary.txt",
    "/./summary.txt",
    "/%73ummary.txt",                 # percent-encoded first letter
    "/summary%2Etxt",
    "/SUMMARY.TXT/../summary.txt",
    "/\\summary.txt",
    "/%5csummary.txt",
    "/review.html/../summary.txt",
    "/nested/review.html",
    "/%00review.html",
    "/review.html%00.txt",
])
def test_encoded_and_traversal_paths_cannot_reach_another_file(tmp_path, path):
    output = _build(tmp_path)
    (output / "summary.txt").write_text("secret totals\n", encoding="utf-8")
    with _Served(output) as base:
        assert _status_of(base, path) in (400, 403, 404), path


def test_the_root_url_never_lists_the_output_directory(tmp_path):
    """A generated index would advertise the manifests, the state and the log."""
    output = _build(tmp_path)
    (output / "summary.txt").write_text("secret totals\n", encoding="utf-8")
    (output / "delete_local.txt").write_text("C:/Photos/IMG.jpg\n", encoding="utf-8")
    with _Served(output) as base:
        body = urlopen(base + "/").read().decode("utf-8", "replace")
        # It lands on the review page instead of an index.
        assert "<h1>Review</h1>" in body
        for name in ("summary.txt", "delete_local.txt", "inventory.sqlite",
                     "review_state.json", "thumbs/"):
            assert name not in body, name


# --- state durability -------------------------------------------------------

def test_an_unreadable_state_file_is_kept_and_reported(tmp_path):
    output = _build(tmp_path)
    state_path = output / "review_state.json"
    state_path.write_text('{"version": 1, "groups": {"1": {"action": "acce',
                          encoding="utf-8")   # truncated write
    original = state_path.read_text(encoding="utf-8")

    with _Served(output) as base:
        status = _get(base, "/api/status")
        assert "state_warning" in status
        assert "review_state.json" in status["state_warning"]
        # Review still works, starting empty.
        assert status["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 0}
        assert state_path.read_text(encoding="utf-8") == original   # untouched so far

        assert _post(base, {"group_id": 1, "action": "accept"})["ok"] is True

    kept = sorted(output.glob("review_state.corrupt-*.json"))
    assert len(kept) == 1
    assert kept[0].read_text(encoding="utf-8") == original
    # The new file is valid and holds the decision that was just made.
    fresh = json.loads(state_path.read_text(encoding="utf-8"))
    assert fresh["version"] == 1 and "1" in fresh["groups"]


def test_a_future_schema_version_is_not_overwritten(tmp_path):
    """A newer build's state must not be destroyed by an older one."""
    output = _build(tmp_path)
    state_path = output / "review_state.json"
    future = json.dumps({"version": 99, "groups": {"1": {"action": "accept"}},
                         "unknown_section": [1, 2, 3]})
    state_path.write_text(future, encoding="utf-8")

    with _Served(output) as base:
        warning = _get(base, "/api/status")["state_warning"]
        assert "99" in warning
        assert _get(base, "/api/status")["queues"]["DONE"] == 0   # not trusted
        _post(base, {"group_id": 1, "action": "mark"})

    kept = sorted(output.glob("review_state.corrupt-*.json"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == future


@pytest.mark.parametrize("body", [
    "",
    "not json at all",
    "[]",
    '"a string"',
    '{"version": 1, "groups": []}',
    '{"version": 1',
])
def test_every_unreadable_shape_is_preserved_before_the_first_write(tmp_path, body):
    output = _build(tmp_path)
    state_path = output / "review_state.json"
    state_path.write_text(body, encoding="utf-8")
    state = review_state.ReviewState(output)
    assert state.load_error is not None
    assert state.warning()
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    kept = sorted(output.glob("review_state.corrupt-*.json"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == body
    # And the state is usable from here on.
    assert review_state.ReviewState(output).get_group(1)["action"] == "accept"


def test_a_readable_state_file_is_never_quarantined(tmp_path):
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    reloaded = review_state.ReviewState(output)
    assert reloaded.load_error is None and reloaded.warning() is None
    reloaded.apply_decision(2, {"action": "mark", "timestamp": 2})
    assert not list(output.glob("review_state.corrupt-*.json"))
    # Both decisions are on disk, each with its undo step.
    final = review_state.ReviewState(output)
    assert final.history_depth() == 2
    assert final.get_group(1)["action"] == "accept"
    assert final.get_group(2)["action"] == "mark"


def test_quarantine_happens_once_not_on_every_write(tmp_path):
    output = _build(tmp_path)
    (output / "review_state.json").write_text("broken", encoding="utf-8")
    state = review_state.ReviewState(output)
    for gid in (1, 2, 3):
        state.apply_decision(gid, {"action": "mark", "timestamp": gid})
    assert len(list(output.glob("review_state.corrupt-*.json"))) == 1


def test_the_state_write_is_fsynced_before_the_rename(tmp_path, monkeypatch):
    """Without the flush+fsync, the rename can outlive the data it points at."""
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    events: list[str] = []
    real_fsync, real_replace = os.fsync, Path.replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def spy_replace(self, target):
        events.append(f"replace:{Path(target).name}")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(Path, "replace", spy_replace)
    state.apply_decision(1, {"action": "accept", "timestamp": 1})

    assert "fsync" in events
    rename = next(index for index, name in enumerate(events)
                  if name.startswith("replace:review_state.json"))
    assert events.index("fsync") < rename, events


def test_a_directory_fsync_failure_does_not_fail_the_write(tmp_path, monkeypatch):
    """Not every platform supports it; the decision must still be recorded."""
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    real_fsync = os.fsync
    seen = {"calls": 0}

    def flaky(fd):
        seen["calls"] += 1
        if seen["calls"] > 1:            # the directory handle
            raise OSError("fsync not supported here")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", flaky)
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    monkeypatch.undo()
    assert review_state.ReviewState(output).get_group(1)["action"] == "accept"


def test_the_state_file_is_written_compactly(tmp_path):
    """It is rewritten on every decision; indentation roughly triples it."""
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    for gid in (1, 2, 3):
        state.apply_decision(gid, {"action": "mark", "timestamp": gid},
                             "fingerprint")
    text = (output / "review_state.json").read_text(encoding="utf-8")
    assert "\n" not in text.strip()
    assert ", " not in text and '": ' not in text
    # Still valid, and still readable by the next process.
    assert json.loads(text)["version"] == 1
    assert review_state.ReviewState(output).summary()["marked"] == 3


def test_a_failed_write_leaves_no_temp_files_behind(tmp_path, monkeypatch):
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    good = (output / "review_state.json").read_text(encoding="utf-8")

    monkeypatch.setattr(Path, "replace",
                        lambda self, target: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        state.apply_decision(2, {"action": "mark", "timestamp": 2})
    monkeypatch.undo()

    assert not list(output.glob(".review_state_*.tmp"))
    # The previous good state survived the failure.
    assert (output / "review_state.json").read_text(encoding="utf-8") == good


# --- a quarantine that fails must not overwrite the file --------------------

def _fail_quarantine_only(monkeypatch, output: Path) -> None:
    """Make only the quarantine rename fail, leaving normal writes alone."""
    real_replace = Path.replace

    def picky(self, target):
        if "corrupt-" in Path(target).name:
            raise OSError("permission denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", picky)


def test_a_decision_is_refused_when_the_bad_file_cannot_be_set_aside(tmp_path,
                                                                    monkeypatch):
    """Overwriting it anyway would break the promise the warning just made."""
    output = _build(tmp_path)
    state_path = output / "review_state.json"
    state_path.write_text('{"version": 1, "groups": {"1": {"action": "acce',
                          encoding="utf-8")
    original = state_path.read_bytes()

    state = review_state.ReviewState(output)
    assert state.load_error is not None
    _fail_quarantine_only(monkeypatch, output)

    with pytest.raises(OSError, match="could not be set aside"):
        state.apply_decision(1, {"action": "accept", "timestamp": 1})

    # The unreadable original keeps its exact bytes, and nothing replaced it.
    assert state_path.read_bytes() == original
    assert not list(output.glob("review_state.corrupt-*.json"))
    assert not list(output.glob(".review_state_*.tmp"))
    # The refused decision is not in memory either.
    assert state.get_group(1) is None
    assert state.history_depth() == 0
    # And the warning is still on, so the UI keeps telling the user.
    assert state.warning()

    # Once the cause is fixed, the very next attempt quarantines and succeeds.
    monkeypatch.undo()
    state.apply_decision(1, {"action": "accept", "timestamp": 1})
    kept = sorted(output.glob("review_state.corrupt-*.json"))
    assert len(kept) == 1 and kept[0].read_bytes() == original
    assert json.loads(state_path.read_text(encoding="utf-8"))["groups"]["1"]["action"] \
        == "accept"


def test_the_api_refuses_the_write_and_keeps_the_file_when_quarantine_fails(
        tmp_path, monkeypatch):
    output = _build(tmp_path)
    state_path = output / "review_state.json"
    future = json.dumps({"version": 99, "groups": {"1": {"action": "accept"}}})
    state_path.write_text(future, encoding="utf-8")

    with _Served(output) as base:
        warning = _get(base, "/api/status")["state_warning"]
        assert "99" in warning
        _fail_quarantine_only(monkeypatch, output)

        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"group_id": 1, "action": "accept"})
        assert excinfo.value.code == 400

        # Server-side state is unchanged, and the warning persists.
        status = _get(base, "/api/status")
        assert status["queues"] == {"PENDING": 3, "LATER": 0, "DONE": 0}
        assert status["undo_depth"] == 0
        assert status["state_warning"] == warning

    # The newer-version file the reviewer must not lose is byte-identical.
    assert state_path.read_text(encoding="utf-8") == future
    assert not list(output.glob("review_state.corrupt-*.json"))


# --- a failed write must not leave the decision in memory -------------------

def _fail_the_state_write(monkeypatch) -> None:
    """Make the final rename of the state file fail, as a full disk would."""
    real_replace = Path.replace

    def picky(self, target):
        if Path(target).name == "review_state.json":
            raise OSError("no space left on device")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", picky)


def test_a_failed_decision_write_leaves_neither_disk_nor_memory_changed(
        tmp_path, monkeypatch):
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "accept", "timestamp": 1}, "fp1")
    on_disk = (output / "review_state.json").read_bytes()

    _fail_the_state_write(monkeypatch)
    with pytest.raises(OSError):
        state.apply_decision(2, {"action": "mark", "timestamp": 2}, "fp2")
    monkeypatch.undo()

    # Memory matches the disk exactly: the failed decision is nowhere.
    assert state.get_group(2) is None
    assert state.get_group(1)["action"] == "accept"
    assert state.history_depth() == 1
    assert state.summary() == {"total": 1, "reviewed": 1, "marked": 0}
    assert state.queue_tally([1, 2, 3])["counts"] == {"PENDING": 2, "LATER": 0,
                                                     "DONE": 1}
    assert (output / "review_state.json").read_bytes() == on_disk
    assert not list(output.glob(".review_state_*.tmp"))

    # The next successful write does not smuggle the failed decision along.
    state.apply_decision(3, {"action": "mark", "timestamp": 3}, "fp3")
    reloaded = review_state.ReviewState(output)
    assert sorted(reloaded.snapshot()["groups"]) == ["1", "3"]
    assert reloaded.get_group(2) is None
    assert reloaded.history_depth() == 2


def test_a_failed_overwrite_of_an_existing_decision_rolls_back(tmp_path,
                                                               monkeypatch):
    """The previous decision for that group must survive intact."""
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "pick", "file_id": 2, "timestamp": 1}, "fp")

    _fail_the_state_write(monkeypatch)
    with pytest.raises(OSError):
        state.apply_decision(1, {"action": "mark", "timestamp": 2}, "fp")
    monkeypatch.undo()

    kept = state.get_group(1)
    assert kept["action"] == "pick" and kept["file_id"] == 2
    assert state.history_depth() == 1
    assert review_state.ReviewState(output).get_group(1)["action"] == "pick"


def test_a_failed_undo_write_leaves_the_history_intact(tmp_path, monkeypatch):
    """U has to remain pressable: the step must not be consumed."""
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "accept", "timestamp": 1}, "fp1")
    state.apply_decision(2, {"action": "mark", "timestamp": 2}, "fp2")
    on_disk = (output / "review_state.json").read_bytes()

    _fail_the_state_write(monkeypatch)
    with pytest.raises(OSError):
        state.undo_last()
    monkeypatch.undo()

    # The mark was not rolled back in memory, and the step is still there.
    assert state.get_group(2)["action"] == "mark"
    assert state.history_depth() == 2
    assert (output / "review_state.json").read_bytes() == on_disk

    # Pressing U again now works, and undoes exactly one step.
    undone = state.undo_last()
    assert undone["group_id"] == 2 and undone["restored"] is None
    assert state.get_group(2) is None
    assert state.history_depth() == 1
    assert review_state.ReviewState(output).get_group(2) is None


def test_a_failed_clear_write_rolls_back(tmp_path, monkeypatch):
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.apply_decision(1, {"action": "accept", "timestamp": 1}, "fp")

    _fail_the_state_write(monkeypatch)
    with pytest.raises(OSError):
        state.revert_decision(1)
    monkeypatch.undo()

    assert state.get_group(1)["action"] == "accept"
    assert state.history_depth() == 1
    assert review_state.ReviewState(output).get_group(1)["action"] == "accept"


def test_a_failed_set_group_and_clear_group_roll_back(tmp_path, monkeypatch):
    """The non-undoable paths get the same treatment."""
    output = _build(tmp_path)
    state = review_state.ReviewState(output)
    state.set_group(1, {"action": "accept", "timestamp": 1}, "fp")

    _fail_the_state_write(monkeypatch)
    with pytest.raises(OSError):
        state.set_group(2, {"action": "mark", "timestamp": 2}, "fp")
    assert state.get_group(2) is None
    with pytest.raises(OSError):
        state.clear_group(1)
    assert state.get_group(1)["action"] == "accept"
    monkeypatch.undo()
    assert sorted(review_state.ReviewState(output).snapshot()["groups"]) == ["1"]


def test_the_api_reports_nothing_it_did_not_store(tmp_path, monkeypatch):
    output = _build(tmp_path)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        _fail_the_state_write(monkeypatch)

        with pytest.raises(HTTPError) as excinfo:
            _post(base, {"group_id": 2, "action": "mark"})
        assert excinfo.value.code == 400
        monkeypatch.undo()

        # The queues never counted the decision that failed to be written.
        status = _get(base, "/api/status")
        assert status["queues"] == {"PENDING": 2, "LATER": 0, "DONE": 1}
        assert status["undo_depth"] == 1
        later = _get(base, "/api/page?view=GROUPS&queue=LATER&page=1&page_size=50")
        assert later["total"] == 0

        # And undo walks back the accept, not the failed mark.
        undone = _post(base, {"action": "undo"})["undo"]
        assert undone["group_id"] == 1


# --- the warning reaches the UI --------------------------------------------

def test_the_page_envelope_carries_the_state_warning(tmp_path):
    """The workbench loads pages, not status, so the warning rides along."""
    output = _build(tmp_path)
    (output / "review_state.json").write_text("truncated", encoding="utf-8")
    with _Served(output) as base:
        page = _get(base, "/api/page?view=GROUPS&queue=PENDING&page=1&page_size=50")
        assert "state_warning" in page
        assert "review_state.json" in page["state_warning"]


def test_a_healthy_state_carries_no_warning_anywhere(tmp_path):
    output = _build(tmp_path)
    with _Served(output) as base:
        _post(base, {"group_id": 1, "action": "accept"})
        page = _get(base, "/api/page?view=GROUPS&queue=PENDING&page=1&page_size=50")
        assert page["state_warning"] is None
        assert "state_warning" not in _get(base, "/api/status")
