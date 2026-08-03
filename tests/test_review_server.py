"""The review server pages a large library from SQLite and never touches the HDD."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from PIL import Image

from src import db, review_server, thumbnails


def _build_output(tmp_path: Path, count: int = 250, thumbs: bool = True,
                  groups: int = 0) -> tuple[Path, Path]:
    """A realistic output dir: inventory DB, review index, SSD thumbnails."""
    output = tmp_path / "output"
    photos = tmp_path / "photos"
    output.mkdir(parents=True)
    photos.mkdir(parents=True)
    directory = thumbnails.thumbs_dir(output)
    directory.mkdir(parents=True, exist_ok=True)
    conn = db.open_db(output / "inventory.sqlite")

    file_ids = []
    for index in range(count):
        source = photos / f"IMG_2026{index:04d}.jpg"
        source.write_bytes(b"pretend original bytes")
        file_id = db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": 1234,
            "mtime_ns": 10 + index,
            "exif_datetime": f"2026-01-01 12:{index % 60:02d}:00",
            # Deliberately reversed timestamps so ALL ordering is observable.
            "exif_timestamp": 100000 - index,
            "width": 4000, "height": 3000, "file_kind": "jpg", "scan_status": "done",
        })
        file_ids.append(file_id)
        db.batch_insert_features(conn, [{
            "file_id": file_id, "phash": None, "dinov2_embedding": None,
            "quality_score": 50.0 + index, "quality_meta": "{}", "face_count": 1,
            "faces_json": "[]", "status": "done",
        }])
        if thumbs:
            Image.new("RGB", (32, 24), (index % 255, 40, 90)).save(
                thumbnails.thumb_path(directory, file_id), "JPEG")
            db.batch_upsert_thumbnails(conn, [{
                "file_id": file_id, "status": "ok", "max_px": 320, "bytes": 900,
                "source_size_bytes": 1234, "source_mtime_ns": 10 + index,
                "error": None, "created_at": 1,
            }])

    for number in range(groups):
        members = file_ids[number * 2:number * 2 + 2]
        group_id = db.insert_group(conn, "burst", members[0],
                                   [(members[0], True, "keep"), (members[1], False, "dup")], 1)
        db.update_member_decisions(conn, group_id, [
            {"file_id": members[0], "decision": "KEEP", "confidence": 1.0,
             "reason": "keep", "evidence_json": "{}"},
            {"file_id": members[1], "decision": "MAYBE", "confidence": 0.5,
             "reason": "near duplicate", "evidence_json": "{}"},
        ])

    db.build_all_view_index(conn)
    db.replace_review_index(conn, "MAYBE", [
        {"file_id": fid, "group_id": None, "decision": "MAYBE", "risk": 0.5}
        for fid in file_ids[:120]
    ])
    db.replace_review_index(conn, "UNKNOWN", [
        {"file_id": fid, "group_id": None, "decision": "UNKNOWN", "risk": 0.1}
        for fid in file_ids[120:140]
    ])
    conn.commit()
    conn.close()

    (output / "review.html").write_text(
        '<h1>Photo Dedup - Review</h1><div class="notice" id="perf">p</div>', encoding="utf-8")
    (output / "review_summary.json").write_text(
        json.dumps({"views": {"ALL": count}}), encoding="utf-8")
    return output, photos


def _running(output: Path):
    server, url = review_server.start_server(output)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, url.rsplit("/", 1)[0]


class _Served:
    """Context manager that guarantees the server and DB handle are released."""

    def __init__(self, output: Path) -> None:
        self.output = output

    def __enter__(self) -> str:
        self.server, self.thread, self.base = _running(self.output)
        return self.base

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.review_data.close()
        self.thread.join()


def _get(base: str, path: str):
    return json.loads(urlopen(base + path).read())


# --- the central guarantee -------------------------------------------------

def test_paging_and_thumbnails_never_open_an_original(tmp_path, monkeypatch):
    output, photos = _build_output(tmp_path, 250)
    originals = {str(p) for p in photos.iterdir()}
    opened: list[str] = []

    real_open = Path.open

    def tracked_open(self, *args, **kwargs):
        if str(self) in originals:
            opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(
        Image, "open",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("UI must not decode originals")),
    )

    with _Served(output) as base:
        for view in ("ALL", "MAYBE", "UNKNOWN", "GROUPS"):
            page = _get(base, f"/api/page?view={view}&page=1&page_size=100")
            assert page["view"] == view
        first = _get(base, "/api/page?view=ALL&page=1&page_size=50")
        for item in first["items"][:10]:
            assert urlopen(f"{base}/api/thumb/{item['file_id']}.jpg").status == 200
        _get(base, "/api/status")

    assert opened == []


def test_thumbnail_endpoint_has_no_source_fallback(tmp_path):
    """A missing cached thumbnail is a 404, not a re-read of the original."""
    output, _photos = _build_output(tmp_path, 3, thumbs=True)
    victim = sorted(thumbnails.thumbs_dir(output).glob("*.jpg"))[0]
    file_id = int(victim.stem)
    victim.unlink()
    with _Served(output) as base:
        with pytest.raises(HTTPError) as excinfo:
            urlopen(f"{base}/api/thumb/{file_id}.jpg")
        assert excinfo.value.code == 404
        assert not victim.exists()  # nothing regenerated it behind our back


def test_paths_are_never_exposed_to_the_browser(tmp_path):
    output, photos = _build_output(tmp_path, 20)
    with _Served(output) as base:
        page = _get(base, "/api/page?view=ALL&page=1&page_size=50")
        blob = json.dumps(page)
        assert "path" not in page["items"][0]
        for source in photos.iterdir():
            assert str(source) not in blob
            assert str(source.parent) not in blob


# --- ALL view completeness and ordering ------------------------------------

def test_all_view_is_complete_across_pages(tmp_path):
    output, _photos = _build_output(tmp_path, 250)
    with _Served(output) as base:
        seen: list[int] = []
        page = 1
        while True:
            payload = _get(base, f"/api/page?view=ALL&page={page}&page_size=100")
            seen.extend(item["file_id"] for item in payload["items"])
            if page >= payload["pages"]:
                assert payload["total"] == 250
                assert payload["pages"] == 3
                break
            page += 1
        assert len(seen) == 250
        assert len(set(seen)) == 250


def test_all_view_is_ordered_by_capture_time_then_directory_and_name(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    conn = db.open_db(output / "inventory.sqlite")
    rows = [
        ("/photos/2026/02/b.jpg", "b.jpg", 200),
        ("/photos/2026/01/z.jpg", "z.jpg", 100),
        ("/photos/2026/01/a.jpg", "a.jpg", 100),
        ("/photos/2026/00/m.jpg", "m.jpg", 100),
        ("/photos/2026/03/c.jpg", "c.jpg", None),   # falls back to mtime
    ]
    for index, (path, basename, timestamp) in enumerate(rows):
        db.insert_file(conn, {
            "path": path, "basename": basename, "size_bytes": 1, "mtime_ns": 50 * 10 ** 9,
            "exif_timestamp": timestamp, "file_kind": "jpg", "scan_status": "done",
            "width": 1, "height": 1,
        })
        del index
    db.build_all_view_index(conn)
    conn.commit()
    order = [r["basename"] for r in db.review_page(conn, "ALL", 0, 50)]
    conn.close()
    # mtime fallback (50s) sorts first, then ts=100 by directory then name, then ts=200.
    assert order == ["c.jpg", "m.jpg", "a.jpg", "z.jpg", "b.jpg"]


def test_all_view_includes_ungrouped_and_grouped_files_exactly_once(tmp_path):
    output, _photos = _build_output(tmp_path, 20, groups=4)
    with _Served(output) as base:
        payload = _get(base, "/api/page?view=ALL&page=1&page_size=50")
        assert payload["total"] == 20
        ids = [item["file_id"] for item in payload["items"]]
        assert len(ids) == len(set(ids)) == 20
        decisions = {item["decision"] for item in payload["items"]}
        assert "UNGROUPED" in decisions
        assert {"KEEP", "MAYBE"} & decisions


# --- page sizes and paging behaviour ---------------------------------------

@pytest.mark.parametrize("size,pages", [(50, 5), (100, 3), (200, 2)])
def test_supported_page_sizes(tmp_path, size, pages):
    output, _photos = _build_output(tmp_path, 250)
    with _Served(output) as base:
        payload = _get(base, f"/api/page?view=ALL&page=1&page_size={size}")
        assert payload["page_size"] == size
        assert payload["shown"] == size
        assert payload["pages"] == pages
        assert payload["remaining_after_page"] == 250 - size


def test_default_page_size_is_one_hundred(tmp_path):
    output, _photos = _build_output(tmp_path, 250)
    with _Served(output) as base:
        assert _get(base, "/api/page?view=ALL")["page_size"] == 100
    assert review_server.DEFAULT_PAGE_SIZE == 100


def test_unsupported_page_size_and_view_are_rejected(tmp_path):
    output, _photos = _build_output(tmp_path, 10)
    with _Served(output) as base:
        for query in ("view=ALL&page_size=75", "view=ALL&page_size=1000",
                      "view=EVERYTHING", "view=ALL&page=0", "view=ALL&page=abc"):
            with pytest.raises(HTTPError) as excinfo:
                urlopen(f"{base}/api/page?{query}")
            assert excinfo.value.code == 400
    assert review_server.PAGE_SIZES == (50, 100, 200)


def test_last_page_is_clamped_and_partial(tmp_path):
    output, _photos = _build_output(tmp_path, 250)
    with _Served(output) as base:
        payload = _get(base, "/api/page?view=ALL&page=99&page_size=100")
        assert payload["page"] == 3
        assert payload["shown"] == 50
        assert payload["remaining_after_page"] == 0


def test_queue_and_group_views_report_their_own_totals(tmp_path):
    output, _photos = _build_output(tmp_path, 250, groups=5)
    with _Served(output) as base:
        assert _get(base, "/api/page?view=MAYBE&page_size=50")["total"] == 120
        assert _get(base, "/api/page?view=UNKNOWN&page_size=50")["total"] == 20
        groups = _get(base, "/api/page?view=GROUPS&page_size=50")
        assert groups["total"] == 5
        assert len(groups["items"]) == 5
        assert groups["items"][0]["member_count"] == 2
        assert len(groups["items"][0]["members"]) == 2
        assert groups["items"][0]["members"][0]["is_keep"] is True
        status = _get(base, "/api/status")
        assert status["counts"] == {"ALL": 250, "MAYBE": 120, "UNKNOWN": 20, "GROUPS": 5}
        assert status["page_sizes"] == [50, 100, 200]
        assert status["thumb_cache_files"] == 250


def test_missing_thumbnail_status_is_surfaced_to_the_ui(tmp_path):
    output, _photos = _build_output(tmp_path, 5, thumbs=False)
    conn = db.open_db(output / "inventory.sqlite")
    conn.execute(
        "INSERT INTO thumbnails (file_id, status, max_px, error, created_at) "
        "VALUES (1, 'error', 320, 'SOURCE_DECODE_FAILED: broken', 1)")
    conn.commit()
    conn.close()
    with _Served(output) as base:
        items = _get(base, "/api/page?view=ALL&page_size=50")["items"]
        by_id = {item["file_id"]: item for item in items}
        assert by_id[1]["thumb"] == "error"
        assert "SOURCE_DECODE_FAILED" in by_id[1]["thumb_error"]
        assert by_id[2]["thumb"] == "missing"


# --- boundaries ------------------------------------------------------------

def test_server_denies_db_thumb_directory_and_traversal(tmp_path):
    output, _photos = _build_output(tmp_path, 3)
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    with _Served(output) as base:
        for probe in ("/inventory.sqlite", "/thumbs/1.jpg", "/%74humbs/1.jpg",
                      "/../secret.txt", "/api/thumb/../../secret.jpg",
                      "/api/thumb/99999.jpg", "/api/thumb/abc.jpg"):
            with pytest.raises(HTTPError) as excinfo:
                urlopen(base + probe)
            assert excinfo.value.code == 404, probe


def test_review_ui_offers_no_mutation_endpoint(tmp_path):
    output, _photos = _build_output(tmp_path, 3)
    with _Served(output) as base:
        import urllib.request

        request = urllib.request.Request(f"{base}/api/page", method="POST", data=b"x")
        with pytest.raises(HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code in (400, 405, 501)


def test_server_binds_loopback_only(tmp_path):
    output, _photos = _build_output(tmp_path, 2)
    server, url = review_server.start_server(output)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert url.startswith("http://127.0.0.1:")
    finally:
        server.server_close()
        server.review_data.close()


def test_thumb_bytes_rejects_ids_escaping_the_cache_directory(tmp_path):
    output, _photos = _build_output(tmp_path, 2)
    data = review_server.ReviewData(output, output / "inventory.sqlite")
    try:
        assert data.thumb_bytes(999999) is None
        assert data.thumb_bytes(1) is not None
    finally:
        data.close()
