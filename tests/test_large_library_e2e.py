"""Full-pipeline integration for a 100k-style library: thumbs, paging, boundaries.

Uses the deterministic stub backend and synthetic JPEG / Motion-JPEG fixtures,
so it runs on any machine without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

from src import db, execute_local, review_server, run_pipeline, thumbnails

# Xiaomi-style motion photo marker; stage 0 keys on this to classify jpg_motion.
_MOTION_XMP = (
    b'<?xpacket begin="\xef\xbb\xbf"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:GCamera="http://ns.google.com/photos/1.0/camera/" '
    b'GCamera:MicroVideo="1" GCamera:MicroVideoOffset="1024"/>'
    b'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


def _scene(seed: int, size: int = 192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    a, b, c = rng.uniform(-3.0, 3.0, 3)
    g = np.sin(a * np.pi * xx + b * np.pi * yy + c) + np.cos((a + 1.3) * np.pi * yy + c)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[..., 0] = (g * 45).astype(np.uint8)
    out[..., 1] = (g * 255).astype(np.uint8)
    out[..., 2] = ((1.0 - g) * 255).astype(np.uint8)
    return out


def _write_jpeg(path: Path, array: np.ndarray, motion: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if motion:
        Image.fromarray(array).save(path, format="JPEG", quality=92, xmp=_MOTION_XMP)
    else:
        Image.fromarray(array).save(path, format="JPEG", quality=92)


def _library(root: Path) -> dict[str, int]:
    """Plain JPEGs across two month folders, a byte-identical pair, motion photos."""
    counts = {"jpg": 0, "jpg_motion": 0, "mp4_paired": 0}
    for index, hour in enumerate((9, 10, 11, 12, 13, 14)):
        folder = root / ("2026/01" if index < 3 else "2026/02")
        _write_jpeg(folder / f"IMG_202601{index + 1:02d}_{hour:02d}0000.jpg",
                    _scene(700 + index * 13))
        counts["jpg"] += 1

    # Byte-identical duplicate pair -> the only AUTO_REMOVE the policy allows.
    original = root / "2026/01/IMG_20260115_150000.jpg"
    _write_jpeg(original, _scene(999))
    copy = root / "2026/02/IMG_20260115_150000_copy.jpg"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(original.read_bytes())
    counts["jpg"] += 2

    # The real-world main case: the motion video is appended to the JPEG tail,
    # with the vendor XMP marker in the header and no sidecar file at all.
    embedded = root / "2026/02/IMG_20260220_180000.jpg"
    _write_jpeg(embedded, _scene(555), motion=True)
    embedded.write_bytes(
        embedded.read_bytes() + b"\x00\x00\x00\x18ftypmp42" + b"m" * 4096
    )
    counts["jpg_motion"] += 1

    # A paired-sidecar motion photo as well, so the older layout stays covered.
    paired_still = root / "2026/02/IMG_20260221_181500.jpg"
    _write_jpeg(paired_still, _scene(556))
    (root / "2026/02/IMG_20260221_181500.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"v" * 64)
    counts["jpg_motion"] += 1
    counts["mp4_paired"] += 1

    # A deliberately corrupt file: thumbnails must fail honestly, not silently.
    (root / "2026/02/IMG_20260222_190000.jpg").write_bytes(b"definitely not a jpeg")
    counts["jpg"] += 1
    return counts


def _serve(output: Path):
    server, url = review_server.start_server(output)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, url.rsplit("/", 1)[0]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("e2e")
    root = tmp_path / "Photos"
    output = tmp_path / "review"
    counts = _library(root)
    result = run_pipeline.run(str(root), str(output), backend="stub", thumb_px=96)
    return {"root": root, "output": output, "result": result, "counts": counts}


# --- stage 1 thumbnail coverage -------------------------------------------

def test_every_readable_still_image_gets_a_thumbnail(pipeline):
    output = pipeline["output"]
    conn = db.open_db(output / "inventory.sqlite")
    still_images = db.count_still_images(conn)
    stats = db.thumbnail_stats(conn)
    conn.close()
    # Every still image is accounted for: cached or explicitly failed.
    assert stats["recorded_ok"] + stats["recorded_failed"] == still_images
    assert stats["recorded_failed"] == 1          # only the corrupt file
    assert stats["recorded_ok"] == still_images - 1
    cached = list(thumbnails.thumbs_dir(output).glob("*.jpg"))
    assert len(cached) == stats["recorded_ok"]
    for thumb in cached:
        with Image.open(thumb) as image:
            assert max(image.size) <= 96


def test_thumbnail_failure_is_reported_not_hidden(pipeline):
    output = pipeline["output"]
    conn = db.open_db(output / "inventory.sqlite")
    failures = db.thumbnail_failures(conn)
    conn.close()
    assert len(failures) == 1
    assert failures[0]["basename"] == "IMG_20260222_190000.jpg"
    summary = json.loads((output / "review_summary.json").read_text(encoding="utf-8"))
    assert summary["thumbnails"]["recorded_failed"] == 1
    assert summary["thumbnail_failures"][0]["basename"] == "IMG_20260222_190000.jpg"
    assert "failed: 1" in (output / "summary.txt").read_text(encoding="utf-8")


def test_rerun_reuses_thumbnails_without_reopening_sources(pipeline, monkeypatch):
    from src import stage1_features

    monkeypatch.setattr(
        stage1_features, "_open_image_and_sha",
        lambda path: (_ for _ in ()).throw(AssertionError(f"reopened {path}")),
    )
    again = run_pipeline.run(str(pipeline["root"]), str(pipeline["output"]),
                             backend="stub", thumb_px=96)
    assert again["features"]["processed"] == 0
    assert again["features"]["thumbnails"]["created"] == 0


# --- performance and disk reporting ---------------------------------------

def test_performance_and_disk_numbers_are_visible(pipeline):
    output = pipeline["output"]
    performance = pipeline["result"]["performance"]
    for stage in ("stage0", "stage1", "stage2", "stage3"):
        assert performance["stage_seconds"][stage] >= 0
    assert performance["stage1_images_per_second"] > 0
    assert performance["stage0_files_per_second"] > 0
    assert performance["eta_hours"] is not None

    text = (output / "performance.txt").read_text(encoding="utf-8")
    for needle in ("stage0 wall time", "stage1 wall time", "stage2 wall time",
                   "stage3 wall time", "files/s", "images/s",
                   "ROUGH full-library Stage 1 ETA", "ROUGH LINEAR ESTIMATE ONLY",
                   "Thumbnail cache (SSD)", "estimated full-library usage",
                   "SSD free space", "actual usage"):
        assert needle in text, needle

    disk = pipeline["result"]["thumbnail_disk"]
    assert disk["actual_bytes"] > 0
    assert disk["estimated_total_bytes"] is not None
    assert disk["free_bytes"] is not None

    page = (output / "review.html").read_text(encoding="utf-8")
    assert "ROUGH full-library Stage 1 ETA" in page
    assert "SSD free space" in page


# --- paged review over the served output ----------------------------------

def test_all_timeline_pages_completely_and_matches_still_image_count(pipeline):
    output = pipeline["output"]
    conn = db.open_db(output / "inventory.sqlite")
    still_images = db.count_still_images(conn)
    conn.close()
    server, thread, base = _serve(output)
    try:
        payload = json.loads(urlopen(f"{base}/api/page?view=ALL&page_size=50").read())
        assert payload["total"] == still_images
        assert payload["shown"] == still_images        # small fixture fits one page
        timestamps = [item["exif_datetime"] for item in payload["items"]]
        assert timestamps == sorted(timestamps, key=lambda value: value or "")
        for size in (50, 100, 200):
            sized = json.loads(
                urlopen(f"{base}/api/page?view=ALL&page_size={size}").read())
            assert sized["page_size"] == size
            assert sized["total"] == still_images
        # The corrupt photo is present in the timeline with an honest thumb state.
        broken = [item for item in payload["items"]
                  if item["basename"] == "IMG_20260222_190000.jpg"]
        assert broken and broken[0]["thumb"] == "error"
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        thread.join()


def test_served_thumbnails_come_from_the_ssd_cache_only(pipeline, monkeypatch):
    output = pipeline["output"]
    monkeypatch.setattr(
        Image, "open",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no decode while browsing")),
    )
    server, thread, base = _serve(output)
    try:
        payload = json.loads(urlopen(f"{base}/api/page?view=ALL&page_size=100").read())
        served = 0
        for item in payload["items"]:
            if item["thumb"] == "ok":
                assert urlopen(f"{base}/api/thumb/{item['file_id']}.jpg").status == 200
                served += 1
            else:
                with pytest.raises(HTTPError) as excinfo:
                    urlopen(f"{base}/api/thumb/{item['file_id']}.jpg")
                assert excinfo.value.code == 404
        assert served > 0
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        thread.join()


def test_groups_maybe_and_unknown_views_are_reachable(pipeline):
    output = pipeline["output"]
    server, thread, base = _serve(output)
    try:
        status = json.loads(urlopen(f"{base}/api/status").read())
        assert set(status["counts"]) == {"ALL", "MAYBE", "UNKNOWN", "GROUPS"}
        for view in ("MAYBE", "UNKNOWN", "GROUPS"):
            payload = json.loads(urlopen(f"{base}/api/page?view={view}&page_size=50").read())
            assert payload["view"] == view
            assert payload["total"] == status["counts"][view]
        groups = json.loads(urlopen(f"{base}/api/page?view=GROUPS&page_size=50").read())
        assert groups["total"] >= 1
        assert all("members" in item for item in groups["items"])
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        thread.join()


# --- safety boundaries -----------------------------------------------------

def test_embedded_motion_photo_is_classified_thumbnailed_and_intact(pipeline):
    """Video bytes appended to the JPEG tail: one decode still yields a thumbnail."""
    output = pipeline["output"]
    source = pipeline["root"] / "2026/02/IMG_20260220_180000.jpg"
    raw = source.read_bytes()
    assert b"ftypmp42" in raw            # the video really is inside the JPEG
    conn = db.open_db(output / "inventory.sqlite")
    row = db.get_file_by_path(conn, str(source))
    assert row["file_kind"] == "jpg_motion"
    assert row["motion_partner_id"] is None      # embedded, so no sidecar to pair
    thumb_status = conn.execute(
        "SELECT status FROM thumbnails WHERE file_id = ?", (row["id"],)).fetchone()
    conn.close()
    assert thumb_status["status"] == "ok"
    thumb = thumbnails.thumb_path(thumbnails.thumbs_dir(output), int(row["id"]))
    with Image.open(thumb) as image:
        assert max(image.size) <= 96
    # The original, video payload included, is untouched.
    assert source.read_bytes() == raw


def test_paired_sidecar_video_is_not_a_still_image_but_stays_linked(pipeline):
    output = pipeline["output"]
    conn = db.open_db(output / "inventory.sqlite")
    still = db.get_file_by_path(conn, str(pipeline["root"] / "2026/02/IMG_20260221_181500.jpg"))
    video = db.get_file_by_path(conn, str(pipeline["root"] / "2026/02/IMG_20260221_181500.mp4"))
    thumb_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM thumbnails WHERE file_id = ?", (video["id"],)).fetchone()
    conn.close()
    assert still["file_kind"] == "jpg_motion"
    assert video["file_kind"] == "mp4_paired"
    assert still["motion_partner_id"] == video["id"]
    assert thumb_rows["n"] == 0          # videos are not thumbnailed or analysed


def test_review_page_is_chinese_and_localhost_only(pipeline):
    page = (pipeline["output"] / "review.html").read_text(encoding="utf-8")
    assert 'lang="zh-CN"' in page
    assert "照片去重" in page
    assert "全部（时间线）" in page
    assert "每页" in page
    # Internal state names stay English, as agreed.
    assert 'data-view="MAYBE"' in page and 'data-view="UNKNOWN"' in page
    server, thread, base = _serve(pipeline["output"])
    try:
        assert base.startswith("http://127.0.0.1:")
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        thread.join()


def test_ui_exposes_no_manual_marking_endpoint_this_milestone(pipeline):
    """Manual KEEP/REVIEW-LATER marking is deliberately not in this milestone."""
    output = pipeline["output"]
    conn = db.open_db(output / "inventory.sqlite")
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "review_marks" not in tables
    page = (output / "review.html").read_text(encoding="utf-8")
    for absent in ("/api/mark", "method:'POST'", 'method="post"'):
        assert absent not in page


def test_only_byte_identical_duplicates_are_proposed_for_removal(pipeline):
    output = pipeline["output"]
    manifest = (output / "delete_local.txt").read_text(encoding="utf-8").split()
    assert manifest, "the byte-identical pair should yield exactly one removal"
    conn = db.open_db(output / "inventory.sqlite")
    rows = conn.execute(
        "SELECT gm.decision, fe.content_sha256, f.path FROM group_members gm "
        "JOIN files f ON f.id = gm.file_id LEFT JOIN features fe ON fe.file_id = gm.file_id"
    ).fetchall()
    conn.close()
    removals = [r for r in rows if r["decision"] == "AUTO_REMOVE"]
    assert len(removals) == len(manifest)
    hashes = {r["content_sha256"] for r in removals}
    for removal in removals:
        peers = [r for r in rows if r["content_sha256"] == removal["content_sha256"]
                 and r["path"] != removal["path"]]
        assert peers, "an AUTO_REMOVE must have a byte-identical peer"
    assert None not in hashes


def test_originals_are_untouched_by_the_whole_run(pipeline):
    root = pipeline["root"]
    files = sorted(p for p in root.rglob("*") if p.is_file())
    assert len(files) >= 10
    # Nothing was moved into a trash directory and no source vanished.
    assert not (root / "_trash").exists()
    conn = db.open_db(pipeline["output"] / "inventory.sqlite")
    inventoried = {Path(r["path"]) for r in conn.execute("SELECT path FROM files")}
    conn.close()
    assert inventoried.issubset(set(files))


def test_pipeline_module_never_reaches_execute_local(pipeline, monkeypatch):
    """No import of, and no call into, the deletion stage -- checked structurally."""
    import ast

    monkeypatch.setattr(
        execute_local, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    tree = ast.parse(Path(run_pipeline.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
    assert not any("execute_local" in name for name in imported)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "execute_local" not in attributes
    # And running it again really does not delete anything.
    before = sorted(p.name for p in pipeline["root"].rglob("*") if p.is_file())
    run_pipeline.run(str(pipeline["root"]), str(pipeline["output"]),
                     backend="stub", thumb_px=96)
    after = sorted(p.name for p in pipeline["root"].rglob("*") if p.is_file())
    assert before == after


def test_runtime_config_does_not_modify_the_repository_config(pipeline):
    from src import config as config_module

    text = config_module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert f'root: "{pipeline["root"]}"' not in text
    assert str(pipeline["output"]) not in text
