"""Serve a small synthetic review output so the album UI can be driven for real.

Companion to :mod:`album_viewer_browser`: builds a throwaway output directory
(12 photos in 4 groups of 3, plus 2 ungrouped, plus one deliberately failed
thumbnail) and serves it on a fixed loopback port. It writes only under its own
temporary directory and, like the real server, never exposes source paths.

    ./.venv/bin/python scripts/verification/album_viewer_fixture.py [port]

Runs until interrupted. Not a pytest fixture -- pytest covers the same rendering
logic in tests/test_review_album_ui.py without a browser.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw  # noqa: E402

from src import db, review_page, review_server, thumbnails  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18911
GROUPS = 4
PER_GROUP = 3
UNGROUPED = 2
NO_THUMB_INDEX = 5  # exercise the "thumbnail unavailable" tile
COLORS = [(200, 60, 60), (60, 180, 90), (70, 110, 210), (220, 180, 50), (170, 90, 200)]


def build(output: Path, photos: Path) -> None:
    output.mkdir(parents=True)
    photos.mkdir(parents=True)
    thumbs = thumbnails.thumbs_dir(output)
    thumbs.mkdir(parents=True, exist_ok=True)
    conn = db.open_db(output / "inventory.sqlite")

    file_ids = []
    for index in range(GROUPS * PER_GROUP + UNGROUPED):
        source = photos / f"IMG_{index:04d}.jpg"
        image = Image.new("RGB", (900, 675), COLORS[index % len(COLORS)])
        draw = ImageDraw.Draw(image)
        draw.text((40, 40), f"PHOTO {index}", fill=(255, 255, 255))
        draw.rectangle([30, 30, 870, 645], outline=(255, 255, 255), width=6)
        image.save(source, "JPEG")
        stat = source.stat()
        file_id = db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "exif_datetime": f"2026-01-01 12:00:{index:02d}",
            "exif_timestamp": 1000 + index, "width": 900, "height": 675,
            "file_kind": "jpg", "scan_status": "done"})
        file_ids.append(file_id)
        db.batch_insert_features(conn, [{
            "file_id": file_id, "phash": None, "dinov2_embedding": None,
            "quality_score": 40.0 + index * 3, "quality_meta": "{}", "face_count": 1,
            "faces_json": "[]", "status": "done"}])
        if index == NO_THUMB_INDEX:
            db.batch_upsert_thumbnails(conn, [{
                "file_id": file_id, "status": "error", "max_px": 320, "bytes": 0,
                "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                "error": "decode failed", "created_at": 1}])
            continue
        Image.new("RGB", (320, 240), COLORS[index % len(COLORS)]).save(
            thumbnails.thumb_path(thumbs, file_id), "JPEG")
        db.batch_upsert_thumbnails(conn, [{
            "file_id": file_id, "status": "ok", "max_px": 320, "bytes": 900,
            "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "error": None, "created_at": 1}])

    for group in range(GROUPS):
        members = file_ids[group * PER_GROUP:(group + 1) * PER_GROUP]
        keeper = members[1]  # deliberately not the first, so focus != keeper
        group_id = db.insert_group(
            conn, "phash_near", keeper,
            [(m, m == keeper, "keep" if m == keeper else "dup") for m in members], 1)
        db.update_member_decisions(conn, group_id, [
            {"file_id": m, "decision": "KEEP" if m == keeper else "MAYBE",
             "confidence": 1.0, "reason": "GROUP_KEEPER" if m == keeper else "LOW_MARGIN",
             "evidence_json": "{}"} for m in members])
    db.build_all_view_index(conn)
    conn.commit()
    conn.close()

    data = {"groups": [], "queues": {"MAYBE": [], "UNKNOWN": []},
            "delete_paths": [], "total_delete_bytes": 0}
    (output / "review.html").write_text(
        review_page.render_html(data, output, review_limit=0), encoding="utf-8")
    (output / "review_summary.json").write_text('{"views": {}}', encoding="utf-8")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="photo-dedup-album-fixture-"))
    try:
        build(workdir / "output", workdir / "photos")
        server, url = review_server.start_server(workdir / "output", port=PORT)
        print(url, flush=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()
            server.review_data.close()
            thread.join()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
