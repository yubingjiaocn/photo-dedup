"""Stage 1 thumbnails must come from the one decode Stage 1 already performs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src import db, stage1_features, thumbnails
from src.config import load_config


def _library(root: Path, count: int = 3, size=(640, 480)) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"IMG_2026010{index}_120000.jpg"
        Image.new("RGB", size, (10 + index * 30, 60, 120)).save(path, "JPEG", quality=92)
        paths.append(path)
    return paths


def _config(tmp_path: Path, root: Path, **thumb) -> str:
    cfg = {
        "paths": {
            "root": str(root), "db": str(tmp_path / "inventory.sqlite"),
            "output_dir": str(tmp_path / "output"), "models_dir": str(tmp_path / "models"),
            "trash": str(tmp_path / "trash"),
        },
        "features": {"backend": "stub", "batch_size": 4,
                     "thumbnails": {"enabled": True, "max_px": 64, **thumb}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _inventory(conn, path: Path, file_id_hint: int | None = None) -> int:
    stat = path.stat()
    return db.insert_file(conn, {
        "path": str(path), "basename": path.name, "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "width": 640, "height": 480,
        "exif_timestamp": 1000 + (file_id_hint or 0), "file_kind": "jpg",
        "scan_status": "done",
    })


# --- the central guarantee -------------------------------------------------

def test_thumbnail_is_generated_from_the_single_stage1_decode(tmp_path, monkeypatch):
    """One open + one decode per source; the thumb comes from that same object."""
    root = tmp_path / "photos"
    paths = _library(root, 2)
    conn = db.open_db(tmp_path / "db.sqlite")
    rows = [dict(db.get_file(conn, _inventory(conn, p, i))) for i, p in enumerate(paths)]
    conn.commit()

    opens: list[str] = []
    real_open = stage1_features._open_image_and_sha

    def counted(path, telemetry=None):
        opens.append(str(path))
        return real_open(path, telemetry)

    monkeypatch.setattr(stage1_features, "_open_image_and_sha", counted)

    captured: list[tuple[int, int]] = []
    thumbnailer = thumbnails.Thumbnailer(tmp_path / "thumbs", max_px=64)
    real_capture = thumbnailer.capture

    def tracking_capture(row, image):
        captured.append((int(row["id"]), id(image)))
        return real_capture(row, image)

    monkeypatch.setattr(thumbnailer, "capture", tracking_capture)

    # PIL must not be asked to open anything beyond the two counted reads.
    original_pil_open = Image.open
    pil_opens: list[object] = []

    def tracked_pil_open(fp, *args, **kwargs):
        pil_opens.append(fp)
        return original_pil_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked_pil_open)

    stage1_features._process_batch(
        stage1_features.StubBackend(), rows, load_config(), thumbnailer=thumbnailer
    )

    assert opens == [str(p) for p in paths]           # exactly one read per file
    assert len(pil_opens) == 2                        # exactly one decode per file
    assert all(not isinstance(fp, (str, Path)) for fp in pil_opens)  # in-memory buffers
    assert len(captured) == 2
    assert thumbnailer.created == 2
    for path in paths:
        assert str(path) not in [str(fp) for fp in pil_opens]
    for (file_id, _), _path in zip(captured, paths):
        thumb = thumbnails.thumb_path(thumbnailer.directory, file_id)
        assert thumb.is_file()
        with Image.open(thumb) as image:
            assert max(image.size) <= 64
            assert image.format == "JPEG"
    conn.close()


def test_thumbnail_object_identity_matches_the_decoded_feature_image(tmp_path):
    """The thumbnailer receives the very same PIL object the extractors used."""
    root = tmp_path / "photos"
    path = _library(root, 1)[0]
    conn = db.open_db(tmp_path / "db.sqlite")
    row = dict(db.get_file(conn, _inventory(conn, path)))
    conn.commit()

    seen: list[int] = []

    class RecordingBackend(stage1_features.StubBackend):
        def embed_batch(self, images):
            seen.extend(id(image) for image in images)
            return super().embed_batch(images)

    class RecordingThumbnailer(thumbnails.Thumbnailer):
        def capture(self, row, image):
            seen.append(id(image))
            return super().capture(row, image)

    thumbnailer = RecordingThumbnailer(tmp_path / "thumbs", max_px=48)
    stage1_features._process_batch(
        RecordingBackend(), [row], load_config(), thumbnailer=thumbnailer
    )
    assert len(seen) == 2 and seen[0] == seen[1]
    conn.close()


def test_render_thumbnail_does_not_mutate_the_shared_image():
    source = Image.new("RGB", (800, 600), "teal")
    small = thumbnails.render_thumbnail(source, 100)
    assert source.size == (800, 600)      # extractors still see full resolution
    assert max(small.size) == 100
    assert small.mode == "RGB"
    assert small is not source


# --- coverage, resume, staleness ------------------------------------------

def test_full_coverage_then_reuse_without_reopening_sources(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    _library(root, 4)
    config = _config(tmp_path, root)

    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    first = stage1_features.run(config_path=config, backend_override="stub")
    assert first["processed"] == 4
    assert first["thumbnails"]["created"] == 4
    assert first["thumbnails"]["failed"] == 0
    assert first["thumbnails"]["cache_files"] == 4

    opens: list[str] = []
    monkeypatch.setattr(
        stage1_features, "_open_image_and_sha",
        lambda path, *_args: opens.append(str(path)) or (_ for _ in ()).throw(
            AssertionError("re-run must not reopen sources")),
    )
    second = stage1_features.run(config_path=config, backend_override="stub")
    assert second["processed"] == 0
    assert opens == []


def test_changed_source_invalidates_the_cached_thumbnail(tmp_path):
    root = tmp_path / "photos"
    paths = _library(root, 2)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")

    conn = db.open_db(tmp_path / "inventory.sqlite")
    target = db.get_file_by_path(conn, str(paths[0]))
    file_id = int(target["id"])
    stale_bytes = thumbnails.thumb_path(
        thumbnails.thumbs_dir(tmp_path / "output"), file_id).read_bytes()
    conn.close()

    # Replace the photo with different content; a plain re-scan must notice.
    Image.new("RGB", (640, 480), "orange").save(paths[0], "JPEG", quality=95)
    inventory = stage0_inventory.run(config_path=config)
    assert inventory["changed_files"] == 1

    again = stage1_features.run(config_path=config, backend_override="stub")
    assert again["processed"] == 1
    assert again["thumbnails"]["created"] == 1
    fresh_bytes = thumbnails.thumb_path(
        thumbnails.thumbs_dir(tmp_path / "output"), file_id).read_bytes()
    assert fresh_bytes != stale_bytes


def test_failed_redecode_removes_stale_thumbnail_file(tmp_path):
    root = tmp_path / "photos"
    path = _library(root, 1)[0]
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")
    conn = db.open_db(tmp_path / "inventory.sqlite")
    file_id = int(db.get_file_by_path(conn, str(path))["id"])
    conn.close()
    cached = thumbnails.thumb_path(thumbnails.thumbs_dir(tmp_path / "output"), file_id)
    assert cached.is_file()

    path.write_bytes(b"not a jpeg anymore")
    stage0_inventory.run(config_path=config)
    result = stage1_features.run(config_path=config, backend_override="stub")

    assert result["thumbnails"]["failed"] == 1
    assert not cached.exists()
    conn = db.open_db(tmp_path / "inventory.sqlite")
    row = conn.execute("SELECT status FROM thumbnails WHERE file_id = ?", (file_id,)).fetchone()
    assert row["status"] == "error"
    conn.close()


def test_rescan_of_unchanged_files_reports_no_change(tmp_path):
    root = tmp_path / "photos"
    _library(root, 3)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")
    inventory = stage0_inventory.run(config_path=config)
    assert inventory["changed_files"] == 0
    again = stage1_features.run(config_path=config, backend_override="stub")
    assert again["processed"] == 0
    assert again["thumbnails"]["created"] == 0


def test_unchanged_rescan_does_not_reopen_headers(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    _library(root, 3)
    config = _config(tmp_path, root)
    from src import motion_photo, stage0_inventory

    stage0_inventory.run(config_path=config)
    monkeypatch.setattr(
        stage0_inventory, "read_image_header",
        lambda *_: (_ for _ in ()).throw(AssertionError("Pillow header reopened")),
    )
    monkeypatch.setattr(
        motion_photo, "detect_embedded_motion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("motion header reopened")
        ),
    )
    inventory = stage0_inventory.run(config_path=config)
    assert inventory["changed_files"] == 0


def test_limited_inventory_advances_to_new_files_on_rerun(tmp_path):
    root = tmp_path / "photos"
    _library(root, 5)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    assert stage0_inventory.run(config_path=config, limit=2)["files"] == 2
    assert stage0_inventory.run(config_path=config, limit=2)["files"] == 4
    assert stage0_inventory.run(config_path=config, limit=2)["files"] == 5


def test_changed_source_also_invalidates_the_stale_content_hash(tmp_path):
    """Byte identity drives the only AUTO_REMOVE, so a stale SHA is unacceptable."""
    root = tmp_path / "photos"
    paths = _library(root, 2)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")
    conn = db.open_db(tmp_path / "inventory.sqlite")
    file_id = int(db.get_file_by_path(conn, str(paths[0]))["id"])
    before = conn.execute(
        "SELECT content_sha256 FROM features WHERE file_id = ?", (file_id,)
    ).fetchone()["content_sha256"]
    conn.close()

    Image.new("RGB", (640, 480), "purple").save(paths[0], "JPEG", quality=97)
    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")

    conn = db.open_db(tmp_path / "inventory.sqlite")
    after = conn.execute(
        "SELECT content_sha256, status FROM features WHERE file_id = ?", (file_id,)
    ).fetchone()
    conn.close()
    assert after["status"] == "done"
    assert after["content_sha256"] != before


def test_deleted_thumbnail_is_regenerated(tmp_path):
    root = tmp_path / "photos"
    _library(root, 2)
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")
    directory = thumbnails.thumbs_dir(tmp_path / "output")
    victim = sorted(directory.glob("*.jpg"))[0]
    victim.unlink()

    again = stage1_features.run(config_path=config, backend_override="stub")
    assert again["thumbnails"]["created"] == 1
    assert victim.is_file()


def test_pixel_size_change_invalidates_the_cache(tmp_path):
    root = tmp_path / "photos"
    _library(root, 2)
    from src import stage0_inventory

    stage0_inventory.run(config_path=_config(tmp_path, root, max_px=48))
    stage1_features.run(config_path=_config(tmp_path, root, max_px=48),
                        backend_override="stub")
    again = stage1_features.run(config_path=_config(tmp_path, root, max_px=96),
                                backend_override="stub")
    assert again["thumbnails"]["created"] == 2
    for thumb in thumbnails.thumbs_dir(tmp_path / "output").glob("*.jpg"):
        with Image.open(thumb) as image:
            assert max(image.size) <= 96


# --- honest failures -------------------------------------------------------

def test_unreadable_source_records_a_failure_and_never_fakes_success(tmp_path):
    root = tmp_path / "photos"
    _library(root, 1)
    broken = root / "IMG_20260109_120000.jpg"
    broken.write_bytes(b"this is not a jpeg")
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stats = stage1_features.run(config_path=config, backend_override="stub")

    assert stats["thumbnails"]["failed"] == 1
    assert stats["thumbnails"]["recorded_failed"] == 1
    conn = db.open_db(tmp_path / "inventory.sqlite")
    failures = db.thumbnail_failures(conn)
    assert len(failures) == 1
    assert failures[0]["basename"] == broken.name
    assert thumbnails.SOURCE_DECODE_FAILED in failures[0]["error"]
    # No thumbnail file was written for the failing photo.
    failed_id = failures[0]["file_id"]
    assert not thumbnails.thumb_path(
        thumbnails.thumbs_dir(tmp_path / "output"), failed_id).is_file()
    conn.close()


def test_write_failure_is_recorded_and_leaves_no_partial_file(tmp_path, monkeypatch):
    thumbnailer = thumbnails.Thumbnailer(tmp_path / "thumbs", max_px=32)
    monkeypatch.setattr(thumbnails, "save_atomic",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    record = thumbnailer.capture({"id": 5, "size_bytes": 1, "mtime_ns": 2},
                                 Image.new("RGB", (100, 100)))
    assert record["status"] == thumbnails.STATUS_ERROR
    assert "disk full" in record["error"]
    assert thumbnailer.created == 0 and thumbnailer.failed == 1
    assert not thumbnails.thumb_path(tmp_path / "thumbs", 5).exists()


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    destination = tmp_path / "thumbs" / "1.jpg"
    written = thumbnails.save_atomic(Image.new("RGB", (20, 20), "red"), destination)
    assert destination.is_file() and written > 0
    assert list((tmp_path / "thumbs").glob("*.tmp")) == []


def test_thumbnails_can_be_disabled(tmp_path):
    root = tmp_path / "photos"
    _library(root, 1)
    cfg = {
        "paths": {"root": str(root), "db": str(tmp_path / "inventory.sqlite"),
                  "output_dir": str(tmp_path / "output"),
                  "models_dir": str(tmp_path / "models")},
        "features": {"backend": "stub", "thumbnails": {"enabled": False}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    from src import stage0_inventory

    stage0_inventory.run(config_path=str(path))
    stats = stage1_features.run(config_path=str(path), backend_override="stub")
    assert stats["thumbnails"]["enabled"] is False
    assert not thumbnails.thumbs_dir(tmp_path / "output").exists()


def test_stable_naming_uses_the_inventory_file_id():
    assert thumbnails.thumb_path("/cache", 12345).name == "12345.jpg"


def test_invalid_thumbnail_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_px"):
        thumbnails.Thumbnailer(tmp_path, max_px=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        thumbnails.Thumbnailer(tmp_path, jpeg_quality=0)


def test_estimate_uses_measured_average_or_reports_unknown():
    assert thumbnails.estimate_total_bytes(100_000, 30_000.0) == 3_000_000_000
    assert thumbnails.estimate_total_bytes(100_000, None) is None


def test_feature_error_rows_stay_terminal_for_features(tmp_path):
    """A decode failure records a thumbnail error without resurrecting features."""
    root = tmp_path / "photos"
    (root).mkdir()
    broken = root / "IMG_20260101_120000.jpg"
    broken.write_bytes(b"nope")
    config = _config(tmp_path, root)
    from src import stage0_inventory

    stage0_inventory.run(config_path=config)
    stage1_features.run(config_path=config, backend_override="stub")
    second = stage1_features.run(config_path=config, backend_override="stub")
    # done_error remains terminal: no infinite retry loop from the thumb clause.
    assert second["processed"] == 0
    conn = db.open_db(tmp_path / "inventory.sqlite")
    meta = json.loads(conn.execute("SELECT quality_meta FROM features").fetchone()[0])
    assert meta["error"] is True
    conn.close()
