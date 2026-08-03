"""Oversize exclusion and bounded IQA are hard Stage-1 resource guards."""

from __future__ import annotations

import json

import numpy as np
import yaml
from PIL import Image

from src import db, feature_admission, stage1_features
from src.config import Config
from src.stage1_backends import StubBackend, _bounded_iqa_image


def _config(tmp_path, db_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"db": str(db_path), "output_dir": str(tmp_path / "out")},
        "features": {
            "backend": "stub", "max_process_megapixels": 64,
            "max_process_aspect_ratio": 3.0,
            "iqa_max_long_edge": 1000, "thumbnails": {"enabled": False},
        },
    }), encoding="utf-8")
    return str(path)


def _row(path, kind="jpg", width=10000, height=7000):
    return {
        "path": str(path), "basename": path.name, "size_bytes": 1, "mtime_ns": 1,
        "width": width, "height": height, "file_kind": kind, "scan_status": "done",
    }


def test_oversize_skips_decode_backend_rerun_and_mp4(tmp_path, monkeypatch):
    db_path = tmp_path / "inventory.sqlite"
    conn = db.open_db(db_path)
    still = db.insert_file(conn, _row(tmp_path / "huge.jpg"))
    video = db.insert_file(conn, _row(tmp_path / "huge.mp4", kind="mp4_only"))
    conn.commit()
    conn.close()
    monkeypatch.setattr(stage1_features, "_open_image_and_sha",
                        lambda *_: (_ for _ in ()).throw(AssertionError("decoded")))
    monkeypatch.setattr(stage1_features, "resolve_backend",
                        lambda *_: (_ for _ in ()).throw(AssertionError("backend loaded")))

    first = stage1_features.run(_config(tmp_path, db_path), backend_override="stub")
    second = stage1_features.run(_config(tmp_path, db_path), backend_override="stub")
    conn = db.open_db(db_path)
    status = conn.execute("SELECT status FROM features WHERE file_id=?", (still,)).fetchone()[0]
    video_features = conn.execute(
        "SELECT COUNT(*) FROM features WHERE file_id=?", (video,)
    ).fetchone()[0]
    conn.close()
    assert first["processed"] == second["processed"] == 0
    assert first["skipped_oversize"] == second["skipped_oversize"] == 1
    assert status == "skipped_oversize"
    assert video_features == 0


def test_previously_done_oversize_is_migrated_out_of_stage2(tmp_path):
    conn = db.open_db(tmp_path / "inventory.sqlite")
    file_id = db.insert_file(conn, _row(tmp_path / "old.jpg"))
    db.batch_insert_features(conn, [{
        "file_id": file_id, "phash": b"12345678", "content_sha256": "old",
        "dinov2_embedding": b"old", "quality_score": 99, "quality_meta": "{}",
        "face_count": 0, "faces_json": "[]", "status": "done",
    }])
    peer = db.insert_file(conn, _row(tmp_path / "peer.jpg", width=4000, height=3000))
    group_id = db.insert_group(
        conn, "burst", peer, [(peer, True, "keep"), (file_id, False, "candidate")], 1
    )
    skipped = feature_admission.mark_oversize_skipped(conn, 64_000_000)
    conn.commit()
    assert [row["id"] for row in skipped] == [file_id]
    assert db.load_features_joined(conn) == []
    assert file_id not in [row["id"] for row in db.iter_files_for_features(conn)]
    assert conn.execute("SELECT COUNT(*) FROM groups WHERE id=?", (group_id,)).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM group_members WHERE group_id=?", (group_id,)
    ).fetchone()[0] == 0
    conn.close()


def test_extreme_aspect_skips_but_three_to_one_boundary_is_kept(tmp_path, monkeypatch):
    db_path = tmp_path / "inventory.sqlite"
    conn = db.open_db(db_path)
    panoramic = db.insert_file(conn, _row(tmp_path / "panorama.jpg", width=12001, height=4000))
    boundary = db.insert_file(conn, _row(tmp_path / "three-to-one.jpg", width=12000, height=4000))
    conn.commit()
    conn.close()

    decoded: list[str] = []

    def fake_decode(path):
        decoded.append(path.name)
        return Image.new("RGB", (12, 4), "gray"), "sha"

    monkeypatch.setattr(stage1_features, "_open_image_and_sha", fake_decode)
    result = stage1_features.run(_config(tmp_path, db_path), backend_override="stub")

    conn = db.open_db(db_path)
    statuses = dict(conn.execute(
        "SELECT file_id, status FROM features WHERE file_id IN (?, ?)",
        (panoramic, boundary),
    ).fetchall())
    conn.close()
    assert result["skipped_aspect_ratio"] == 1
    assert result["skipped_pixel_limit"] == 0
    assert statuses[panoramic] == "skipped_oversize"
    assert statuses[boundary] == "done"
    assert decoded == ["three-to-one.jpg"]


def test_64mp_50mp_and_normal_widescreen_photos_remain_eligible(tmp_path, monkeypatch):
    db_path = tmp_path / "inventory.sqlite"
    conn = db.open_db(db_path)
    expected = {
        "exact-64mp.jpg": (8000, 8000),
        "phone-50mp.jpg": (8192, 6144),
        "wide-16x9.jpg": (7680, 4320),
        "wide-21x9.jpg": (7000, 3000),
    }
    ids = {
        name: db.insert_file(conn, _row(tmp_path / name, width=size[0], height=size[1]))
        for name, size in expected.items()
    }
    conn.commit()
    conn.close()

    decoded: list[str] = []

    def fake_decode(path):
        decoded.append(path.name)
        return Image.new("RGB", (8, 8), "gray"), "sha"

    monkeypatch.setattr(stage1_features, "_open_image_and_sha", fake_decode)
    result = stage1_features.run(_config(tmp_path, db_path), backend_override="stub")

    conn = db.open_db(db_path)
    statuses = dict(conn.execute(
        f"SELECT file_id, status FROM features WHERE file_id IN "
        f"({', '.join('?' for _ in ids)})",
        tuple(ids.values()),
    ).fetchall())
    conn.close()
    assert result["skipped_oversize"] == 0
    assert set(decoded) == set(expected)
    assert set(statuses.values()) == {"done"}


def test_iqa_input_is_aspect_preserving_and_bounded_with_scale_metadata():
    image = Image.new("RGB", (4000, 2000), "gray")
    bounded, scale = _bounded_iqa_image(image, 1000)
    assert bounded.size == (1000, 500)
    assert scale == 0.25
    backend = StubBackend(iqa_max_long_edge=1000)
    _, meta = backend.quality(image)
    assert meta["iqa_input_size"] == [1000, 500]
    assert meta["iqa_scale"] == 0.25
    assert np.isfinite(meta["sharpness"])
    assert json.loads(json.dumps(meta))["iqa_input_size"] == [1000, 500]


def test_stub_resolution_uses_configured_iqa_bound():
    cfg = Config({"features": {"iqa_max_long_edge": 777}})
    backend = stage1_features.resolve_backend(cfg, "stub")
    assert backend.iqa_max_long_edge == 777
