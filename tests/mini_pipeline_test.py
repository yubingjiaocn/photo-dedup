"""End-to-end mini pipeline test on 10 synthetic images.

Scenario (all using the deterministic 'stub' feature backend, no GPU):

  * BURST     -- 3 near-identical frames, seconds apart, no person
                 -> must collapse into exactly ONE group (keep 1, delete 2).
  * CHECK-IN  -- 3 frames of the same background with a "person" (red block)
                 at very different positions, seconds apart
                 -> must NOT group (the face-position guard splits them).
  * INDEPENDENT -- 4 distinct scenes, far apart in time
                 -> must NOT group.

Assertion: stage 2 produces exactly 1 group, and it is the burst group;
stage 3 lists exactly 2 files to delete.

Design notes to keep the synthetic data honest:
  * Backgrounds use LOW red (R < ~60) so the stub face detector (which keys on
    bright-red blocks) never fires on the background -- only on the injected
    "person" block. Burst frames therefore correctly read as faceless.
  * Scenes are smooth low-frequency patterns (not per-pixel noise) so that
    downsampled embeddings + pHash are distinct between scenes and stable
    within a burst.
  * The check-in "person" block is large and moves far, so those frames are
    pHash-distant (never caught by the exact-dup layer) and only reach the
    burst layer, where the face guard protects them.
"""

import numpy as np
import pytest
import yaml
from PIL import Image

from src import db
from src import stage0_inventory as s0
from src import stage1_features as s1
from src import stage2_cluster as s2
from src import stage3_report as s3


# --- synthetic image helpers ------------------------------------------------

def _scene(seed: int, size: int = 256) -> np.ndarray:
    """A smooth, low-frequency, low-red color pattern unique to ``seed``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    a, b, c = rng.uniform(-3.0, 3.0, 3)
    g = np.sin(a * np.pi * xx + b * np.pi * yy + c) + np.cos((a + 1.3) * np.pi * yy + c)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = (g * 45).astype(np.uint8)          # low red -> no false faces
    img[..., 1] = (g * 255).astype(np.uint8)
    img[..., 2] = ((1.0 - g) * 255).astype(np.uint8)
    return img


def _jitter(base: np.ndarray, seed: int, std: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = base.astype(np.float32) + rng.normal(0, std, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _add_person(img: np.ndarray, cx: int, w: int = 60, h: int = 140) -> np.ndarray:
    """Paint a bright-red block (a stand-in person) centered near column cx."""
    out = img.copy()
    x0 = max(0, cx - w // 2)
    x1 = min(img.shape[1], x0 + w)
    y0 = img.shape[0] - h
    out[y0:, x0:x1, 0] = 230
    out[y0:, x0:x1, 1] = 30
    out[y0:, x0:x1, 2] = 30
    return out


def _save(arr: np.ndarray, path):
    Image.fromarray(arr).save(path, format="JPEG", quality=95)


def _build_library(root):
    root.mkdir(parents=True, exist_ok=True)
    # BURST: same base, tiny jitter, 12:00:00 / 05 / 10
    burst_base = _scene(seed=101)
    for i, sec in enumerate((0, 5, 10)):
        _save(_jitter(burst_base, seed=200 + i),
              root / f"IMG_20260101_1200{sec:02d}.jpg")
    # CHECK-IN: same background, person at far-apart positions, 13:00:00 / 05 / 10
    checkin_base = _scene(seed=303)
    for i, (sec, cx) in enumerate(((0, 40), (5, 128), (10, 216))):
        frame = _add_person(_jitter(checkin_base, seed=400 + i), cx=cx)
        _save(frame, root / f"IMG_20260101_1300{sec:02d}.jpg")
    # INDEPENDENT: 4 distinct scenes, far apart in time
    for i, hh in enumerate((14, 15, 16, 17)):
        _save(_scene(seed=500 + i * 37),
              root / f"IMG_20260101_{hh:02d}0000.jpg")


def _write_config(tmp_path, root):
    cfg = {
        "paths": {
            "root": str(root),
            "db": str(tmp_path / "inventory.sqlite"),
            "trash": str(tmp_path / "_trash"),
            "output_dir": str(tmp_path / "output"),
            "models_dir": str(tmp_path / "models"),
        },
        "features": {"backend": "stub", "batch_size": 4},
        "cluster": {
            "burst_window_seconds": 30,
            "dinov2_threshold": 0.92,
            "phash_hamming_threshold": 2,
            "face_pose_shift_ratio": 0.30,
            "min_face_score": 0.6,
            "enable_loose_similar": False,
        },
        "execute": {"mode": "move", "dry_run": True},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return str(cfg_path)


# --- the test ---------------------------------------------------------------

def test_mini_pipeline(tmp_path):
    root = tmp_path / "Photos"
    _build_library(root)
    cfg_path = _write_config(tmp_path, root)

    # Stage 0: inventory
    inv = s0.run(config_path=cfg_path)
    assert inv["files"] == 10
    assert inv["jpg"] == 10  # no motion partners in this fixture

    # Stage 1: features (stub backend)
    feat = s1.run(config_path=cfg_path, backend_override="stub")
    assert feat["processed"] == 10

    # Stage 2: cluster
    stats = s2.run(config_path=cfg_path)
    assert stats["groups"] == 1, f"expected exactly 1 group, got {stats}"
    assert stats["to_delete"] == 2

    # Verify the single group is the BURST group (all members at 12:00:xx).
    conn = db.open_db(tmp_path / "inventory.sqlite")
    groups = list(db.iter_groups(conn))
    assert len(groups) == 1
    members = db.group_members(conn, groups[0]["id"])
    assert len(members) == 3
    assert all(m["basename"].startswith("IMG_20260101_1200") for m in members)
    assert sum(1 for m in members if m["is_keep"]) == 1
    conn.close()

    # Stage 3: report + delete lists
    rep = s3.run(config_path=cfg_path)
    assert rep["groups"] == 1
    assert rep["delete_files"] == 2

    out = tmp_path / "output"
    assert (out / "review.html").exists()
    assert (out / "summary.txt").exists()
    dl = (out / "delete_local.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(dl) == 2
    cloud = (out / "delete_cloud.json").read_text(encoding="utf-8")
    assert cloud.count("filename") == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
