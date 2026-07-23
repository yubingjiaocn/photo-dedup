"""Tests for clustering primitives: DSU, face split, keep-selection, pHash."""

import numpy as np
from PIL import Image

from src import stage2_cluster as C
from src import quality as Q
from src.config import load_config


# --- union-find -------------------------------------------------------------

def test_dsu_union_find():
    d = C.DSU(5)
    d.union(0, 1)
    d.union(1, 2)
    assert d.find(0) == d.find(2)
    assert d.find(0) != d.find(3)


# --- check-in face split ----------------------------------------------------

def _face(x, y, w=20, h=20, score=0.95):
    return {"bbox": [x, y, w, h], "score": score, "landmarks": []}


def test_face_pose_shift_moved():
    fi = [_face(10, 40)]   # center x ~0.2 of a 100px-wide frame
    fj = [_face(80, 40)]   # center x ~0.9
    shift = C.face_pose_shift(fi, 100, 100, fj, 100, 100, 0.6)
    assert shift is not None and shift > 0.6
    assert C.should_split_by_face(shift, 0.30) is True


def test_face_pose_shift_same_spot():
    fi = [_face(40, 40)]
    fj = [_face(42, 41)]
    shift = C.face_pose_shift(fi, 100, 100, fj, 100, 100, 0.6)
    assert shift is not None and shift < 0.30
    assert C.should_split_by_face(shift, 0.30) is False


def test_face_pose_shift_no_face():
    shift = C.face_pose_shift([], 100, 100, [_face(40, 40)], 100, 100, 0.6)
    assert shift is None
    assert C.should_split_by_face(shift, 0.30) is False


def test_low_confidence_face_ignored():
    fi = [_face(10, 40, score=0.2)]
    fj = [_face(80, 40, score=0.2)]
    # both below min_score -> treated as faceless -> no split
    assert C.face_pose_shift(fi, 100, 100, fj, 100, 100, 0.6) is None


# --- keep selection ---------------------------------------------------------

def test_select_keep_prefers_quality():
    cfg = load_config()
    members = [
        {"quality_score": 20.0, "quality_meta": "{}", "width": 4000, "height": 3000, "size_bytes": 100},
        {"quality_score": 80.0, "quality_meta": "{}", "width": 4000, "height": 3000, "size_bytes": 100},
    ]
    keep, scores = C.select_keep(members, cfg)
    assert keep == 1
    assert scores[1] > scores[0]


def test_select_keep_face_quality_counts():
    cfg = load_config()
    members = [
        {"quality_score": 50.0, "quality_meta": '{"face_quality": 0.0}',
         "width": 4000, "height": 3000, "size_bytes": 100},
        {"quality_score": 50.0, "quality_meta": '{"face_quality": 0.9}',
         "width": 4000, "height": 3000, "size_bytes": 100},
    ]
    keep, _ = C.select_keep(members, cfg)
    assert keep == 1  # same IQA, better face wins


# --- exact-dup layer --------------------------------------------------------

def test_layer1_exact_dup_bands():
    h0 = 0x0123456789ABCDEF
    h1 = h0 ^ 0b11        # 2 bits different -> within threshold 2
    h2 = 0xFFFFFFFFFFFFFFFF ^ h0  # maximally different
    d = C.DSU(3)
    edges = C.layer1_exact_dup(d, [h0, h1, h2], threshold=2)
    assert d.find(0) == d.find(1)
    assert d.find(0) != d.find(2)
    assert any(t == "exact_dup" for _, _, t in edges)


# --- pHash / hamming --------------------------------------------------------

def test_phash_identical_zero_distance():
    rng = np.random.default_rng(0)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    img = Image.fromarray(arr)
    h1 = Q.phash_bytes(img)
    h2 = Q.phash_bytes(img.copy())
    assert Q.hamming(h1, h2) == 0


def test_phash_distinct_images_differ():
    rng = np.random.default_rng(1)
    a = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))
    b = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))
    assert Q.hamming(Q.phash_bytes(a), Q.phash_bytes(b)) > 5
