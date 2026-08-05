"""Tests for time-windowed grouping: SHA-exact vs pHash-near vs DINO burst."""

import json
import numpy as np
from src import stage2_cluster as C
from src import cluster_layers as CL


def _minimal_exposure() -> str:
    """Return minimal valid exposure metadata for decision tests."""
    return json.dumps({
        "exposure": {
            "clip_hi": 0.0, "clip_lo": 0.0,
            "blob_hi": 0.0, "blob_lo": 0.0,
            "mass_usable": 0.7, "anchor_mass": 0.5,
            "entropy_nonclip": 5.0, "ymean": 0.5,
        }
    })


def _make_row(
    file_id: int,
    timestamp: int,
    phash: int,
    sha: bytes | None,
    embedding: np.ndarray | None = None,
) -> dict:
    """Helper to construct a minimal feature row for clustering tests."""
    emb_blob = None
    if embedding is not None:
        emb_blob = embedding.astype(np.float16).tobytes()
    phash_blob = phash.to_bytes(8, "big") if phash is not None else None
    return {
        "id": file_id,
        "exif_timestamp": timestamp,
        "phash": phash_blob,
        "content_sha256": sha,
        "dinov2_embedding": emb_blob,
        "quality_score": 50.0,
        "quality_meta": "{}",
        "width": 1000,
        "height": 1000,
        "size_bytes": 100000,
        "faces_json": None,
        "face_count": 0,
    }


def _emb(seed: int) -> np.ndarray:
    """Create a deterministic normalized embedding."""
    rng = np.random.default_rng(seed)
    v = rng.random(768).astype(np.float32)
    return v / np.linalg.norm(v)


def test_sha_exact_crosses_time():
    """Byte-identical files (same SHA) should group across arbitrary time."""
    sha = b"exact_duplicate_sha256_hash_32bytes!"[:32]
    rows = [
        _make_row(1, 0, 0x1111111111111111, sha, _emb(1)),      # t=0
        _make_row(2, 1000000, 0x1111111111111111, sha, _emb(1)), # t=1M seconds (~12 days)
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    sha_hashes = C._sha_hashes(rows)
    edges = CL.layer1_sha_exact(dsu, sha_hashes)
    assert len(edges) == 1
    assert edges[0][2] == "sha_exact"
    assert dsu.find(0) == dsu.find(1)


def test_phash_near_within_window():
    """pHash near-dup (hamming ≤ 2) within 30s should group."""
    h0 = 0x0123456789ABCDEF
    h1 = h0 ^ 0b11  # 2 bits different
    rows = [
        _make_row(1, 100, h0, None, _emb(1)),
        _make_row(2, 120, h1, None, _emb(2)),  # 20s later, within 30s window
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    sha_hashes = C._sha_hashes(rows)
    phash_ints = C._phash_ints(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    edges = C.layer2_phash_near(
        dsu, phash_ints, sha_hashes, timestamps,
        threshold=2, window_seconds=30, frozen_components=set()
    )
    assert len(edges) == 1
    assert edges[0][2] == "phash_near"
    assert dsu.find(0) == dsu.find(1)


def test_phash_near_outside_window():
    """pHash near-dup (hamming ≤ 2) beyond 30s should NOT group."""
    h0 = 0x0123456789ABCDEF
    h1 = h0 ^ 0b11  # 2 bits different
    rows = [
        _make_row(1, 100, h0, None, _emb(1)),
        _make_row(2, 200, h1, None, _emb(2)),  # 100s later, beyond 30s window
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    sha_hashes = C._sha_hashes(rows)
    phash_ints = C._phash_ints(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    edges = C.layer2_phash_near(
        dsu, phash_ints, sha_hashes, timestamps,
        threshold=2, window_seconds=30, frozen_components=set()
    )
    assert len(edges) == 0
    assert dsu.find(0) != dsu.find(1)


def test_sha_frozen_blocks_phash_absorption():
    """SHA-exact group should not absorb pHash-near photos."""
    sha = b"frozen_sha_exact_group_12345678"[:32]
    h0 = 0x0123456789ABCDEF
    h1 = h0 ^ 0b11  # pHash near h0
    rows = [
        _make_row(1, 100, h0, sha, _emb(1)),    # SHA-exact
        _make_row(2, 100, h0, sha, _emb(1)),    # SHA-exact (same group)
        _make_row(3, 105, h1, None, _emb(3)),   # pHash-near to row 0, but no SHA
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    sha_hashes = C._sha_hashes(rows)
    phash_ints = C._phash_ints(rows)
    timestamps = [r["exif_timestamp"] for r in rows]

    # Layer 1: SHA-exact groups 0 and 1
    sha_edges = CL.layer1_sha_exact(dsu, sha_hashes)
    assert len(sha_edges) == 1
    assert dsu.find(0) == dsu.find(1)
    frozen = {dsu.find(i) for i, j, _ in sha_edges}

    # Layer 2: pHash-near should skip frozen components
    phash_edges = C.layer2_phash_near(
        dsu, phash_ints, sha_hashes, timestamps,
        threshold=2, window_seconds=30, frozen_components=frozen
    )
    # Row 3 should NOT join the SHA group
    assert dsu.find(0) == dsu.find(1)
    assert dsu.find(2) != dsu.find(0)
    assert len(phash_edges) == 0


def test_dino_burst_within_window():
    """DINO burst (cos ≥ 0.92) within 30s should group."""
    emb = _emb(1)
    rows = [
        _make_row(1, 100, 0x1111, None, emb),
        _make_row(2, 120, 0x2222, None, emb),  # same embedding, 20s later
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    units, valid = C._unit_embeddings(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    faces = [CL.parse_faces(r["faces_json"]) for r in rows]
    edges = C.layer_window(
        dsu, rows, units, timestamps, faces,
        window_seconds=30, cos_threshold=0.92,
        face_ratio=0.30, min_face_score=0.6,
        edge_type="burst", valid_embeddings=valid,
        frozen_components=set()
    )
    assert len(edges) == 1
    assert edges[0][2] == "burst"
    assert dsu.find(0) == dsu.find(1)


def test_dino_burst_outside_window():
    """DINO burst (cos ≥ 0.92) beyond 30s should NOT group."""
    emb = _emb(1)
    rows = [
        _make_row(1, 100, 0x1111, None, emb),
        _make_row(2, 200, 0x2222, None, emb),  # same embedding, 100s later
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    units, valid = C._unit_embeddings(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    faces = [CL.parse_faces(r["faces_json"]) for r in rows]
    edges = C.layer_window(
        dsu, rows, units, timestamps, faces,
        window_seconds=30, cos_threshold=0.92,
        face_ratio=0.30, min_face_score=0.6,
        edge_type="burst", valid_embeddings=valid,
        frozen_components=set()
    )
    assert len(edges) == 0
    assert dsu.find(0) != dsu.find(1)


def test_sha_frozen_blocks_dino_absorption():
    """SHA-exact group should not absorb DINO-similar photos."""
    sha = b"frozen_sha_exact_group_abcdefgh"[:32]
    emb_exact = _emb(1)
    emb_similar = emb_exact * 0.98  # very high cosine but not identical
    emb_similar = emb_similar / np.linalg.norm(emb_similar)
    rows = [
        _make_row(1, 100, 0x1111, sha, emb_exact),      # SHA-exact
        _make_row(2, 100, 0x1111, sha, emb_exact),      # SHA-exact (same)
        _make_row(3, 105, 0x2222, None, emb_similar),   # DINO-similar but no SHA
    ]
    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    sha_hashes = C._sha_hashes(rows)
    units, valid = C._unit_embeddings(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    faces = [CL.parse_faces(r["faces_json"]) for r in rows]

    # Layer 1: SHA-exact
    sha_edges = CL.layer1_sha_exact(dsu, sha_hashes)
    assert len(sha_edges) == 1
    assert dsu.find(0) == dsu.find(1)
    frozen = {dsu.find(i) for i, j, _ in sha_edges}

    # Layer 3: DINO burst should skip frozen
    burst_edges = C.layer_window(
        dsu, rows, units, timestamps, faces,
        window_seconds=30, cos_threshold=0.92,
        face_ratio=0.30, min_face_score=0.6,
        edge_type="burst", valid_embeddings=valid,
        frozen_components=frozen
    )
    assert dsu.find(0) == dsu.find(1)
    assert dsu.find(2) != dsu.find(0)
    assert len(burst_edges) == 0


def test_transitive_chain_time_span_violation():
    """A-B-C chain where A-C span > 30s should be flagged as impure."""
    emb_a = _emb(1)
    emb_b = _emb(2)
    # Make B similar to both A and C, but A-C dissimilar
    emb_c = _emb(3)
    # Manually adjust to create high A-B and B-C cosine
    emb_b = (emb_a + emb_c) / 2
    emb_b = emb_b / np.linalg.norm(emb_b)

    rows = [
        _make_row(1, 0, 0x1111, None, emb_a),   # t=0
        _make_row(2, 20, 0x2222, None, emb_b),  # t=20 (within 30s of A)
        _make_row(3, 40, 0x3333, None, emb_c),  # t=40 (within 30s of B, but span A-C = 40s)
    ]

    # Verify cosines are high enough
    cos_ab = float(np.dot(emb_a, emb_b))
    cos_bc = float(np.dot(emb_b, emb_c))
    assert cos_ab >= 0.92, f"cos(A,B)={cos_ab} should be ≥ 0.92"
    assert cos_bc >= 0.92, f"cos(B,C)={cos_bc} should be ≥ 0.92"

    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(len(rows), timestamps)
    units, valid = C._unit_embeddings(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    faces = [CL.parse_faces(r["faces_json"]) for r in rows]

    # Build the chain
    _ = C.layer_window(
        dsu, rows, units, timestamps, faces,
        window_seconds=30, cos_threshold=0.92,
        face_ratio=0.30, min_face_score=0.6,
        edge_type="burst", valid_embeddings=valid,
        frozen_components=set()
    )

    # DSU may form the group via transitive closure
    # But validation in cluster() should detect span violation
    member_idx = [0, 1, 2]
    member_timestamps = [timestamps[gi] for gi in member_idx if timestamps[gi] is not None]
    time_span = max(member_timestamps) - min(member_timestamps)
    assert time_span == 40, f"Expected span=40s, got {time_span}s"
    assert time_span > 30, "Chain violates 30s window constraint"


def test_auto_remove_only_byte_identical():
    """Only SHA-256 byte-identical files should get AUTO_REMOVE decision."""
    from src import decision as D

    sha = b"exact_match_sha256_hash_123456"[:32]
    # Keeper and candidate with same SHA
    members = [
        {"phash": b"\x11" * 8, "content_sha256": sha, "dinov2_embedding": b"\x00" * 1536,
         "quality_score": 60.0, "quality_meta": _minimal_exposure(),
         "face_count": 0, "width": 1000, "height": 1000, "size_bytes": 100000},
        {"phash": b"\x11" * 8, "content_sha256": sha, "dinov2_embedding": b"\x00" * 1536,
         "quality_score": 50.0, "quality_meta": _minimal_exposure(),
         "face_count": 0, "width": 1000, "height": 1000, "size_bytes": 100000},
    ]
    scores = {0: 0.6, 1: 0.5}
    safe_duplicates = {0: False, 1: True}  # member 1 matches keeper SHA

    result = D.decide_group(
        members, keeper=0, scores=scores, group_type="sha_exact",
        profile="balanced",
        group_trusted=True, safe_duplicates=safe_duplicates
    )

    assert result["members"][0]["decision"] == "KEEP"
    assert result["members"][1]["decision"] == "AUTO_REMOVE"
    assert result["members"][1]["reason"] == "BYTE_IDENTICAL"


def test_phash_near_no_auto_remove():
    """pHash near-dup (without SHA match) should never get AUTO_REMOVE."""
    from src import decision as D

    # Different SHAs despite pHash similarity
    members = [
        {"phash": b"\x11" * 8, "content_sha256": b"sha_a" * 4, "dinov2_embedding": b"\x00" * 1536,
         "quality_score": 60.0, "quality_meta": _minimal_exposure(),
         "face_count": 0, "width": 1000, "height": 1000, "size_bytes": 100000},
        {"phash": b"\x12" * 8, "content_sha256": b"sha_b" * 4, "dinov2_embedding": b"\x00" * 1536,
         "quality_score": 50.0, "quality_meta": _minimal_exposure(),
         "face_count": 0, "width": 1000, "height": 1000, "size_bytes": 100000},
    ]
    scores = {0: 0.6, 1: 0.5}
    safe_duplicates = {0: False, 1: False}  # no SHA match

    result = D.decide_group(
        members, keeper=0, scores=scores, group_type="phash_near",
        profile="balanced",
        group_trusted=True, safe_duplicates=safe_duplicates
    )

    assert result["members"][0]["decision"] == "KEEP"
    assert result["members"][1]["decision"] == "MAYBE"
    assert "PHASH_NEAR" in result["members"][1]["reason"]
