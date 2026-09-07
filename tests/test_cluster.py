"""Tests for clustering primitives: DSU, face split, keep-selection, pHash."""

import numpy as np
from PIL import Image

from src import stage2_cluster as C
from src import cluster_layers as CL
from src import quality as Q
from src.config import load_config


# --- union-find -------------------------------------------------------------

def test_dsu_union_find():
    d = CL.DSU(5)
    d.union(0, 1)
    d.union(1, 2)
    assert d.find(0) == d.find(2)
    assert d.find(0) != d.find(3)


def test_dsu_span_tracking():
    timestamps = [100, 120, 140, 200, 220]
    d = CL.DSU(5, timestamps)
    d.union(0, 1)  # 100-120 = 20s
    d.union(1, 2)  # 100-140 = 40s
    assert d.span(0) == 40
    assert d.span(3) == 0  # singleton


def test_dsu_union_if_span_within_accepts():
    timestamps = [100, 120, 140]
    d = CL.DSU(3, timestamps)
    # Merge 0(100) and 1(120) -> span 20s <= 30s window
    assert d.union_if_span_within(0, 1, 30) is True
    assert d.find(0) == d.find(1)


def test_dsu_union_if_span_within_rejects():
    timestamps = [100, 120, 150]
    d = CL.DSU(3, timestamps)
    d.union(0, 1)  # 100-120 = 20s
    # Attempting 0(100-120) + 2(150) -> span 50s > 30s window
    assert d.union_if_span_within(0, 2, 30) is False
    assert d.find(0) != d.find(2)


def test_dsu_none_timestamp_rejected():
    """None timestamp components cannot merge with span-enforced union.

    union_if_span_within must reject any merge involving None timestamps,
    preventing visual layers from grouping photos without capture time.
    """
    timestamps = [100, None, 140]
    d = CL.DSU(3, timestamps)
    # CRITICAL: Cannot merge components when any member has None timestamp
    assert d.union_if_span_within(0, 1, 30) is False, "Should reject 100 + None"
    assert d.union_if_span_within(1, 2, 30) is False, "Should reject None + 140"
    # All three remain separate components
    assert d.find(0) != d.find(1)
    assert d.find(1) != d.find(2)
    # Standard union (for SHA-exact layer) still works
    d.union(0, 2)
    assert d.find(0) == d.find(2)
    assert d.span(0) == 40  # 140 - 100


# --- check-in face split ----------------------------------------------------

def _face(x, y, w=20, h=20, score=0.95):
    return {"bbox": [x, y, w, h], "score": score, "landmarks": []}


def test_face_pose_shift_moved():
    fi = [_face(10, 40)]
    fj = [_face(80, 40)]
    shift = CL.face_pose_shift(fi, 100, 100, fj, 100, 100, 0.6)
    assert shift is not None and shift > 0.6
    assert CL.should_split_by_face(shift, 0.30) is True


def test_face_pose_shift_same_spot():
    fi = [_face(40, 40)]
    fj = [_face(42, 41)]
    shift = CL.face_pose_shift(fi, 100, 100, fj, 100, 100, 0.6)
    assert shift is not None and shift < 0.30
    assert CL.should_split_by_face(shift, 0.30) is False


def test_face_pose_shift_no_face():
    shift = CL.face_pose_shift([], 100, 100, [_face(40, 40)], 100, 100, 0.6)
    assert shift is None
    assert CL.should_split_by_face(shift, 0.30) is False


def test_low_confidence_face_ignored():
    fi = [_face(10, 40, score=0.2)]
    fj = [_face(80, 40, score=0.2)]
    assert CL.face_pose_shift(fi, 100, 100, fj, 100, 100, 0.6) is None


def _identity_face(values, *, width=160, score=0.95):
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return {
        "bbox": [10, 10, width, 180], "score": score,
        "identity_embedding": vector.astype(np.float16).tobytes().hex(),
    }


def test_identity_gate_accepts_same_anonymous_subject():
    assert CL.identity_compatible(
        [_identity_face([1, 0, 0])], [_identity_face([0.99, 0.01, 0])],
        0.6, 80, 0.45,
    ) is True


def test_identity_gate_rejects_different_subject():
    assert CL.identity_compatible(
        [_identity_face([1, 0, 0])], [_identity_face([0, 1, 0])],
        0.6, 80, 0.45,
    ) is False


def test_identity_gate_fails_closed_when_face_evidence_missing_or_small():
    assert CL.identity_compatible([_face(10, 10)], [_face(10, 10)], 0.6, 80, 0.45) is False
    assert CL.identity_compatible(
        [_identity_face([1, 0], width=40)], [_identity_face([1, 0], width=40)],
        0.6, 80, 0.45,
    ) is False
    assert CL.identity_compatible([], [], 0.6, 80, 0.45) is True


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

def test_layer1_sha_exact():
    """SHA-256 byte-exact grouping with star topology."""
    sha_a = b"exact_duplicate_hash_a_123456"[:32]
    sha_b = b"different_hash_b_7890abcdefgh"[:32]
    timestamps = [100, 120, 140]
    d = CL.DSU(3, timestamps)
    edges = CL.layer1_sha_exact(d, [sha_a, sha_a, sha_b])
    assert d.find(0) == d.find(1)
    assert d.find(0) != d.find(2)
    assert len(edges) == 1  # star: k-1 edges for k=2 members
    assert any(t == "sha_exact" for _, _, t in edges)


def test_layer1_sha_star_topology():
    """SHA exact should use star (k-1 edges), not full mesh (k*(k-1)/2)."""
    sha_a = b"a" * 32
    timestamps = [100, 110, 120, 130, 140]
    d = CL.DSU(5, timestamps)
    edges = CL.layer1_sha_exact(d, [sha_a] * 5)
    assert len(edges) == 4  # k-1 = 4, not 10
    # All should be same component
    root = d.find(0)
    assert all(d.find(i) == root for i in range(5))


def test_layer2_phash_near_windowed():
    """pHash near-dup with time window and span enforcement."""
    h0 = 0x0123456789ABCDEF
    h1 = h0 ^ 0b11
    h2 = 0xFFFFFFFFFFFFFFFF ^ h0
    timestamps = [100, 120, 140]
    d = CL.DSU(3, timestamps)
    edges = C.layer2_phash_near(
        d, [h0, h1, h2], [None, None, None], timestamps,
        threshold=2, window_seconds=30, frozen_components=set()
    )
    assert d.find(0) == d.find(1)
    assert d.find(0) != d.find(2)
    assert any(t == "phash_near" for _, _, t in edges)


def test_layer2_phash_prevents_chain_violation():
    """pHash with span enforcement: A(0)-B(20)-C(40) chain split by 30s window."""
    h0 = 0x1111111111111111
    h1 = h0 ^ 0b11  # within threshold to h0
    h2 = h1 ^ 0b01  # within threshold to h1, but would create A-B-C chain
    timestamps = [100, 120, 140]
    d = CL.DSU(3, timestamps)
    C.layer2_phash_near(
        d, [h0, h1, h2], [None, None, None], timestamps,
        threshold=2, window_seconds=30, frozen_components=set()
    )
    # DSU should refuse merge that would exceed window -> split into multiple components
    components = {d.find(i) for i in range(3)}
    assert len(components) > 1  # Not all same component
    # Verify each component's span <= 30s
    for root in set(d.find(i) for i in range(3)):
        assert d.span(root) <= 30


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


# --- integration tests ------------------------------------------------------

def test_integration_span_enforcement():
    """Integration: span enforcement prevents cross-window pHash/DINO grouping.

    Constructs real A(0)-B(20)-C(40) transitive chain where:
    - A-B pairwise distance is within threshold
    - B-C pairwise distance is within threshold
    - BUT A-C would violate 30s window -> must split into multiple components
    - Each persisted group's span MUST be <= configured window (30s)
    """
    import tempfile
    from pathlib import Path
    from src import db

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        conn = db.open_db(db_path)
        cfg = load_config()

        t0 = 1000000
        t1 = t0 + 20
        t2 = t0 + 40
        # Construct transitive chain: A-B within 2, B-C within 2, but A-C > 30s span
        h_a = 0x1111111111111111
        h_b = h_a ^ 0b11      # hamming=2 to A
        h_c = h_b ^ 0b01      # hamming=1 to B, but A-B-C span=40s > 30s

        conn.execute(
            "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
            ("/a.jpg", t0, 100, 1920, 1080, "jpg"),
        )
        conn.execute(
            "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
            ("/b.jpg", t1, 100, 1920, 1080, "jpg"),
        )
        conn.execute(
            "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
            ("/c.jpg", t2, 100, 1920, 1080, "jpg"),
        )
        conn.commit()

        # Production format: float16 embeddings. Identical normalized DINO for all -> would group if span OK
        emb = np.random.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_f16 = emb.astype(np.float16)
        emb_bytes = emb_f16.tobytes()
        meta = '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}'
        for fid, ph, sha in [(1, h_a, "a"*64), (2, h_b, "b"*64), (3, h_c, "c"*64)]:
            conn.execute(
                "INSERT INTO features (file_id, phash, content_sha256, dinov2_embedding, quality_score, quality_meta, faces_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fid, ph.to_bytes(8, "big"), sha, emb_bytes, 50.0, meta, "[]", "done"),
            )
        conn.commit()

        # Run clustering: DINO layer must actually participate
        C.cluster(conn, cfg, scope=None)

        # Verify: A-B-C NOT all in same group (span constraint enforcement)
        rows = conn.execute("SELECT DISTINCT id FROM groups").fetchall()
        assert len(rows) > 0, "Should have created at least one group"

        group_map = {}
        for fid in [1, 2, 3]:
            gid_row = conn.execute(
                "SELECT group_id FROM group_members WHERE file_id = ?", (fid,)
            ).fetchone()
            if gid_row:
                group_map[fid] = gid_row[0]

        # Must NOT all be in same group
        if len(group_map) == 3:
            assert not (
                group_map[1] == group_map[2] == group_map[3]
            ), "A(0s)-B(20s)-C(40s) should NOT all group together (40s > 30s window)"

        # CRITICAL: every persisted group's span must be <= window for non-SHA components
        window = int(cfg.cluster.get("burst_window_seconds", 30))
        for grp_row in conn.execute("SELECT id, group_type FROM groups").fetchall():
            gid = grp_row["id"]
            gtype = grp_row["group_type"]
            if gtype == "sha_exact":
                continue  # SHA-exact has no time constraint
            # Get timestamps of all members in this group
            member_timestamps = [
                r["exif_timestamp"] for r in conn.execute(
                    "SELECT f.exif_timestamp FROM group_members gm "
                    "JOIN files f ON f.id = gm.file_id WHERE gm.group_id = ?", (gid,)
                ).fetchall() if r["exif_timestamp"] is not None
            ]
            if len(member_timestamps) >= 2:
                span = max(member_timestamps) - min(member_timestamps)
                assert span <= window, (
                    f"Group {gid} type={gtype} has span={span}s > window={window}s. "
                    "Span enforcement failed to prevent transitive chain violation."
                )

        conn.close()


def test_integration_sha_cross_week_frozen():
    """Integration: SHA-exact groups span arbitrary time and stay frozen.

    Byte-identical files (SHA-256 match) group regardless of time gap.
    SHA groups are frozen after Layer 1 and never absorb additional photos
    via visual similarity in later layers.
    """
    import tempfile
    from pathlib import Path
    from src import db

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        conn = db.open_db(db_path)
        cfg = load_config()

        week1 = 1000000
        week2 = week1 + 7 * 86400
        sha_same = "a" * 64
        h = 0xAAAAAAAAAAAAAAAA

        for ts, path in [(week1, "/x.jpg"), (week2, "/y.jpg")]:
            conn.execute(
                "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
                (path, ts, 100, 1920, 1080, "jpg"),
            )
        conn.commit()

        # Production format: float16
        emb = np.random.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_f16 = emb.astype(np.float16)
        emb_bytes = emb_f16.tobytes()
        meta = '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}'
        for fid in [1, 2]:
            conn.execute(
                "INSERT INTO features (file_id, phash, content_sha256, dinov2_embedding, quality_score, quality_meta, faces_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fid, h.to_bytes(8, "big"), sha_same, emb_bytes, 50.0, meta, "[]", "done"),
            )
        conn.commit()

        C.cluster(conn, cfg, scope=None)

        g1 = conn.execute(
            "SELECT group_id FROM group_members WHERE file_id = 1"
        ).fetchone()
        g2 = conn.execute(
            "SELECT group_id FROM group_members WHERE file_id = 2"
        ).fetchone()
        assert g1 is not None and g2 is not None, "Both files should be in groups"
        assert g1[0] == g2[0], "SHA-exact files should be grouped despite week gap"

        gtype = conn.execute(
            "SELECT group_type FROM groups WHERE id = ?", (g1[0],)
        ).fetchone()
        assert gtype[0] == "sha_exact", f"Expected sha_exact group, got {gtype[0]}"

        conn.close()


def test_integration_rerun_idempotent():
    """Integration: re-running cluster produces semantically identical groups.

    Group IDs may change across runs (auto-increment), but the semantic
    grouping (which file_ids belong together) must be stable. Also verifies
    that persisted decision/group_type/member reasons remain consistent.
    """
    import tempfile
    from pathlib import Path
    from src import db

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        conn = db.open_db(db_path)
        cfg = load_config()

        sha_a = "a" * 64
        ts = 1000000
        h = 0x1111111111111111

        conn.execute(
            "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
            ("/a.jpg", ts, 100, 1920, 1080, "jpg"),
        )
        conn.execute(
            "INSERT INTO files (path, exif_timestamp, size_bytes, width, height, file_kind) VALUES (?, ?, ?, ?, ?, ?)",
            ("/b.jpg", ts + 10, 100, 1920, 1080, "jpg"),
        )
        conn.commit()

        # Production format: float16
        emb = np.random.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_f16 = emb.astype(np.float16)
        emb_bytes = emb_f16.tobytes()
        meta = '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}'
        for fid in [1, 2]:
            conn.execute(
                "INSERT INTO features (file_id, phash, content_sha256, dinov2_embedding, quality_score, quality_meta, faces_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fid, h.to_bytes(8, "big"), sha_a, emb_bytes, 50.0, meta, "[]", "done"),
            )
        conn.commit()

        # First run
        C.cluster(conn, cfg, scope=None)
        groups1_semantic = set()
        decision1 = {}
        for row in conn.execute(
            "SELECT gm.file_id, gm.group_id, gm.decision, gm.reason, g.group_type "
            "FROM group_members gm JOIN groups g ON g.id = gm.group_id ORDER BY gm.file_id"
        ).fetchall():
            # Record semantic grouping
            groups1_semantic.add(row["file_id"])
            decision1[row["file_id"]] = (row["decision"], row["reason"], row["group_type"])

        # Second run
        C.cluster(conn, cfg, scope=None)
        groups2_semantic = set()
        decision2 = {}
        for row in conn.execute(
            "SELECT gm.file_id, gm.group_id, gm.decision, gm.reason, g.group_type "
            "FROM group_members gm JOIN groups g ON g.id = gm.group_id ORDER BY gm.file_id"
        ).fetchall():
            groups2_semantic.add(row["file_id"])
            decision2[row["file_id"]] = (row["decision"], row["reason"], row["group_type"])

        # Verify semantic stability: same files grouped
        assert groups1_semantic == groups2_semantic, "File grouping changed across runs"
        # Verify decision stability: same decisions/reasons
        assert decision1 == decision2, "Decisions/reasons changed across runs"

        conn.close()
