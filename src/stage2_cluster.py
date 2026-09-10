"""Stage 2 -- clustering (CPU-only, fast, re-runnable).

Reads the ``features`` table and forms groups of duplicate / near-duplicate
still images, choosing one keeper per group. Tweaking a threshold means
re-running *only* this stage -- no images are read.

Three layers, applied in priority order (a file lands in exactly one group):

  Layer 1  sha_exact       byte-identical SHA-256            (any time, frozen after Layer 1)
  Layer 2  phash_near      pHash hamming <= 2, <= 30s apart (visual window)
  Layer 3  burst           <= 30s apart AND DINOv2 cos >= t  (+ check-in split)
  Layer 4  similar_scene   longer window, stricter cos       (off by default)

Time constraints
----------------
- **Byte-identical** (SHA-256 match): no time restriction. These are true duplicates
  and may span arbitrary time if copied/synced across dates. SHA groups are frozen
  after Layer 1 and never absorb additional photos via visual similarity.
- **pHash near-duplicates** (hamming ≤ 2 but SHA mismatch): 30-second window. Without
  byte identity these are visual approximations that must stay temporally local.
- **DINO bursts** and **similar_scene**: existing 30s/300s windows continue to apply.
- All visual groups (non-SHA) enforce: max(timestamp) - min(timestamp) ≤ window,
  blocking A-B-C transitive chains that would violate the span constraint.

Check-in split
--------------
The hard case: a landmark/building is the subject and a person re-poses in
front of it. Such frames are visually near-identical (high DINOv2 cos) but are
NOT duplicates. If a candidate burst pair both contain a face and the dominant
face center moves more than ``face_pose_shift_ratio`` of the frame, we refuse
to union them -- so a run of check-in shots stays as separate keepers.

Efficiency
----------
SHA-exact uses hash bucketing. pHash-near uses time-sorted sliding window with
shared-band prefilter: split the 64-bit pHash into four 16-bit bands; for hamming
<= 2 (pigeonhole) two colliding items must share >=2 bands, so we skip pairs with
fewer than 2 shared bands before computing full Hamming. This is O(K²) per window
where K is the burst size, not a full O(N²) scan. Burst uses time sliding window,
so comparisons stay local.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, load_config
from . import db
from . import root_scope
from . import quality as Q
from . import decision as D
from . import cluster_layers as CL
from . import phase_selection as PS

PRIORITY = {"sha_exact": 4, "phash_near": 3, "burst": 2, "similar_scene": 1}


def _face_quality_of(meta_json: Optional[str]) -> float:
    if not meta_json:
        return 0.0
    try:
        meta = json.loads(meta_json)
        return float(meta.get("face_quality") or 0.0)
    except (ValueError, TypeError):
        return 0.0


def select_keep(members: Sequence[Dict[str, Any]], cfg: Config) -> Tuple[int, Dict[int, float]]:
    """Return (index_of_keeper, {index: score}) for a group's member dicts."""
    w = cfg.quality
    scores: Dict[int, float] = {}
    for idx, m in enumerate(members):
        rb = Q.resolution_bonus(m.get("width"), m.get("height"),
                                float(w.get("resolution_ref_mp", 12.0)))
        s = Q.keep_score(
            m.get("quality_score"),
            _face_quality_of(m.get("quality_meta")),
            rb,
            weight_iqa=float(w.get("weight_iqa", 0.6)),
            weight_face=float(w.get("weight_face", 0.3)),
            weight_resolution=float(w.get("weight_resolution", 0.1)),
        )
        scores[idx] = s
    keep = max(range(len(members)),
               key=lambda i: (scores[i], members[i].get("size_bytes") or 0))
    return keep, scores


def _sha_hashes(rows: Sequence[Any]) -> List[Optional[str]]:
    """Extract SHA-256 hashes, keeping None for missing hashes."""
    return [r["content_sha256"] if r["content_sha256"] else None for r in rows]


def _phash_ints(rows: Sequence[Any]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for r in rows:
        blob = r["phash"]
        out.append(int.from_bytes(blob, "big") if blob else None)
    return out


def layer2_phash_near(
    dsu: CL.DSU,
    phash_ints: Sequence[Optional[int]],
    sha_hashes: Sequence[Optional[str]],
    timestamps: Sequence[Optional[int]],
    threshold: int,
    window_seconds: int,
    frozen_components: set[int],
    faces: Optional[Sequence[List[Dict[str, Any]]]] = None,
    require_face_identity: bool = False,
    identity_cosine_threshold: float = 0.45,
    identity_min_width_px: float = 80.0,
    min_face_score: float = 0.6,
) -> List[Tuple[int, int, str]]:
    """Union files whose pHash hamming <= threshold AND within window, enforcing span.

    Skips pairs where:
    - Either file is in a frozen SHA-exact component
    - Either has the same SHA-256 (covered by layer1_sha_exact)
    - Merging would violate span constraint
    """
    if threshold < 0:
        raise ValueError(f"phash_hamming_threshold={threshold} must be >= 0")
    if threshold > 2:
        raise ValueError(f"phash_hamming_threshold={threshold} > 2 not supported by current band algorithm")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds={window_seconds} must be > 0")
    edges: List[Tuple[int, int, str]] = []
    n = len(phash_ints)
    bands = 4
    band_bits = 16
    mask = (1 << band_bits) - 1

    for i in range(n):
        if phash_ints[i] is None or timestamps[i] is None:
            continue
        if dsu.find(i) in frozen_components:
            continue
        ti = timestamps[i]

        j = i + 1
        while j < n and timestamps[j] is not None and (timestamps[j] - ti) <= window_seconds:
            if phash_ints[j] is None:
                j += 1
                continue
            if dsu.find(j) in frozen_components:
                j += 1
                continue
            if sha_hashes[i] is not None and sha_hashes[i] == sha_hashes[j]:
                j += 1
                continue
            if dsu.find(i) == dsu.find(j):
                j += 1
                continue

            hi, hj = phash_ints[i], phash_ints[j]
            shared_bands = sum(
                1 for band in range(bands)
                if ((hi >> (band * band_bits)) & mask) == ((hj >> (band * band_bits)) & mask)
            )
            if shared_bands >= 2:
                if bin(hi ^ hj).count("1") <= threshold:
                    if require_face_identity and faces is not None and not CL.identity_compatible(
                        faces[i], faces[j], min_face_score, identity_min_width_px,
                        identity_cosine_threshold,
                    ):
                        j += 1
                        continue
                    if dsu.union_if_span_within(i, j, window_seconds):
                        edges.append((i, j, "phash_near"))
            j += 1
    return edges


def _unit_embeddings(rows: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    embs = np.zeros((len(rows), 768), dtype=np.float32)
    valid = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        blob = r["dinov2_embedding"]
        if blob:
            try:
                v = Q.blob_to_embedding(blob)
            except (ValueError, TypeError):
                continue
            if v.shape == (768,) and np.all(np.isfinite(v)) and np.linalg.norm(v) > 0:
                embs[i] = v
                valid[i] = True
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms, valid


def layer_window(
    dsu: CL.DSU,
    rows: Sequence[Any],
    units: np.ndarray,
    timestamps: Sequence[int],
    faces: Sequence[List[Dict[str, Any]]],
    *,
    window_seconds: int,
    cos_threshold: float,
    face_ratio: float,
    min_face_score: float,
    edge_type: str,
    valid_embeddings: Optional[np.ndarray] = None,
    frozen_components: Optional[set[int]] = None,
    require_face_identity: bool = False,
    identity_cosine_threshold: float = 0.45,
    identity_min_width_px: float = 80.0,
) -> List[Tuple[int, int, str]]:
    """Time-windowed similarity union with span enforcement and face split."""
    if window_seconds <= 0:
        raise ValueError(f"window_seconds={window_seconds} must be > 0")
    edges: List[Tuple[int, int, str]] = []
    frozen = frozen_components or set()
    n = len(rows)
    for i in range(n):
        ti = timestamps[i]
        if ti is None or (valid_embeddings is not None and not valid_embeddings[i]):
            continue
        if dsu.find(i) in frozen:
            continue
        j = i + 1
        while j < n and timestamps[j] is not None and (timestamps[j] - ti) <= window_seconds:
            if (valid_embeddings is not None and not valid_embeddings[j]):
                j += 1
                continue
            if dsu.find(j) in frozen:
                j += 1
                continue
            if dsu.find(i) != dsu.find(j):
                cos = float(np.dot(units[i], units[j]))
                if cos >= cos_threshold:
                    if require_face_identity and not CL.identity_compatible(
                        faces[i], faces[j], min_face_score, identity_min_width_px,
                        identity_cosine_threshold,
                    ):
                        j += 1
                        continue
                    shift = CL.face_pose_shift(
                        faces[i], rows[i]["width"], rows[i]["height"],
                        faces[j], rows[j]["width"], rows[j]["height"],
                        min_face_score,
                    )
                    if not CL.should_split_by_face(shift, face_ratio):
                        if dsu.union_if_span_within(i, j, window_seconds):
                            edges.append((i, j, edge_type))
            j += 1
    return edges


def cluster(conn, cfg: Config, scope: Any = None) -> Dict[str, int]:
    """Run all enabled clustering layers and persist groups. Returns stats."""
    rows = db.load_features_joined(conn, scope=scope)
    n = len(rows)
    if n == 0:
        print("[stage2] no features found -- run stage1 first.")
        return {"groups": 0, "files": 0}

    timestamps = [r["exif_timestamp"] for r in rows]
    dsu = CL.DSU(n, timestamps)
    phash_ints = _phash_ints(rows)
    sha_hashes = _sha_hashes(rows)
    units, valid_embeddings = _unit_embeddings(rows)
    faces = [CL.parse_faces(r["faces_json"]) for r in rows]

    cc = cfg.cluster
    edges: List[Tuple[int, int, str]] = []

    # Layer 1: SHA-256 byte-exact (any time, frozen after this layer)
    sha_edges = CL.layer1_sha_exact(dsu, sha_hashes)
    edges += sha_edges
    frozen_components = {dsu.find(i) for i, j, _ in sha_edges}

    # Layer 2: pHash near-duplicates (30s window, skip frozen SHA components)
    edges += layer2_phash_near(
        dsu, phash_ints, sha_hashes, timestamps,
        threshold=int(cc.get("phash_hamming_threshold", 2)),
        window_seconds=int(cc.get("burst_window_seconds", 30)),
        frozen_components=frozen_components,
        faces=faces,
        require_face_identity=bool(cc.get("require_face_identity", False)),
        identity_cosine_threshold=float(cc.get("face_identity_cosine_threshold", 0.45)),
        identity_min_width_px=float(cc.get("face_identity_min_width_px", 80.0)),
        min_face_score=float(cc.get("min_face_score", 0.6)),
    )

    # Layer 3: DINO burst (30s window, skip frozen SHA components)
    edges += layer_window(
        dsu, rows, units, timestamps, faces,
        window_seconds=int(cc.get("burst_window_seconds", 30)),
        cos_threshold=float(cc.get("dinov2_threshold", 0.92)),
        face_ratio=float(cc.get("face_pose_shift_ratio", 0.30)),
        min_face_score=float(cc.get("min_face_score", 0.6)),
        edge_type="burst",
        valid_embeddings=valid_embeddings,
        frozen_components=frozen_components,
        require_face_identity=bool(cc.get("require_face_identity", False)),
        identity_cosine_threshold=float(cc.get("face_identity_cosine_threshold", 0.45)),
        identity_min_width_px=float(cc.get("face_identity_min_width_px", 80.0)),
    )

    # Layer 4: Similar scene (optional, longer window, skip frozen)
    if bool(cc.get("enable_loose_similar", False)):
        edges += layer_window(
            dsu, rows, units, timestamps, faces,
            window_seconds=int(cc.get("loose_window_seconds", 300)),
            cos_threshold=float(cc.get("loose_dinov2_threshold", 0.96)),
            face_ratio=float(cc.get("face_pose_shift_ratio", 0.30)),
            min_face_score=float(cc.get("min_face_score", 0.6)),
            edge_type="similar_scene",
            valid_embeddings=valid_embeddings,
            frozen_components=frozen_components,
            require_face_identity=bool(cc.get("require_face_identity", False)),
            identity_cosine_threshold=float(cc.get("face_identity_cosine_threshold", 0.45)),
            identity_min_width_px=float(cc.get("face_identity_min_width_px", 80.0)),
        )

    # Label each component by its strongest edge type.
    comp_type: Dict[int, str] = {}
    for i, j, t in edges:
        root = dsu.find(i)
        if PRIORITY[t] > PRIORITY.get(comp_type.get(root, ""), 0):
            comp_type[root] = t

    # Gather components with >= 2 members.
    comps: Dict[int, List[int]] = {}
    for idx in range(n):
        comps.setdefault(dsu.find(idx), []).append(idx)

    db.clear_groups(conn, scope=scope)
    created_at = int(time.time())
    group_count = 0
    decision_counts = {k: 0 for k in D.VALID_DECISIONS}
    reason_counts: Dict[str, int] = {}
    profile = str(cfg.decision.get("profile", "balanced"))
    for root, member_idx in comps.items():
        if len(member_idx) < 2:
            continue
        members = [dict(rows[i]) for i in member_idx]
        for local, gi in enumerate(member_idx):
            if not valid_embeddings[gi]:
                members[local]["dinov2_embedding"] = None
        keep_local, scores = select_keep(members, cfg)
        keep_file_id = int(rows[member_idx[keep_local]]["id"])
        gtype = comp_type.get(root, "burst")
        phase_result = None
        if gtype != "sha_exact":
            phases = PS.segment_phases(
                members,
                max_gap_seconds=int(cc.get("phase_max_gap_seconds", 4)),
                embedding_boundary=float(cc.get("phase_embedding_boundary", 0.93)),
                position_shift=float(cc.get("phase_position_shift", 0.22)),
                group_type=gtype,
            )
            phase_result = PS.select_phase_keepers(
                members, phases,
                keepers_per_phase=int(cc.get("keepers_per_phase", 1)),
                max_group_keepers=int(cc.get("max_group_keepers", 3)),
                diversity_similarity=float(cc.get("keeper_diversity_similarity", 0.965)),
                mmr_quality_weight=float(cc.get("keeper_mmr_quality_weight", 0.7)),
                score_policy=cc.get("keeper_score_policy", "per_member"),
                score_change_margin=cc.get("keeper_score_change_margin", 0.06),
                diversity_policy=cc.get("keeper_diversity_policy", "mmr"),
                diversity_quality_slack=cc.get("keeper_diversity_quality_slack", 0.04),
                pose_policy=cc.get("keeper_pose_policy", "off"),
                pose_displacement_threshold=cc.get("keeper_pose_displacement_threshold", 0.4),
                local_quality_policy=cc.get("keeper_local_quality_policy", "off"),
                local_quality_similarity=cc.get("keeper_local_quality_similarity", 0.9),
                local_quality_penalty=cc.get("keeper_local_quality_penalty", 0.08),
            )
            # Logical-phase utility is the single authority for keeper choice,
            # decision margins, and persisted evidence in visual groups.
            scores = phase_result["utility_scores"]
            keep_local = phase_result["keepers"][0]
            keep_file_id = int(rows[member_idx[keep_local]]["id"])

        # Validate time-span constraint for visual groups
        # CRITICAL: visual groups (non-SHA) must have complete timestamps for all members
        member_timestamps = [timestamps[gi] for gi in member_idx]
        member_timestamps_valid = [t for t in member_timestamps if t is not None]
        if gtype != "sha_exact":
            # Visual groups MUST have timestamps for ALL members
            if len(member_timestamps_valid) != len(member_idx):
                # Incomplete timestamp coverage -> reject as untrusted
                group_trusted = False
                purity = -1.0
            elif len(member_timestamps_valid) >= 2:
                time_span = max(member_timestamps_valid) - min(member_timestamps_valid)
                expected_window = {
                    "phash_near": int(cc.get("burst_window_seconds", 30)),
                    "burst": int(cc.get("burst_window_seconds", 30)),
                    "similar_scene": int(cc.get("loose_window_seconds", 300)),
                }.get(gtype, int(cc.get("burst_window_seconds", 30)))
                if time_span > expected_window:
                    # Transitive chain violation - downgrade trust
                    group_trusted = False
                    purity = -1.0
                else:
                    # DSU reachability is not enough: require every burst member to remain
                    # close to the selected representative, preventing A~B~C chaining from
                    # turning an outlier into an automatic removal.
                    if all(valid_embeddings[gi] for gi in member_idx):
                        purity = min(float(np.dot(units[member_idx[keep_local]], units[gi]))
                                     for gi in member_idx)
                    else:
                        purity = -1.0
                    group_trusted = gtype == "burst" and purity >= float(cc.get("dinov2_threshold", 0.92))
            else:
                # Visual group with <2 valid timestamps -> untrusted
                group_trusted = False
                purity = -1.0
        else:
            # SHA-exact groups: always trusted for byte-identical members
            if all(valid_embeddings[gi] for gi in member_idx):
                purity = min(float(np.dot(units[member_idx[keep_local]], units[gi]))
                             for gi in member_idx)
            else:
                purity = -1.0
            group_trusted = gtype == "sha_exact"

        keeper_sha = rows[member_idx[keep_local]]["content_sha256"]
        safe_duplicates = {
            local: bool(keeper_sha and keeper_sha == rows[gi]["content_sha256"])
            for local, gi in enumerate(member_idx)
        }
        result = D.decide_group(
            members, keep_local, scores, gtype, profile=profile,
            group_trusted=group_trusted,
            safe_duplicates=safe_duplicates,
        )
        if phase_result is not None:
            for extra_keeper in phase_result["keepers"]:
                if extra_keeper == keep_local:
                    continue
                pose_extra = (phase_result.get("pose_coverage") or {}).get("added_keeper") == extra_keeper
                result["members"][extra_keeper] = {
                    "decision": "KEEP", "confidence": 1.0,
                    "reason": "POSE_VARIANT_KEEPER" if pose_extra else "LOGICAL_PHASE_KEEPER",
                    "evidence": {
                        "utility_score": float(scores[extra_keeper]),
                        "pair_margin": float(scores[keep_local] - scores[extra_keeper]),
                        "logical_phase_protection": not pose_extra,
                        **({"pose_variation_protection": True, "semantic_phase_authority": False,
                            "pose_comparisons": phase_result["pose_coverage"]["comparisons"]} if pose_extra else {}),
                        "physical_group_split": False,
                    },
                }
            result["state"] = "REVIEW_REQUIRED"
        member_tuples = []
        for local, gi in enumerate(member_idx):
            fid = int(rows[gi]["id"])
            record = result["members"][local]
            is_keep = record["decision"] == "KEEP"
            reason = record["reason"]
            member_tuples.append((fid, is_keep, reason))
            decision_counts[record["decision"]] += 1
            reason_counts[reason.split(":", 1)[0]] = reason_counts.get(reason.split(":", 1)[0], 0) + 1
        gid = db.insert_group(
            conn, gtype, keep_file_id, member_tuples, created_at,
            decision_state=result["state"],
            confidence=min(r["confidence"] for r in result["members"].values()),
            policy_version=D.POLICY_VERSION,
            decision_json=json.dumps({"profile": profile, "group_purity": purity,
                                      "group_trusted": group_trusted,
                                      "phase_selection": phase_result}),
        )
        db.update_member_decisions(conn, gid, [
            {"file_id": int(rows[gi]["id"]), **result["members"][local],
             "evidence_json": json.dumps(result["members"][local]["evidence"])}
            for local, gi in enumerate(member_idx)
        ])
        group_count += 1
    conn.commit()
    db.set_meta(conn, "stage2_done_at", str(created_at))
    db.set_meta(conn, "stage2_run_id", uuid.uuid4().hex)

    candidates = sum(decision_counts[k] for k in ("AUTO_REMOVE", "MAYBE", "UNKNOWN"))
    stats = {
        "files": n, "groups": group_count,
        "to_delete": decision_counts["AUTO_REMOVE"],
        "auto_remove": decision_counts["AUTO_REMOVE"],
        "maybe": decision_counts["MAYBE"], "unknown": decision_counts["UNKNOWN"],
        "auto_coverage": decision_counts["AUTO_REMOVE"] / candidates if candidates else 0.0,
        "reasons": reason_counts, "profile": profile,
    }
    if scope is not None and getattr(scope, "bound", False):
        stats["scope"] = root_scope.summary(conn, scope)
    db.set_meta(conn, "stage2_stats", json.dumps(stats))
    conn.commit()
    print(f"[stage2] {stats}")
    return stats


def run(config_path: Optional[str] = None,
        root_override: Optional[str] = None) -> Dict[str, int]:
    cfg = load_config(config_path)
    cfg.validate()  # Fail closed on invalid thresholds before any DB access
    # Fail closed before the database is opened/created (see stage 1).
    root_scope.preflight_stage(cfg.db_path, root_override, cfg.declared_root)
    conn = db.open_db(cfg.db_path)
    # Never cluster unscoped: an unbound directory adopts --root or a declared
    # paths.root, else we refuse. A defaulted root never binds anything.
    scope = root_scope.adopt_or_resolve(
        conn, root_override, db_path=str(cfg.db_path), declared_root=cfg.declared_root
    )
    stats = cluster(conn, cfg, scope=scope)
    conn.close()
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: clustering")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None,
                    help="verify the output directory belongs to this photo root")
    args = ap.parse_args(argv)
    run(config_path=args.config, root_override=args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
