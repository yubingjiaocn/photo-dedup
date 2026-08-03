"""Stage 2 -- clustering (CPU-only, fast, re-runnable).

Reads the ``features`` table and forms groups of duplicate / near-duplicate
still images, choosing one keeper per group. Tweaking a threshold means
re-running *only* this stage -- no images are read.

Three layers, applied in priority order (a file lands in exactly one group):

  Layer 1  exact_dup      pHash hamming <= threshold        (any time)
  Layer 2  burst          <= 30s apart AND DINOv2 cos >= t   (+ check-in split)
  Layer 3  similar_scene  longer window, stricter cos        (off by default)

Check-in split
--------------
The hard case: a landmark/building is the subject and a person re-poses in
front of it. Such frames are visually near-identical (high DINOv2 cos) but are
NOT duplicates. If a candidate burst pair both contain a face and the dominant
face center moves more than ``face_pose_shift_ratio`` of the frame, we refuse
to union them -- so a run of check-in shots stays as separate keepers.

Efficiency
----------
Exact-dup uses multi-index hashing: split the 64-bit pHash into four 16-bit
bands; for hamming <= 2 (pigeonhole) two colliding items must share >=2 bands,
so we only compare within band buckets -- no O(N^2) scan. Burst uses a time
sliding window, so comparisons stay local.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, load_config
from . import db
from . import quality as Q
from . import decision as D

PRIORITY = {"exact_dup": 3, "burst": 2, "similar_scene": 1}


# --- union-find ------------------------------------------------------------

class DSU:
    """Disjoint-set union with path compression + union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


# --- face helpers ----------------------------------------------------------

def parse_faces(faces_json: Optional[str]) -> List[Dict[str, Any]]:
    if not faces_json:
        return []
    try:
        return json.loads(faces_json)
    except (ValueError, TypeError):
        return []


def dominant_face_center(
    faces: Sequence[Dict[str, Any]], width: int, height: int, min_score: float
) -> Optional[Tuple[float, float]]:
    """Normalised (cx, cy) in 0..1 of the highest-confidence usable face."""
    usable = [f for f in faces if float(f.get("score", 0.0)) >= min_score]
    if not usable or not width or not height:
        return None
    best = max(usable, key=lambda f: float(f.get("score", 0.0)))
    x, y, w, h = (float(v) for v in (best.get("bbox") or [0, 0, 0, 0])[:4])
    return ((x + w / 2.0) / width, (y + h / 2.0) / height)


def face_pose_shift(
    faces_i: Sequence[Dict[str, Any]], wi: int, hi: int,
    faces_j: Sequence[Dict[str, Any]], wj: int, hj: int,
    min_score: float,
) -> Optional[float]:
    """Normalised face-center displacement between two frames.

    Returns None when either frame lacks a usable face (so the split rule does
    not apply -- backgrounds-only bursts group normally).
    """
    ci = dominant_face_center(faces_i, wi, hi, min_score)
    cj = dominant_face_center(faces_j, wj, hj, min_score)
    if ci is None or cj is None:
        return None
    dx = abs(ci[0] - cj[0])
    dy = abs(ci[1] - cj[1])
    return max(dx, dy)


def should_split_by_face(shift: Optional[float], ratio: float) -> bool:
    return shift is not None and shift > ratio


# --- keep selection --------------------------------------------------------

def _face_quality_of(meta_json: Optional[str]) -> float:
    if not meta_json:
        return 0.0
    try:
        meta = json.loads(meta_json)
        return float(meta.get("face_quality") or 0.0)
    except (ValueError, TypeError):
        return 0.0


def select_keep(members: Sequence[Dict[str, Any]], cfg: Config) -> Tuple[int, Dict[int, float]]:
    """Return (index_of_keeper, {index: score}) for a group's member dicts.

    Each member dict needs: quality_score, quality_meta, width, height, size_bytes.
    Ties break toward the larger file (original vs recompressed).
    """
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
    # Pick max score; tie-break on larger size then smaller id-order.
    keep = max(range(len(members)),
               key=lambda i: (scores[i], members[i].get("size_bytes") or 0))
    return keep, scores


# --- pHash exact-dup layer -------------------------------------------------

def _phash_ints(rows: Sequence[Any]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for r in rows:
        blob = r["phash"]
        out.append(int.from_bytes(blob, "big") if blob else None)
    return out


def layer1_exact_dup(
    dsu: DSU, phash_ints: Sequence[Optional[int]], threshold: int
) -> List[Tuple[int, int, str]]:
    """Union files whose pHash hamming distance <= threshold. Returns edges."""
    edges: List[Tuple[int, int, str]] = []
    bands = 4
    band_bits = 16
    mask = (1 << band_bits) - 1
    for band in range(bands):
        shift = band * band_bits
        buckets: Dict[int, List[int]] = {}
        for idx, h in enumerate(phash_ints):
            if h is None:
                continue
            buckets.setdefault((h >> shift) & mask, []).append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    if dsu.find(i) == dsu.find(j):
                        continue
                    hi, hj = phash_ints[i], phash_ints[j]
                    if bin(hi ^ hj).count("1") <= threshold:
                        dsu.union(i, j)
                        edges.append((i, j, "exact_dup"))
    return edges


# --- burst / similar layer -------------------------------------------------

def _unit_embeddings(rows: Sequence[Any]) -> np.ndarray:
    embs = np.zeros((len(rows), 768), dtype=np.float32)
    for i, r in enumerate(rows):
        blob = r["dinov2_embedding"]
        if blob:
            v = Q.blob_to_embedding(blob)
            if v.shape[0] == 768:
                embs[i] = v
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def layer_window(
    dsu: DSU,
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
) -> List[Tuple[int, int, str]]:
    """Time-windowed similarity union with check-in face split. Returns edges."""
    edges: List[Tuple[int, int, str]] = []
    n = len(rows)
    for i in range(n):
        ti = timestamps[i]
        if ti is None:
            continue
        j = i + 1
        while j < n and timestamps[j] is not None and (timestamps[j] - ti) <= window_seconds:
            if dsu.find(i) != dsu.find(j):
                cos = float(np.dot(units[i], units[j]))
                if cos >= cos_threshold:
                    shift = face_pose_shift(
                        faces[i], rows[i]["width"], rows[i]["height"],
                        faces[j], rows[j]["width"], rows[j]["height"],
                        min_face_score,
                    )
                    if not should_split_by_face(shift, face_ratio):
                        dsu.union(i, j)
                        edges.append((i, j, edge_type))
            j += 1
    return edges


# --- orchestration ---------------------------------------------------------

def cluster(conn, cfg: Config) -> Dict[str, int]:
    """Run all enabled clustering layers and persist groups. Returns stats."""
    rows = db.load_features_joined(conn)
    n = len(rows)
    if n == 0:
        print("[stage2] no features found -- run stage1 first.")
        return {"groups": 0, "files": 0}

    dsu = DSU(n)
    phash_ints = _phash_ints(rows)
    units = _unit_embeddings(rows)
    timestamps = [r["exif_timestamp"] for r in rows]
    faces = [parse_faces(r["faces_json"]) for r in rows]

    cc = cfg.cluster
    edges: List[Tuple[int, int, str]] = []
    edges += layer1_exact_dup(dsu, phash_ints, int(cc.get("phash_hamming_threshold", 2)))
    edges += layer_window(
        dsu, rows, units, timestamps, faces,
        window_seconds=int(cc.get("burst_window_seconds", 30)),
        cos_threshold=float(cc.get("dinov2_threshold", 0.92)),
        face_ratio=float(cc.get("face_pose_shift_ratio", 0.30)),
        min_face_score=float(cc.get("min_face_score", 0.6)),
        edge_type="burst",
    )
    if bool(cc.get("enable_loose_similar", False)):
        edges += layer_window(
            dsu, rows, units, timestamps, faces,
            window_seconds=int(cc.get("loose_window_seconds", 300)),
            cos_threshold=float(cc.get("loose_dinov2_threshold", 0.96)),
            face_ratio=float(cc.get("face_pose_shift_ratio", 0.30)),
            min_face_score=float(cc.get("min_face_score", 0.6)),
            edge_type="similar_scene",
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

    db.clear_groups(conn)
    created_at = int(time.time())
    group_count = 0
    decision_counts = {k: 0 for k in D.VALID_DECISIONS}
    reason_counts: Dict[str, int] = {}
    profile = str(cfg.decision.get("profile", "balanced"))
    for root, member_idx in comps.items():
        if len(member_idx) < 2:
            continue
        members = [dict(rows[i]) for i in member_idx]
        keep_local, scores = select_keep(members, cfg)
        keep_file_id = int(rows[member_idx[keep_local]]["id"])
        gtype = comp_type.get(root, "burst")
        keep_hash = rows[member_idx[keep_local]]["phash"]
        distances = {
            local: Q.hamming(keep_hash, rows[gi]["phash"])
            for local, gi in enumerate(member_idx)
        }
        # DSU reachability is not enough: require every burst member to remain
        # close to the selected representative, preventing A~B~C chaining from
        # turning an outlier into an automatic removal.
        purity = min(float(np.dot(units[member_idx[keep_local]], units[gi]))
                     for gi in member_idx)
        group_trusted = gtype == "exact_dup" or (
            gtype == "burst" and purity >= float(cc.get("dinov2_threshold", 0.92))
        )
        result = D.decide_group(
            members, keep_local, scores, gtype, profile=profile,
            phash_distances=distances, group_trusted=group_trusted,
        )
        member_tuples = []
        for local, gi in enumerate(member_idx):
            fid = int(rows[gi]["id"])
            is_keep = local == keep_local
            record = result["members"][local]
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
                                      "group_trusted": group_trusted}),
        )
        db.update_member_decisions(conn, gid, [
            {"file_id": int(rows[gi]["id"]), **result["members"][local],
             "evidence_json": json.dumps(result["members"][local]["evidence"])}
            for local, gi in enumerate(member_idx)
        ])
        group_count += 1
    conn.commit()
    db.set_meta(conn, "stage2_done_at", str(created_at))

    candidates = sum(decision_counts[k] for k in ("AUTO_REMOVE", "MAYBE", "UNKNOWN"))
    stats = {
        "files": n, "groups": group_count,
        "to_delete": decision_counts["AUTO_REMOVE"],
        "auto_remove": decision_counts["AUTO_REMOVE"],
        "maybe": decision_counts["MAYBE"], "unknown": decision_counts["UNKNOWN"],
        "auto_coverage": decision_counts["AUTO_REMOVE"] / candidates if candidates else 0.0,
        "reasons": reason_counts, "profile": profile,
    }
    db.set_meta(conn, "stage2_stats", json.dumps(stats))
    conn.commit()
    print(f"[stage2] {stats}")
    return stats


def run(config_path: Optional[str] = None) -> Dict[str, int]:
    cfg = load_config(config_path)
    conn = db.open_db(cfg.db_path)
    stats = cluster(conn, cfg)
    conn.close()
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: clustering")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    run(config_path=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
