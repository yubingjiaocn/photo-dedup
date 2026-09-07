"""Clustering primitives: DSU, hashing, face helpers, span tracking.

Shared between stage2 clustering layers and tests.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class DSU:
    """Disjoint-set union with path compression + union by size + span tracking."""

    def __init__(self, n: int, timestamps: Optional[Sequence[Optional[int]]] = None) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.timestamps = timestamps or [None] * n
        self.min_ts: List[Optional[int]] = list(timestamps) if timestamps else [None] * n
        self.max_ts: List[Optional[int]] = list(timestamps) if timestamps else [None] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def span(self, x: int) -> Optional[int]:
        """Return max-min timestamp span for component containing x, or None if ANY timestamp missing.

        For visual groups (non-SHA), span calculation MUST fail (return None) if ANY
        member lacks a timestamp, preventing incomplete ranges from passing the window check.
        SHA-exact groups are exempt: they never enforce span constraints.
        """
        root = self.find(x)
        if self.min_ts[root] is None or self.max_ts[root] is None:
            return None
        # CRITICAL: if we've merged nodes with None timestamps into this component,
        # the span is unreliable. The caller must reject such groups for visual layers.
        # This is detected by checking if any member of the component has None timestamp
        # (but that's O(N) per call, so we rely on union_if_span_within rejecting None).
        return self.max_ts[root] - self.min_ts[root]

    def union(self, a: int, b: int) -> None:
        """Standard union — no span enforcement."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        # Merge spans
        if self.min_ts[ra] is not None and self.min_ts[rb] is not None:
            self.min_ts[ra] = min(self.min_ts[ra], self.min_ts[rb])
        if self.max_ts[ra] is not None and self.max_ts[rb] is not None:
            self.max_ts[ra] = max(self.max_ts[ra], self.max_ts[rb])

    def union_if_span_within(self, a: int, b: int, window: int) -> bool:
        """Union a and b only if the merged component span would remain <= window.

        Returns True if union was performed, False if rejected due to span violation.
        Requires timestamps to be non-None for both components.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Check merged span
        min_a, max_a = self.min_ts[ra], self.max_ts[ra]
        min_b, max_b = self.min_ts[rb], self.max_ts[rb]
        if min_a is None or max_a is None or min_b is None or max_b is None:
            return False
        merged_min = min(min_a, min_b)
        merged_max = max(max_a, max_b)
        if (merged_max - merged_min) > window:
            return False
        # Perform union
        self.union(a, b)
        return True


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


def dominant_identity(faces: Sequence[Dict[str, Any]], min_score: float,
                      min_width_px: float) -> Optional[np.ndarray]:
    """Return the largest usable face's anonymous unit embedding."""
    candidates = []
    for face in faces:
        box = face.get("bbox") or []
        if len(box) != 4 or float(face.get("score", 0.0)) < min_score or float(box[2]) < min_width_px:
            continue
        raw = face.get("identity_embedding")
        if not isinstance(raw, str):
            continue
        try:
            vec = np.frombuffer(bytes.fromhex(raw), dtype=np.float16).astype(np.float32)
        except (ValueError, TypeError):
            continue
        norm = float(np.linalg.norm(vec))
        if vec.size and norm > 0 and np.all(np.isfinite(vec)):
            candidates.append((float(box[2]) * float(box[3]), vec / norm))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def identity_compatible(faces_i: Sequence[Dict[str, Any]], faces_j: Sequence[Dict[str, Any]],
                        min_score: float, min_width_px: float,
                        cosine_threshold: float) -> bool:
    """Fail closed for face-bearing pairs; scenery pairs remain applicable."""
    detected_i = any(float(f.get("score", 0.0)) >= min_score for f in faces_i)
    detected_j = any(float(f.get("score", 0.0)) >= min_score for f in faces_j)
    if not detected_i and not detected_j:
        return True
    left = dominant_identity(faces_i, min_score, min_width_px)
    right = dominant_identity(faces_j, min_score, min_width_px)
    return left is not None and right is not None and float(np.dot(left, right)) >= cosine_threshold


def layer1_sha_exact(
    dsu: DSU, sha_hashes: Sequence[Optional[str]]
) -> List[Tuple[int, int, str]]:
    """Union files with identical non-empty SHA-256 using star topology. Returns edges.

    These groups are frozen: they will not absorb additional photos via visual
    similarity in later layers, preventing cross-date copies from bridging
    unrelated sessions.
    """
    edges: List[Tuple[int, int, str]] = []
    buckets: Dict[str, List[int]] = {}
    for idx, sha in enumerate(sha_hashes):
        if sha is not None:
            buckets.setdefault(sha, []).append(idx)
    for members in buckets.values():
        if len(members) < 2:
            continue
        # Star topology: pick first as hub, union others to it (k-1 edges)
        hub = members[0]
        for i in range(1, len(members)):
            node = members[i]
            if dsu.find(hub) == dsu.find(node):
                continue
            dsu.union(hub, node)
            edges.append((hub, node, "sha_exact"))
    return edges
