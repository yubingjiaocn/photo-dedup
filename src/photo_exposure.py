"""Auditable local photo-exposure router for a production CV -> VLM cascade.

The module has no label, provider, database, group-ID, or write dependency.  It
reads decoded images supplied by its caller, builds near-homogeneous
micro-clusters by direct representative comparison, and emits exact
``file_id + content_sha256`` identities for the photos that may be exposed to a
VLM.  A contact sheet is never an accounting shortcut: callers count every
source identity represented in it.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

ROUTER_VERSION = "photo-exposure-router-v1"

# Kept for compatibility with the original B-route experiment.  These are
# label-free pixel gates, not semantic-stage classifiers.
POLICIES = {
    "strict": {"mae": 0.012, "tile_p95": 0.035, "dhash": 2, "seconds": 0},
    "balanced": {"mae": 0.025, "tile_p95": 0.060, "dhash": 4, "seconds": 1},
    "aggressive": {"mae": 0.040, "tile_p95": 0.100, "dhash": 8, "seconds": 1},
}

# Production-frontier profiles.  local_resolve_* gates are deliberately much
# tighter than representative compression.  A profile can therefore save VLM
# photos without claiming that CV understands expression or action stages.
ROUTER_PROFILES = {
    "recall_first": {
        "cluster": POLICIES["strict"],
        "local_resolve_mae": 0.006,
        "local_resolve_tile_p95": 0.015,
        "local_resolve_dhash": 1,
        "quality_margin": 0.18,
    },
    "balanced": {
        "cluster": POLICIES["balanced"],
        "local_resolve_mae": 0.010,
        "local_resolve_tile_p95": 0.025,
        "local_resolve_dhash": 2,
        "quality_margin": 0.14,
    },
    "efficient": {
        "cluster": POLICIES["aggressive"],
        "local_resolve_mae": 0.016,
        "local_resolve_tile_p95": 0.040,
        "local_resolve_dhash": 4,
        "quality_margin": 0.10,
    },
}


def descriptor(image):
    """Return a deterministic, compact local descriptor from a decoded image."""
    rgb = (
        np.asarray(
            image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255
    )
    gray = np.asarray(
        image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16
    )
    high = np.asarray(
        image.convert("L").resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    gradient = float(
        np.mean(np.abs(np.diff(high, axis=0)))
        + np.mean(np.abs(np.diff(high, axis=1)))
    )
    # Histograms and edge layout are cheap surrogates for composition/lighting
    # drift.  They are reported for audit, but never treated as human preference.
    hist = np.concatenate(
        [np.histogram(rgb[:, :, i], bins=16, range=(0, 1), density=False)[0] for i in range(3)]
    ).astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    edge = np.hypot(np.diff(high, axis=1)[:-1], np.diff(high, axis=0)[:, :-1])
    edge_blocks = edge[:248, :248].reshape(8, 31, 8, 31).mean(axis=(1, 3)) / 255
    return {
        "rgb": rgb,
        "dhash": gray[:, 1:] > gray[:, :-1],
        "quality": gradient,
        "hist": hist,
        "edge": edge_blocks,
        "aspect": image.width / image.height,
    }


def pair_metrics(left, right):
    """Compute auditable pair features, or None when comparison is unsafe."""
    if left["time"] is None or right["time"] is None:
        return None
    a, b = left["descriptor"], right["descriptor"]
    if abs(a["aspect"] - b["aspect"]) > 0.001:
        return None
    pixels = np.mean(np.abs(a["rgb"] - b["rgb"]), axis=2)
    metrics = {
        "mae": float(pixels.mean()),
        "tile_p95": float(
            np.quantile(pixels.reshape(8, 8, 8, 8).mean(axis=(1, 3)), 0.95)
        ),
        "dhash": int(np.count_nonzero(a["dhash"] != b["dhash"])),
        "seconds": abs(left["time"] - right["time"]),
        "hist_l1": float(np.abs(a.get("hist", 0) - b.get("hist", 0)).sum()),
        "edge_mae": float(np.abs(a.get("edge", 0) - b.get("edge", 0)).mean()),
    }
    return metrics


def _within(stats, policy):
    return stats is not None and all(stats[k] <= policy[k] for k in policy)


def comparison(left, right, policy):
    stats = pair_metrics(left, right)
    return stats if _within(stats, policy) else None


def micro_clusters(frames, policy):
    """Choose by local gradient, then attach only direct near-duplicates."""
    ordered = sorted(frames, key=lambda x: (-x["descriptor"]["quality"], x["identity"]))
    clusters = []
    for frame in ordered:
        for cluster in clusters:
            stats = comparison(frame, cluster["source"], policy)
            if stats is not None:
                cluster["members"].append(frame["identity"])
                cluster["omissions"].append(
                    {"frame": frame["identity"], "comparison": stats}
                )
                break
        else:
            clusters.append(
                {
                    "source": frame,
                    "representative": frame["identity"],
                    "members": [frame["identity"]],
                    "omissions": [],
                }
            )
    return [{k: v for k, v in c.items() if k != "source"} for c in clusters]


def _quality_margin(frames):
    qualities = sorted((float(f["descriptor"]["quality"]) for f in frames), reverse=True)
    if len(qualities) < 2 or qualities[0] <= 0:
        return 0.0
    return (qualities[0] - qualities[1]) / qualities[0]


def _locally_resolvable(frames, clusters, profile):
    """Resolve only one extremely homogeneous cluster with a clear quality lead."""
    if len(clusters) != 1 or len(frames) < 2:
        return False, []
    comparisons = [o["comparison"] for o in clusters[0]["omissions"]]
    limits = {
        "mae": profile["local_resolve_mae"],
        "tile_p95": profile["local_resolve_tile_p95"],
        "dhash": profile["local_resolve_dhash"],
    }
    reasons = []
    for key, limit in limits.items():
        observed = max((c[key] for c in comparisons), default=math.inf)
        reasons.append({"feature": key, "observed_max": observed, "limit": limit})
        if observed > limit:
            return False, reasons
    margin = _quality_margin(frames)
    reasons.append(
        {"feature": "quality_margin", "observed_min": margin, "limit": profile["quality_margin"]}
    )
    return margin >= profile["quality_margin"], reasons


def route_group(frames, profile_name="recall_first"):
    """Route one group without labels or semantic inference.

    Frames must contain ``identity`` (the canonical ``I<id>@<sha>`` string), a
    trusted/unknown ``time``, and a descriptor.  The returned ``vlm_exposed``
    list is the complete source-photo allowlist for constructing VLM assets.
    """
    if profile_name not in ROUTER_PROFILES:
        raise ValueError(f"unknown router profile: {profile_name}")
    if len(frames) < 2:
        raise ValueError("at least two frames required")
    identities = [f["identity"] for f in frames]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate frame identity")
    profile = ROUTER_PROFILES[profile_name]
    clusters = micro_clusters(frames, profile["cluster"])
    local, local_evidence = _locally_resolvable(frames, clusters, profile)
    representatives = [c["representative"] for c in clusters]
    # Two-frame selection semantics remain owned by v3.  This layer may only
    # decide exposure; it never emits a replacement selection for a pair.
    pair_passthrough = len(frames) == 2
    if local and not pair_passthrough:
        route = "local_resolved"
        local_selected = representatives
        exposed = []
    else:
        route = "vlm_required"
        local_selected = []
        exposed = representatives
    return {
        "router_version": ROUTER_VERSION,
        "profile": profile_name,
        "route": route,
        "pair_selection_contract": "v3_unchanged" if pair_passthrough else None,
        "member_count": len(frames),
        "micro_clusters": clusters,
        "local_selected": local_selected,
        "vlm_required": route == "vlm_required",
        "vlm_exposed": exposed,
        "member_to_representative_ratio": len(frames) / max(len(representatives), 1),
        "local_resolution_evidence": local_evidence,
        "accounting_rule": "Every source in any page, crop, storyboard, or contact sheet counts as exposed.",
    }
