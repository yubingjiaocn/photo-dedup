"""Explainable exposure features and labels (numpy/OpenCV, no model weights).

Raw metrics are persisted in ``quality_meta`` so policy thresholds can change
without re-reading the photo library.  The 512 px long-edge resize is part of
this feature definition; changing it requires recalibration and a Stage 1 run.

The resize is split out as :func:`prepare` so Stage 1 can perform it on a
producer thread (it is pure numpy/Pillow and releases the GIL) while the metrics
themselves still run on the main thread once face boxes are known.
:func:`extract` is exactly ``prepare`` followed by :func:`extract_prepared`, so
there is only one definition of the feature and the threaded and serial paths
cannot drift apart.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

DEFAULT_THRESHOLDS = {
    "hi_reject": 0.35, "blob_reject": 0.20, "anchor_reject": 0.10,
    "lo_reject": 0.60, "anchor_reject_dark": 0.05,
    "entropy_reject": 3.00, "mass_reject": 0.25,
    "hi_maybe": 0.15, "lo_maybe": 0.35,
    "anchor_maybe": 0.25, "mass_maybe": 0.45,
}


def resize_rgb(image: Image.Image, long_edge: int = 512) -> np.ndarray:
    """Return an HxWx3 float32 sRGB array with longest edge <= long_edge."""
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, float(long_edge) / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def prepare(image: Image.Image, long_edge: int = 512) -> np.ndarray:
    """The thread-safe half of :func:`extract`: just the calibrated resize."""
    return resize_rgb(image, long_edge)


def _largest_blob(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    try:
        import cv2
        u8 = mask.astype(np.uint8)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        if count <= 1:
            return 0.0
        return float(stats[1:, cv2.CC_STAT_AREA].max() / mask.size)
    except ImportError:  # conservative fallback; production requirements include OpenCV
        return float(mask.mean())


def _metrics(rgb: np.ndarray) -> Dict[str, float]:
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    hi = rgb.min(axis=2) >= (250.0 / 255.0)
    lo = rgb.max(axis=2) <= (5.0 / 255.0)
    keep = ~(hi | lo)
    entropy = 0.0
    if int(keep.sum()) > 64:
        hist = np.histogram(y[keep], bins=64, range=(0.0, 1.0))[0]
        p = hist[hist > 0].astype(np.float64) / hist.sum()
        entropy = float(-(p * np.log2(p)).sum())
    return {
        "clip_hi": float(hi.mean()), "clip_lo": float(lo.mean()),
        "blob_hi": _largest_blob(hi), "blob_lo": _largest_blob(lo),
        "mass_usable": float(((y > 0.03) & (y < 0.97)).mean()),
        "anchor_mass": float(((y > 0.25) & (y < 0.85)).mean()),
        "entropy_nonclip": entropy, "ymean": float(y.mean()),
    }


def extract(image: Image.Image, faces: Sequence[Mapping[str, Any]], long_edge: int = 512) -> Dict[str, Any]:
    """Extract global metrics and per-face ROI exposure from one decoded image."""
    return extract_prepared(prepare(image, long_edge), image.size, faces, long_edge)


def extract_prepared(rgb: np.ndarray, source_size: Tuple[int, int],
                     faces: Sequence[Mapping[str, Any]], long_edge: int = 512) -> Dict[str, Any]:
    """Metrics from an already-resized array (identical to :func:`extract`).

    ``source_size`` is the decoded image's ``(width, height)``: face boxes are in
    full-resolution coordinates and must be mapped onto ``rgb``.
    """
    out: Dict[str, Any] = _metrics(rgb)
    out["decode_long_edge"] = int(long_edge)
    ow, oh = source_size
    h, w = rgb.shape[:2]
    rois = []
    for face in faces:
        box = face.get("bbox") or []
        if len(box) < 4:
            continue
        x, y, bw, bh = map(float, box[:4])
        x0, x1 = max(0, round(x * w / ow)), min(w, round((x + bw) * w / ow))
        y0, y1 = max(0, round(y * h / oh)), min(h, round((y + bh) * h / oh))
        if x1 <= x0 or y1 <= y0:
            continue
        fm = _metrics(rgb[y0:y1, x0:x1])
        rois.append({"score": float(face.get("score", 0.0)), **fm})
    out["faces"] = rois
    return out


def classify(metrics: Mapping[str, Any], thresholds: Mapping[str, float] | None = None) -> Tuple[str, str, float]:
    """Return (ok|maybe|reject, human reason, severity); all rejects use AND gates."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ch, cl = float(metrics["clip_hi"]), float(metrics["clip_lo"])
    bh, bl = float(metrics["blob_hi"]), float(metrics["blob_lo"])
    anc = float(metrics["anchor_mass"])
    mass, ent = float(metrics["mass_usable"]), float(metrics["entropy_nonclip"])
    for face in metrics.get("faces", []):
        fy = float(face["ymean"])
        fhi, flo = float(face["clip_hi"]), float(face["clip_lo"])
        if 0.18 <= fy <= 0.85 and fhi < 0.20 and flo < 0.20:
            if ch >= t["hi_maybe"] or (cl >= t["lo_maybe"] and anc < t["anchor_maybe"]):
                return "maybe", f"frame clipped hi {ch:.0%}/lo {cl:.0%}, face well exposed", max(ch, cl)
            return "ok", "face well exposed", 0.0
        if fhi > 0.60 or flo > 0.60:
            return "reject", f"face itself clipped hi {fhi:.0%}/lo {flo:.0%}", max(fhi, flo)
    if ch >= t["hi_reject"] and bh >= t["blob_reject"] and anc < t["anchor_reject"]:
        return "reject", f"blown {ch:.0%} (blob {bh:.0%}), anchor {anc:.0%}", ch
    if cl >= t["lo_reject"] and anc < t["anchor_reject_dark"] and ent < t["entropy_reject"]:
        return "reject", f"black {cl:.0%}, anchor {anc:.0%}, entropy {ent:.1f}", cl
    if mass < t["mass_reject"] and anc < t["anchor_reject_dark"]:
        return "reject", f"only {mass:.0%} usable tones, anchor {anc:.0%}", 1.0 - mass
    if ch >= t["hi_maybe"]:
        return "maybe", f"clip hi {ch:.0%} (blob {bh:.0%}), anchor {anc:.0%}", ch
    if cl >= t["lo_maybe"] and anc < t["anchor_maybe"]:
        return "maybe", f"clip lo {cl:.0%} (blob {bl:.0%}), anchor {anc:.0%}", cl
    if mass < t["mass_maybe"]:
        return "maybe", f"only {mass:.0%} usable tones", 1.0 - mass
    return "ok", "", 0.0
