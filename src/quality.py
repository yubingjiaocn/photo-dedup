"""Image-quality helpers: pHash, sharpness, face quality, and keep-scoring.

This module is intentionally dependency-light (numpy + Pillow only) so it can
be exercised by the test suite and the "stub" feature backend without the
heavy GPU stack. The real MUSIQ / CLIP-IQA metrics come from ``pyiqa`` and are
only imported inside stage1's torch backend.

Keep-score
----------
Per group we pick the file with the highest::

    score = 0.6 * iqa_norm          # MUSIQ normalised to 0..1
          + 0.3 * face_quality      # 0..1, 0 when no face
          + 0.1 * resolution_bonus  # 0..1, capped at resolution_ref_mp

See docs/ALGORITHM.md for the reasoning behind each weight.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

try:  # Pillow is a hard dependency; guard only to give a nice message.
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise ImportError("Pillow is required for src.quality") from exc


# --- perceptual hash -------------------------------------------------------

def _dct_matrix(n: int) -> np.ndarray:
    """DCT-II basis matrix (unnormalised; ordering is all we need for hashing)."""
    k = np.arange(n)
    return np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))


def _phash_numpy(image: "Image.Image", hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Pure-numpy perceptual hash (no scipy). Returns a 64-bit int."""
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    m = _dct_matrix(img_size)
    dct = m @ pixels @ m.T
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # skip DC term when thresholding
    bits = (low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash_bytes(image: "Image.Image", hash_size: int = 8) -> bytes:
    """Compute an 8-byte (64-bit) perceptual hash for a PIL image.

    Prefers the ``imagehash`` library (matches the design spec); falls back to
    the self-contained numpy implementation if it is unavailable.
    """
    try:
        import imagehash  # type: ignore

        h = imagehash.phash(image, hash_size=hash_size)
        bits = np.asarray(h.hash, dtype=bool).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
    except Exception:
        value = _phash_numpy(image, hash_size=hash_size)
    return value.to_bytes((hash_size * hash_size) // 8, "big")


def hamming(a: bytes, b: bytes) -> int:
    """Hamming distance between two equal-length hash blobs."""
    if a is None or b is None or len(a) != len(b):
        return 64  # treat as maximally different
    x = int.from_bytes(a, "big") ^ int.from_bytes(b, "big")
    return int(bin(x).count("1"))


# --- sharpness -------------------------------------------------------------

def variance_of_laplacian(gray: np.ndarray) -> float:
    """Variance of the Laplacian: a cheap, robust focus/sharpness proxy.

    Higher = sharper. Uses a 3x3 discrete Laplacian via numpy (no OpenCV).
    """
    g = gray.astype(np.float64)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    # Trim the wrapped border so np.roll edge artifacts don't skew the variance.
    if lap.shape[0] > 2 and lap.shape[1] > 2:
        lap = lap[1:-1, 1:-1]
    return float(lap.var())


def normalize_sharpness(v: float, ref: float = 500.0) -> float:
    """Map a raw Laplacian variance to 0..1 via a saturating curve."""
    if v <= 0:
        return 0.0
    return float(v / (v + ref))


# --- resolution ------------------------------------------------------------

def resolution_bonus(width: Optional[int], height: Optional[int], ref_mp: float = 12.0) -> float:
    """0..1 bonus rewarding higher resolution, capped at ``ref_mp`` megapixels."""
    if not width or not height:
        return 0.0
    megapixels = (width * height) / 1_000_000.0
    return float(min(1.0, megapixels / ref_mp))


# --- face quality ----------------------------------------------------------

def compute_face_quality(image_rgb: np.ndarray, faces: Sequence[Dict[str, Any]]) -> float:
    """Heuristic face-quality score in 0..1.

    Takes the highest-confidence face, multiplies its detector confidence by
    the normalised sharpness of its crop. Returns 0.0 if there are no faces.
    ``faces`` items must have ``bbox`` = [x, y, w, h] and ``score``.
    """
    if not faces:
        return 0.0
    best = max(faces, key=lambda f: float(f.get("score", 0.0)))
    conf = float(np.clip(best.get("score", 0.0), 0.0, 1.0))
    bbox = best.get("bbox") or [0, 0, 0, 0]
    x, y, w, h = (int(round(v)) for v in bbox[:4])
    H, W = image_rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + max(1, w)), min(H, y + max(1, h))
    if x1 <= x0 or y1 <= y0:
        return conf  # bbox unusable; fall back to confidence only
    crop = image_rgb[y0:y1, x0:x1]
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop
    sharp = normalize_sharpness(variance_of_laplacian(gray))
    # Blend so a confidently-detected but slightly soft face still scores well.
    return float(conf * (0.5 + 0.5 * sharp))


# --- keep score ------------------------------------------------------------

def keep_score(
    quality_score: Optional[float],
    face_quality: Optional[float],
    res_bonus: float,
    *,
    weight_iqa: float = 0.6,
    weight_face: float = 0.3,
    weight_resolution: float = 0.1,
    iqa_max: float = 100.0,
) -> float:
    """Combine IQA + face quality + resolution into a single keep score.

    ``quality_score`` is the raw MUSIQ value (0..100) and is normalised here.
    """
    iqa_norm = 0.0 if quality_score is None else float(np.clip(quality_score / iqa_max, 0.0, 1.0))
    fq = 0.0 if face_quality is None else float(np.clip(face_quality, 0.0, 1.0))
    rb = float(np.clip(res_bonus, 0.0, 1.0))
    return weight_iqa * iqa_norm + weight_face * fq + weight_resolution * rb


# --- embedding helpers -----------------------------------------------------

def embedding_to_blob(vec: np.ndarray) -> bytes:
    """Serialize an embedding to a float16 BLOB for SQLite storage."""
    return np.asarray(vec, dtype=np.float16).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a float16 BLOB back to a float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix for an (N, D) array."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    return unit @ unit.T
