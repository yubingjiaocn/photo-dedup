"""Motion-photo (live-photo) detection.

Xiaomi / Samsung / Google "motion photos" come in two shapes:

1. **Embedded** -- a single ``.jpg`` with an MP4 appended after the JPEG EOI,
   plus an XMP marker (``GCamera:MicroVideo`` / ``MotionPhoto``) in the header.
2. **Paired**   -- a ``.jpg`` next to a same-named ``.mp4`` / ``.MP`` /
   ``IMG_x.jpg.MP`` sidecar video.

The pipeline treats the video as an *attachment* that lives and dies with its
JPEG: if the JPEG is kept the video is kept; if the JPEG is deleted the video
goes too. ``file_kind`` records how each file participates:

    'jpg'         -> a plain still image
    'jpg_motion'  -> a still that owns a motion video (embedded or paired)
    'mp4_paired'  -> a video that is the partner of some jpg_motion
    'mp4_only'    -> a standalone video (real clip, not a live photo)

Embedded detection is cheap by default (header XMP markers only, a few hundred
KB read) to respect the HDD-sequential budget of stage 0. A full trailing-MP4
scan is available for small files / tests via ``full_scan_max_bytes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --- extension sets --------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXTS = {".mp4", ".mp", ".mov", ".m4v", ".3gp"}

# XMP / vendor markers written into the JPEG header for motion photos.
MOTION_XMP_MARKERS: Tuple[bytes, ...] = (
    b"MotionPhoto",
    b"MotionPhotoVersion",
    b"MicroVideo",
    b"MicroVideoOffset",
    b"GCamera:MicroVideo",
    b"GCamera",
    b"Item:Semantic",  # Container:Directory motion-photo layout
)

# ftyp "brand" signatures that indicate an appended MP4/QuickTime stream.
MP4_BRANDS: Tuple[bytes, ...] = (
    b"ftypmp4", b"ftypmp42", b"ftypisom", b"ftypiso2",
    b"ftypheic", b"ftypqt", b"ftypM4V", b"ftypavc1", b"ftyp3gp",
)

JPEG_EOI = b"\xff\xd9"


# --- basic type checks -----------------------------------------------------

def is_image(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


# --- embedded detection ----------------------------------------------------

def has_trailing_mp4(data: bytes) -> bool:
    """True if ``data`` looks like a JPEG with an MP4 appended after its EOI.

    We locate an ``ftyp`` box and require a JPEG EOI (``FFD9``) to appear before
    it -- that is the signature of a still image with a video stapled on the end
    (as opposed to a file that is *itself* an MP4, where ftyp sits at offset 0).
    """
    idx = data.find(b"ftyp")
    while idx != -1:
        # A real appended MP4 has ftyp at box offset 4 (after the size field),
        # never at the very start of the file.
        if idx >= 4 and data.rfind(JPEG_EOI, 0, idx) != -1:
            return True
        idx = data.find(b"ftyp", idx + 1)
    return False


def detect_embedded_motion(
    path: Path | str,
    *,
    head_bytes: int = 262_144,
    full_scan_max_bytes: int = 0,
) -> bool:
    """Detect whether a JPEG embeds a motion video.

    Strategy (cheapest first):
      1. Read the header (``head_bytes``) and look for vendor XMP markers.
         This is the authoritative, low-IO signal for Xiaomi/Samsung/Google.
      2. Optionally (``full_scan_max_bytes`` > 0 and file no larger than it)
         read the whole file and look for a trailing MP4 ftyp box.

    Returns ``False`` on any read error (caller logs / marks scan_error).
    """
    p = Path(path)
    if p.suffix.lower() not in {".jpg", ".jpeg"}:
        return False
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            head = fh.read(head_bytes)
            if any(marker in head for marker in MOTION_XMP_MARKERS):
                return True
            if full_scan_max_bytes and size <= full_scan_max_bytes:
                fh.seek(0)
                data = fh.read()
                return has_trailing_mp4(data)
    except OSError:
        return False
    return False


# --- paired detection ------------------------------------------------------

def _video_base_candidates(video_name: str) -> set[str]:
    """Base names a paired image could have, for a given video filename.

    Handles both ``IMG_x.mp4`` (-> ``img_x``) and ``IMG_x.jpg.MP``
    (-> ``img_x.jpg``) styles.
    """
    name_lower = video_name.lower()
    cands: set[str] = set()
    for ext in VIDEO_EXTS:
        if name_lower.endswith(ext):
            cands.add(name_lower[: -len(ext)])
    cands.add(Path(video_name).stem.lower())
    return cands


def find_image_for_video(video: Path, dir_files: Sequence[Path]) -> Optional[Path]:
    """Given a video, return the still image it pairs with (or None)."""
    cands = _video_base_candidates(video.name)
    matches = [other for other in dir_files if other != video and is_image(other)
               and (other.name.lower() in cands or other.stem.lower() in cands)]
    return matches[0] if len(matches) == 1 else None


def find_video_for_image(image: Path, dir_files: Sequence[Path]) -> Optional[Path]:
    """Given a still image, return a sidecar video that pairs with it (or None)."""
    matches = [other for other in dir_files if other != image and is_video(other)
               and find_image_for_video(other, dir_files) == image]
    return matches[0] if len(matches) == 1 else None


# --- unified classification ------------------------------------------------

def classify_file(
    path: Path,
    dir_files: Sequence[Path],
    *,
    embedded_head_bytes: int = 262_144,
    embedded_full_scan_max_bytes: int = 0,
) -> Tuple[str, Optional[Path]]:
    """Classify a single file into a ``file_kind`` (+ optional partner path).

    ``dir_files`` are the sibling entries in the same directory (used for
    paired detection). Returns ``(file_kind, partner_path_or_None)``.
    """
    p = Path(path)
    if is_image(p):
        if p.suffix.lower() in {".jpg", ".jpeg"} and detect_embedded_motion(
            p,
            head_bytes=embedded_head_bytes,
            full_scan_max_bytes=embedded_full_scan_max_bytes,
        ):
            return "jpg_motion", None  # video lives inside the same file
        partner = find_video_for_image(p, dir_files)
        if partner is not None:
            return "jpg_motion", partner
        return "jpg", None
    if is_video(p):
        partner = find_image_for_video(p, dir_files)
        if partner is not None:
            return "mp4_paired", partner
        return "mp4_only", None
    # Unknown extension -> treat as standalone (should be filtered by scan list).
    return "mp4_only", None


def build_dir_index(paths: Iterable[Path]) -> Dict[Path, List[Path]]:
    """Group an iterable of paths by parent directory (for classification)."""
    index: Dict[Path, List[Path]] = {}
    for p in paths:
        index.setdefault(p.parent, []).append(p)
    return index
