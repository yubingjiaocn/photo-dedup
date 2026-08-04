"""Motion-photo (live-photo) detection.

The supported Xiaomi / Samsung / Google layout is **embedded**: a single
``.jpg`` with an MP4 appended after the JPEG EOI plus an XMP marker
(``GCamera:MicroVideo`` / ``MotionPhoto``) in the header. Separate MP4/MOV
files are standalone videos and are never paired by filename. ``file_kind``:

    'jpg'         -> a plain still image
    'jpg_motion'  -> a still with an embedded motion video
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


# --- unified classification ------------------------------------------------

def classify_file(
    path: Path,
    dir_files: Sequence[Path],
    *,
    embedded_head_bytes: int = 262_144,
    embedded_full_scan_max_bytes: int = 0,
) -> Tuple[str, Optional[Path]]:
    """Classify a single file into a ``file_kind`` (+ optional partner path).

    ``dir_files`` is retained for API compatibility but deliberately ignored:
    filename-based JPG/video sidecar pairing is unsupported.
    """
    p = Path(path)
    if is_image(p):
        if p.suffix.lower() in {".jpg", ".jpeg"} and detect_embedded_motion(
            p,
            head_bytes=embedded_head_bytes,
            full_scan_max_bytes=embedded_full_scan_max_bytes,
        ):
            return "jpg_motion", None  # video lives inside the same file
        return "jpg", None
    if is_video(p):
        return "mp4_only", None
    # Unknown extension -> treat as standalone (should be filtered by scan list).
    return "mp4_only", None


def build_dir_index(paths: Iterable[Path]) -> Dict[Path, List[Path]]:
    """Group an iterable of paths by parent directory (for classification)."""
    index: Dict[Path, List[Path]] = {}
    for p in paths:
        index.setdefault(p.parent, []).append(p)
    return index
