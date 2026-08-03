"""Stage 1 thumbnail cache -- derived from the one decode Stage 1 already does.

Design rules (they are the whole point of this module):

* **No extra source read.** :meth:`Thumbnailer.capture` accepts the decoded
  ``PIL.Image`` object Stage 1 already holds for embeddings/IQA/faces. It never
  receives a path, never calls ``Image.open``, and never touches the HDD.
* **Downscale without mutating the shared image.** ``Image.thumbnail()`` works
  in place, which would corrupt the object the other extractors still use, so
  we allocate a small destination with ``resize`` instead (also avoids copying
  a full-resolution 50 MP frame).
* **Atomic writes.** Each JPEG lands via a temporary file plus ``os.replace``,
  so a crash never leaves a half-written thumbnail that later looks valid.
* **Honest failures.** A failure is recorded as ``status='error'`` with the
  reason. Nothing is written to disk and nothing pretends to have succeeded.
* **Stale-safe reuse.** A cached thumbnail counts as current only when the DB
  row matches the source ``size_bytes``/``mtime_ns`` *and* the configured pixel
  size, *and* the JPEG still exists on the SSD.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_MAX_PX = 320
DEFAULT_JPEG_QUALITY = 80
DIR_NAME = "thumbs"
SUFFIX = ".jpg"
STATUS_OK = "ok"
STATUS_ERROR = "error"
SOURCE_DECODE_FAILED = "SOURCE_DECODE_FAILED"


def thumbs_dir(output_dir: str | os.PathLike) -> Path:
    """Thumbnail cache directory inside the (SSD) output directory."""
    return Path(output_dir) / DIR_NAME


def thumb_path(directory: str | os.PathLike, file_id: int) -> Path:
    """Stable, path-free name: the inventory file id."""
    return Path(directory) / f"{int(file_id)}{SUFFIX}"


def row_value(row: Any, key: str, default: Any = None) -> Any:
    """Read a column from a ``sqlite3.Row`` (no ``.get``) or a plain mapping."""
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        keys = row.keys()
    except AttributeError:
        return default
    return row[key] if key in keys else default


def render_thumbnail(image: Any, max_px: int = DEFAULT_MAX_PX) -> Any:
    """Return a new bounded RGB image; the input object is left untouched."""
    from PIL import Image

    if max_px < 1:
        raise ValueError("max_px must be at least 1")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError(f"degenerate image size: {image.size}")
    scale = min(1.0, float(max_px) / float(max(width, height)))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = image.resize(target, Image.LANCZOS, reducing_gap=2.0)
    return small if small.mode == "RGB" else small.convert("RGB")


def save_atomic(image: Any, destination: Path, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> int:
    """Write one JPEG atomically. Returns bytes written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            image.save(handle, format="JPEG", quality=int(jpeg_quality), optimize=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.stat().st_size


def directory_usage(directory: str | os.PathLike) -> Tuple[int, int]:
    """(file count, bytes) of the cache directory. Missing directory -> (0, 0)."""
    files = 0
    total = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.endswith(SUFFIX):
                    continue
                try:
                    if entry.is_file():
                        files += 1
                        total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0, 0
    return files, total


def ids_on_disk(directory: str | os.PathLike) -> Set[int]:
    """File ids that currently have a JPEG in the cache directory."""
    found: Set[int] = set()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                stem, _, extension = entry.name.rpartition(".")
                if extension == SUFFIX.lstrip(".") and stem.isdigit():
                    found.add(int(stem))
    except OSError:
        return found
    return found


def estimate_total_bytes(still_images: int, average_bytes: Optional[float]) -> Optional[int]:
    """Rough cache size for the whole library from measured samples."""
    if average_bytes is None or still_images < 0:
        return None
    return int(round(float(average_bytes) * int(still_images)))


class Thumbnailer:
    """Generates and books-keeps the SSD thumbnail cache during Stage 1."""

    def __init__(
        self,
        directory: str | os.PathLike,
        max_px: int = DEFAULT_MAX_PX,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> None:
        if int(max_px) < 1:
            raise ValueError("thumbnails.max_px must be at least 1")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("thumbnails.jpeg_quality must be within 1..100")
        self.directory = Path(directory)
        self.max_px = int(max_px)
        self.jpeg_quality = int(jpeg_quality)
        self.created = 0
        self.reused = 0
        self.failed = 0
        self.bytes_created = 0
        self.failures: List[Dict[str, Any]] = []
        self._pending: List[Dict[str, Any]] = []

    # -- staleness ----------------------------------------------------------
    def is_current(self, row: Any) -> bool:
        """True only when the cached JPEG provably matches this source file."""
        if row_value(row, "thumb_status") != STATUS_OK:
            return False
        if row_value(row, "thumb_max_px") != self.max_px:
            return False
        for cached, live in (
            ("thumb_source_size_bytes", "size_bytes"),
            ("thumb_source_mtime_ns", "mtime_ns"),
        ):
            cached_value = row_value(row, cached)
            live_value = row_value(row, live)
            if cached_value is None or live_value is None or cached_value != live_value:
                return False
        return thumb_path(self.directory, int(row_value(row, "id"))).is_file()

    def missing_on_disk(self, recorded_ids: Iterable[int]) -> List[int]:
        """Recorded-as-ok ids whose JPEG has disappeared from the SSD."""
        present = ids_on_disk(self.directory)
        return sorted(int(i) for i in recorded_ids if int(i) not in present)

    # -- generation ---------------------------------------------------------
    def capture(self, row: Any, image: Any) -> Optional[Dict[str, Any]]:
        """Create the thumbnail for ``row`` from an already-decoded ``image``."""
        file_id = int(row_value(row, "id"))
        if self.is_current(row):
            self.reused += 1
            return None
        destination = thumb_path(self.directory, file_id)
        try:
            small = render_thumbnail(image, self.max_px)
            written = save_atomic(small, destination, self.jpeg_quality)
        except (OSError, ValueError) as exc:
            return self._record_failure(row, f"{type(exc).__name__}: {exc}")
        self.created += 1
        self.bytes_created += written
        record = {
            "file_id": file_id,
            "status": STATUS_OK,
            "max_px": self.max_px,
            "bytes": written,
            "source_size_bytes": row_value(row, "size_bytes"),
            "source_mtime_ns": row_value(row, "mtime_ns"),
            "error": None,
            "created_at": int(time.time()),
        }
        self._pending.append(record)
        return record

    def record_source_failure(self, row: Any, reason: str = SOURCE_DECODE_FAILED) -> Dict[str, Any]:
        """Record that no thumbnail exists because the source could not be decoded."""
        return self._record_failure(row, reason)

    def _record_failure(self, row: Any, error: str) -> Dict[str, Any]:
        file_id = int(row_value(row, "id"))
        self.failed += 1
        record = {
            "file_id": file_id,
            "status": STATUS_ERROR,
            "max_px": self.max_px,
            "bytes": None,
            "source_size_bytes": row_value(row, "size_bytes"),
            "source_mtime_ns": row_value(row, "mtime_ns"),
            "error": str(error)[:500],
            "created_at": int(time.time()),
        }
        self.failures.append({"file_id": file_id, "error": record["error"]})
        self._pending.append(record)
        print(f"[stage1][WARN] thumbnail failed for file_id={file_id}: {record['error']}")
        return record

    # -- bookkeeping --------------------------------------------------------
    def drain(self) -> List[Dict[str, Any]]:
        """Hand accumulated rows to the caller for a single DB write."""
        rows, self._pending = self._pending, []
        return rows

    @property
    def average_bytes(self) -> Optional[float]:
        return self.bytes_created / self.created if self.created else None

    def stats(self) -> Dict[str, Any]:
        files, total_bytes = directory_usage(self.directory)
        return {
            "enabled": True,
            "directory": str(self.directory),
            "max_px": self.max_px,
            "jpeg_quality": self.jpeg_quality,
            "created": self.created,
            "reused": self.reused,
            "failed": self.failed,
            "bytes_created": self.bytes_created,
            "average_bytes": self.average_bytes,
            "cache_files": files,
            "cache_bytes": total_bytes,
            "failures_sample": self.failures[:20],
        }


def disabled_stats() -> Dict[str, Any]:
    """Stats payload when thumbnail generation is switched off in config."""
    return {"enabled": False, "created": 0, "reused": 0, "failed": 0,
            "cache_files": 0, "cache_bytes": 0, "average_bytes": None}
