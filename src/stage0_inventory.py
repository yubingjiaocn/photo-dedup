"""Stage 0 -- filesystem inventory.

Walks the photo library once (sequential, HDD-friendly), and for every image
or video records: path, size, mtime, EXIF capture time, dimensions, and its
embedded motion-photo classification. Everything lands in the
``files`` table. Re-running is idempotent (``INSERT OR IGNORE`` on path) and
commits every ``scan.commit_every`` metadata updates. Re-runs stat existing
rows first and skip header/EXIF reads when size + mtime are unchanged, so a
limited run resumes with the next new/changed files instead of redoing a prefix.

Only header bytes are read per processed file (dimensions + EXIF + motion XMP
marker). With ``--limit``, the remaining directory entries are still counted
without opening them so later size/ETA projections use the real library scope.

Root identity: the first stage0 run binds this output directory/database to the
normalised photo root (see :mod:`src.root_scope`). Re-running with the same root
resumes; a different root fails closed with ``ParameterError`` *before* any row
is written, because mixing two roots in one output directory silently corrupts
every later count and report.
"""

from __future__ import annotations

import argparse
import calendar
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from os import stat_result
from typing import Dict, List, Optional, Tuple

from .config import Config, load_config
from . import db
from . import root_scope
from . import motion_photo as mp

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x

# IMG_20260204_194535.jpg  /  20260204_194535  /  VID_20260204_194535
_FILENAME_TS = re.compile(r"(20\d{6})[_-]?(\d{6})")


# --- EXIF / dimensions -----------------------------------------------------

def _parse_exif_datetime(value: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse an EXIF 'YYYY:MM:DD HH:MM:SS' string to (iso_str, unix_ts)."""
    value = value.strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            iso = dt.strftime("%Y-%m-%d %H:%M:%S")
            ts = calendar.timegm(dt.timetuple())  # tz-agnostic, consistent
            return iso, int(ts)
        except ValueError:
            continue
    return None, None


def _timestamp_from_name(basename: str) -> Tuple[Optional[str], Optional[int]]:
    m = _FILENAME_TS.search(basename)
    if not m:
        return None, None
    date_part, time_part = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
    except ValueError:
        return None, None
    return dt.strftime("%Y-%m-%d %H:%M:%S"), int(calendar.timegm(dt.timetuple()))


def read_image_header(path: Path) -> Dict[str, object]:
    """Read dimensions + EXIF DateTimeOriginal from a still image (header only)."""
    result: Dict[str, object] = {
        "width": None, "height": None, "exif_datetime": None, "exif_timestamp": None,
    }
    try:
        from PIL import Image

        # Header inspection does not allocate the pixel raster. Disable Pillow's
        # decompression-bomb gate only for this single-threaded header read so a
        # 200 MP image still records dimensions and can be rejected by Stage 1
        # before any full decode. The global is restored immediately.
        old_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(path) as img:
                result["width"], result["height"] = img.size
                exif = None
                try:
                    exif = img.getexif()
                except Exception:
                    exif = None
                if exif:
                    # 36867 = DateTimeOriginal, 306 = DateTime
                    dt_val = exif.get(36867) or exif.get(306)
                    if dt_val:
                        iso, ts = _parse_exif_datetime(str(dt_val))
                        result["exif_datetime"], result["exif_timestamp"] = iso, ts
        finally:
            Image.MAX_IMAGE_PIXELS = old_limit
    except Exception:
        pass
    return result


# --- per-file record -------------------------------------------------------

def build_file_meta(
    path: Path, dir_files: List[Path], cfg: Config, st: Optional[stat_result] = None,
) -> Tuple[Dict[str, object], Optional[Path]]:
    """Assemble the ``files`` row dict + partner path for one path (single pass)."""
    st = st or path.stat()
    kind, partner = mp.classify_file(
        path, dir_files,
        embedded_head_bytes=int(cfg.scan.get("embedded_head_bytes", 262144)),
        embedded_full_scan_max_bytes=int(cfg.scan.get("embedded_full_scan_max_bytes", 0)),
    )
    meta: Dict[str, object] = {
        "path": str(path),
        "basename": path.name,
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "file_kind": kind,
        "scan_status": "done",
    }
    if mp.is_image(path) and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        header = read_image_header(path)
        meta.update(header)
    if not meta.get("exif_datetime"):
        iso, ts = _timestamp_from_name(path.name)
        meta["exif_datetime"], meta["exif_timestamp"] = iso, ts
    # Final fallback: file mtime so time-window clustering still has a value.
    if not meta.get("exif_timestamp"):
        meta["exif_timestamp"] = int(st.st_mtime_ns // 1_000_000_000)
    return meta, partner


def _wanted(path: Path, extensions: set[str]) -> bool:
    """Honor the configured allow-list exactly; unsupported formats stay out."""
    return path.suffix.lower() in extensions


def run(
    config_path: Optional[str] = None,
    root_override: Optional[str] = None,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, int]:
    """Run stage 0. Returns stats dict."""
    cfg = load_config(config_path)
    root = Path(root_override) if root_override else cfg.root_path
    if not root.exists():
        raise FileNotFoundError(f"scan root does not exist: {root}")

    # Fail closed on an incompatible output directory before creating/upgrading
    # anything in it.
    root_scope.preflight(cfg.db_path, root)
    conn = db.open_db(cfg.db_path)
    scope = root_scope.bind(conn, root, run_id=run_id, db_path=str(cfg.db_path))
    print(f"[stage0] {root_scope.scope_note(root_scope.summary(conn, scope))}")
    # Older builds inferred same-name JPEG/video sidecars. This library only
    # supports embedded Motion JPEG, so remove those stale links without
    # reopening either source file. Embedded rows never had a partner id.
    scope_clause, scope_params = db.scope_sql(scope, "files")
    conn.execute(
        "UPDATE files SET file_kind='jpg', motion_partner_id=NULL "
        f"WHERE file_kind='jpg_motion' AND motion_partner_id IS NOT NULL AND {scope_clause}",
        scope_params,
    )
    conn.execute(
        "UPDATE files SET file_kind='mp4_only', motion_partner_id=NULL "
        f"WHERE (file_kind='mp4_paired' OR motion_partner_id IS NOT NULL) AND {scope_clause}",
        scope_params,
    )
    conn.commit()
    extensions = {e.lower() for e in cfg.scan.get("extensions", [])}
    commit_every = int(cfg.scan.get("commit_every", 100))
    follow = bool(cfg.scan.get("follow_symlinks", False))

    metadata_processed = 0
    scanned = 0
    unchanged = 0
    refreshed = 0
    discovered_files = 0
    discovered_still_images = 0
    since_commit = 0
    seen_ids: List[int] = []
    t0 = time.time()

    # File-level progress stays visibly alive even inside one huge directory.
    # ``limit`` is an expensive metadata-work budget, not a prefix sample:
    # unchanged rows do not consume it, so repeated runs continue forward.
    progress = tqdm(desc="inventory", unit="file")
    for dirpath, _dirs, filenames in os.walk(root, followlinks=follow):
        dir_path = Path(dirpath)
        all_entries = [dir_path / name for name in filenames]
        all_wanted = sorted(p for p in all_entries if _wanted(p, extensions))
        discovered_files += len(all_wanted)
        discovered_still_images += sum(1 for p in all_wanted if mp.is_image(p))
        if not all_wanted:
            continue

        for p in all_wanted:
            scanned += 1
            progress.update(1)
            if limit is not None and metadata_processed >= limit:
                continue
            try:
                st = p.stat()
            except OSError as exc:
                print(f"[stage0][WARN] stat failed {p}: {exc}")
                continue
            existing = db.get_file_by_path(conn, str(p))
            if (existing is not None
                    and existing["size_bytes"] == st.st_size
                    and existing["mtime_ns"] == st.st_mtime_ns):
                unchanged += 1
                seen_ids.append(int(existing["id"]))
                continue
            try:
                meta, _partner = build_file_meta(p, all_entries, cfg, st=st)
            except OSError as exc:
                print(f"[stage0][WARN] metadata failed {p}: {exc}")
                continue
            fid = db.insert_file(conn, meta, scope=scope)
            seen_ids.append(int(fid))
            # A re-scan must notice a replaced/edited photo, otherwise stale
            # hashes and stale thumbnails would survive (see refresh_file_identity).
            if db.refresh_file_identity(conn, fid, meta):
                refreshed += 1
            metadata_processed += 1
            since_commit += 1
            progress.set_postfix(
                metadata=metadata_processed, unchanged=unchanged, refresh=False
            )
            if since_commit >= commit_every:
                root_scope.mark_seen(conn, scope.run_id, seen_ids)
                seen_ids.clear()
                conn.commit()
                since_commit = 0
    progress.close()
    root_scope.mark_seen(conn, scope.run_id, seen_ids)
    conn.commit()

    db.set_meta(conn, "stage0_root", str(root))
    db.set_meta(conn, "stage0_done_at", str(int(time.time())))
    dt = time.time() - t0
    scope_summary = root_scope.summary(conn, scope)
    stats = {
        "files": db.count_files(conn, scope=scope),
        "still_images": db.count_still_images(conn, scope=scope),
        "discovered_files": discovered_files,
        "library_still_images": discovered_still_images,
        "total_bytes": int(conn.execute(
            f"SELECT COALESCE(SUM(files.size_bytes), 0) FROM files WHERE {scope_clause}",
            scope_params).fetchone()[0]),
        "changed_files": refreshed,
        "jpg": db.count_files(conn, "file_kind='jpg'", scope=scope),
        "jpg_motion": db.count_files(conn, "file_kind='jpg_motion'", scope=scope),
        "mp4_paired": db.count_files(conn, "file_kind='mp4_paired'", scope=scope),
        "mp4_only": db.count_files(conn, "file_kind='mp4_only'", scope=scope),
        "scope": scope_summary,
    }
    if refreshed:
        print(f"[stage0] {refreshed} file(s) changed on disk; their cached features and "
              "thumbnails were invalidated and will be recomputed")
    print(
        f"[stage0] scanned {scanned} files, refreshed metadata for "
        f"{metadata_processed} ({unchanged} unchanged) in {dt:.1f}s -> {stats}"
    )
    conn.close()
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0: filesystem inventory")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None, help="override paths.root")
    ap.add_argument("--limit", type=int, default=None, help="inventory at most N files")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    run(config_path=args.config, root_override=args.root, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
