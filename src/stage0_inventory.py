"""Stage 0 -- filesystem inventory.

Walks the photo library once (sequential, HDD-friendly), and for every image
or video records: path, size, mtime, EXIF capture time, dimensions, and its
motion-photo classification (embedded / paired). Everything lands in the
``files`` table. Re-running is idempotent (``INSERT OR IGNORE`` on path) and
commits every ``scan.commit_every`` files so a Ctrl+C resumes cleanly.

Only header bytes are read per file (dimensions + EXIF + motion XMP marker),
so a 66k-file library costs one pass of small reads, not a full 451GB read.
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
from typing import Dict, List, Optional, Tuple

from .config import Config, load_config
from . import db
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
    except Exception:
        pass
    return result


# --- per-file record -------------------------------------------------------

def build_file_meta(
    path: Path, dir_files: List[Path], cfg: Config
) -> Tuple[Dict[str, object], Optional[Path]]:
    """Assemble the ``files`` row dict + partner path for one path (single pass)."""
    st = path.stat()
    kind, partner = mp.classify_file(
        path,
        dir_files,
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


def run(config_path: Optional[str] = None, root_override: Optional[str] = None) -> Dict[str, int]:
    """Run stage 0. Returns stats dict."""
    cfg = load_config(config_path)
    root = Path(root_override) if root_override else cfg.root_path
    if not root.exists():
        raise FileNotFoundError(f"scan root does not exist: {root}")

    conn = db.open_db(cfg.db_path)
    extensions = {e.lower() for e in cfg.scan.get("extensions", [])}
    commit_every = int(cfg.scan.get("commit_every", 100))
    follow = bool(cfg.scan.get("follow_symlinks", False))

    inserted = 0
    since_commit = 0
    t0 = time.time()

    # partner_links: (file_path, partner_path) collected per directory
    for dirpath, _dirs, filenames in tqdm(os.walk(root, followlinks=follow), desc="inventory", unit="dir"):
        dir_path = Path(dirpath)
        all_entries = [dir_path / name for name in filenames]
        wanted = [p for p in all_entries if _wanted(p, extensions)]
        if not wanted:
            continue

        path_to_id: Dict[Path, int] = {}
        partner_of: Dict[Path, Optional[Path]] = {}
        for p in wanted:
            try:
                meta, partner = build_file_meta(p, all_entries, cfg)
            except OSError as exc:
                print(f"[stage0][WARN] stat failed {p}: {exc}")
                continue
            fid = db.insert_file(conn, meta)
            path_to_id[p] = fid
            partner_of[p] = partner
            inserted += 1
            since_commit += 1
            if since_commit >= commit_every:
                conn.commit()
                since_commit = 0

        # Link motion partners now that every file in the dir has an id.
        for p, partner in partner_of.items():
            if partner is None:
                continue
            fid = path_to_id.get(p)
            pid = path_to_id.get(Path(partner))
            if fid and pid:
                db.set_motion_partner(conn, fid, pid)
    conn.commit()

    db.set_meta(conn, "stage0_root", str(root))
    db.set_meta(conn, "stage0_done_at", str(int(time.time())))
    dt = time.time() - t0
    stats = {
        "files": db.count_files(conn),
        "jpg": db.count_files(conn, "file_kind='jpg'"),
        "jpg_motion": db.count_files(conn, "file_kind='jpg_motion'"),
        "mp4_paired": db.count_files(conn, "file_kind='mp4_paired'"),
        "mp4_only": db.count_files(conn, "file_kind='mp4_only'"),
    }
    print(f"[stage0] inventoried {inserted} files in {dt:.1f}s -> {stats}")
    conn.close()
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0: filesystem inventory")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None, help="override paths.root")
    args = ap.parse_args(argv)
    run(config_path=args.config, root_override=args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
