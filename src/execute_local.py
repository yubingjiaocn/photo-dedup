"""Stage 4 (local) -- move AUTO_REMOVE items to recoverable trash.

Safety-first defaults (from config.yaml):
  * ``execute.dry_run: true``  -> writes a plan, touches nothing.
  * ``execute.mode: move``     -> relocate into ``paths.trash/<timestamp>/`` and
    preserve the original folder tree, instead of hard-deleting.

Every real ``move`` run writes ``_undo_manifest.json`` into its trash session
folder, so ``--undo <session_dir>`` restores everything to its original path.

Usage
-----
    python -m src.execute_local                      # honor config (dry_run first!)
    python -m src.execute_local --no-dry-run         # actually move to _trash
    python -m src.execute_local --undo E:/Photos/_trash/2026-07-23_101500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import load_config
from . import db

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x

UNDO_MANIFEST = "_undo_manifest.json"


def _read_delete_list(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"delete list not found: {path} (run stage3 first)")
    out: List[Path] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Path(line))
    return out


def _verify_manifest(cfg, delete_list: Path, files: List[Path]) -> None:
    """Refuse stale/tampered manifests before any real filesystem mutation."""
    meta_path = cfg.output_dir / "delete_local.meta.json"
    if not meta_path.exists():
        raise RuntimeError("manifest metadata missing; rerun stage3")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(delete_list.read_bytes()).hexdigest()
    if digest != meta.get("paths_sha256") or len(files) != meta.get("count"):
        raise RuntimeError("delete manifest changed after stage3; refusing execution")
    conn = db.open_db(cfg.db_path)
    try:
        run_id = db.get_meta(conn, "stage2_run_id") or db.get_meta(conn, "stage2_done_at")
        groups = list(db.iter_groups(conn))
        policy = groups[0]["policy_version"] if groups else None
    finally:
        conn.close()
    if run_id != meta.get("stage2_run_id") or policy != meta.get("policy_version"):
        raise RuntimeError("delete manifest is stale for current DB policy/run; rerun stage3")


def _dest_for(src: Path, root: Path, session_dir: Path) -> Path:
    """Compute the trash destination, preserving the tree relative to root."""
    try:
        rel = src.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        # Outside the library (e.g. a sidecar elsewhere): keep drive letter safe.
        rel = Path("_external") / src.name
    return session_dir / rel


def move_files(files: List[Path], root: Path, trash_root: Path) -> dict:
    """Move files into a timestamped trash session, writing an undo manifest."""
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    session_dir = trash_root / stamp
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    moved = 0
    missing = 0
    for src in tqdm(files, desc="move", unit="file"):
        if not src.exists():
            missing += 1
            continue
        dest = _dest_for(src, root, session_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Avoid clobbering if two sources map to the same dest name.
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}__{moved}{dest.suffix}")
        shutil.move(str(src), str(dest))
        manifest[str(dest)] = str(src)
        moved += 1
    with open(session_dir / UNDO_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return {"moved": moved, "missing": missing, "session_dir": str(session_dir)}


def undo(session_dir: Path) -> dict:
    """Restore a previous move session from its undo manifest."""
    manifest_path = session_dir / UNDO_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"no undo manifest in {session_dir}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    restored = 0
    missing = 0
    for trashed, original in tqdm(manifest.items(), desc="undo", unit="file"):
        tp = Path(trashed)
        if not tp.exists():
            missing += 1
            continue
        op = Path(original)
        op.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tp), str(op))
        restored += 1
    return {"restored": restored, "missing": missing}


def run(
    config_path: Optional[str] = None,
    mode: Optional[str] = None,
    dry_run: Optional[bool] = None,
    undo_dir: Optional[str] = None,
) -> dict:
    cfg = load_config(config_path)

    if undo_dir:
        result = undo(Path(undo_dir))
        print(f"[execute] undo -> {result}")
        return result

    mode = mode or str(cfg.execute.get("mode", "move"))
    if dry_run is None:
        dry_run = bool(cfg.execute.get("dry_run", True))

    delete_list = cfg.output_dir / "delete_local.txt"
    files = _read_delete_list(delete_list)
    total_bytes = sum(f.stat().st_size for f in files if f.exists())
    gb = total_bytes / (1024 ** 3)

    print(f"[execute] mode={mode} dry_run={dry_run} files={len(files)} (~{gb:.2f} GB)")

    if dry_run:
        plan = cfg.output_dir / "execute_plan.txt"
        with open(plan, "w", encoding="utf-8") as fh:
            fh.write(f"# DRY RUN plan  mode={mode}  files={len(files)}  ~{gb:.2f} GB\n")
            for f in files:
                exists = "OK " if f.exists() else "MISS"
                fh.write(f"{exists}\t{f}\n")
        print(f"[execute] dry run -> wrote plan to {plan}. Re-run with --no-dry-run to apply.")
        return {"planned": len(files), "dry_run": True, "reclaim_gb": gb}

    _verify_manifest(cfg, delete_list, files)

    if mode == "move":
        result = move_files(files, cfg.root_path, cfg.trash_path)
    else:
        raise ValueError(f"unknown mode: {mode}")
    print(f"[execute] done -> {result}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4: execute local deletions")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default=None, choices=["move"],
                    help="automatic items may only be moved to recoverable trash")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false", default=None,
                    help="actually perform the move/delete")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                    help="force dry run (default from config)")
    ap.add_argument("--undo", dest="undo_dir", default=None,
                    help="restore a trash session directory")
    args = ap.parse_args(argv)
    run(config_path=args.config, mode=args.mode, dry_run=args.dry_run, undo_dir=args.undo_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
