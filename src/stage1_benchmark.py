"""Deterministic local benchmark: serial pipeline vs the optimised one.

What this is for
----------------
Proving the *mechanism* on any machine, including CI: the same synthetic library
is processed twice, once with ``cpu_workers=0``/``prefetch_batches=0`` (the
pre-change pipeline, exactly) and once with the optimised defaults, and the two
runs are compared on three axes:

1. **Identical output.** Every feature row must match byte for byte. A faster
   pipeline that changes results is not faster, it is broken.
2. **Fewer model calls.** Reported as counters, so the batching is a measurement
   rather than a claim.
3. **Timing.** Printed, never asserted. On a machine with no GPU the stub backend
   does microseconds of "inference", so there is nothing for prefetch to hide
   behind and the threaded run can legitimately be *slower* than the serial one —
   asserting a speedup here would be asserting a lie. The real numbers live in
   ``docs/STAGE1_THROUGHPUT.md``, measured with the actual weights.

Use ``--sizes`` to make the CPU work realistic; the default is small so the test
suite can run this in a second.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

from . import db, stage0_inventory, stage1_features

# Deliberately mixed orientations and one odd size: that is what makes the IQA
# lane form more than one shape group, which is the interesting case.
DEFAULT_SIZES: Tuple[Tuple[int, int], ...] = (
    (640, 480), (480, 640), (640, 480), (512, 512), (480, 640), (640, 480),
)


def build_library(root: Path, count: int,
                  sizes: Sequence[Tuple[int, int]] = DEFAULT_SIZES) -> List[Path]:
    """Deterministic synthetic JPEGs: same bytes on every machine and run."""
    root.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        path = root / f"IMG_2026{index // 100 + 1:02d}{index % 100:02d}_120000.jpg"
        # A smooth gradient plus a per-image offset: cheap to generate, and every
        # image differs, so per-image scores are distinguishable.
        image = Image.new("RGB", (width, height))
        pixels = image.load()
        for y in range(height):
            for x in range(0, width, 8):
                value = (x + y + index * 17) % 256
                for offset in range(min(8, width - x)):
                    pixels[x + offset, y] = (value, (value * 3) % 256, (value * 7) % 256)
        image.save(path, "JPEG", quality=88)
        made.append(path)
    return made


def write_config(directory: Path, root: Path, *, cpu_workers: int,
                 prefetch_batches: int, batch_size: int, iqa_batch_size: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(directory / "inventory.sqlite"),
                  "output_dir": str(directory / "out"),
                  "models_dir": str(directory / "models"),
                  "trash": str(directory / "trash")},
        "features": {
            "backend": "stub", "batch_size": batch_size,
            "cpu_workers": cpu_workers, "prefetch_batches": prefetch_batches,
            "iqa_batch_size": iqa_batch_size,
            "thumbnails": {"enabled": True, "max_px": 128},
            "telemetry": {"enabled": True, "gpu_event_every": 0},
        },
    }), encoding="utf-8")
    return path


def feature_fingerprint(db_path: Path) -> List[Dict[str, Any]]:
    """Every persisted feature column, ordered, for an exact comparison."""
    conn = db.open_db(db_path)
    rows = conn.execute(
        "SELECT file_id, phash, content_sha256, dinov2_embedding, quality_score, "
        "quality_meta, face_count, faces_json, status FROM features ORDER BY file_id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def run_variant(directory: Path, root: Path, *, cpu_workers: int, prefetch_batches: int,
                batch_size: int, iqa_batch_size: int,
                backend: str = "stub") -> Dict[str, Any]:
    """One full Stage 0 + Stage 1 pass into a fresh output directory."""
    shutil.rmtree(directory, ignore_errors=True)
    config = write_config(directory, root, cpu_workers=cpu_workers,
                          prefetch_batches=prefetch_batches, batch_size=batch_size,
                          iqa_batch_size=iqa_batch_size)
    stage0_inventory.run(config_path=str(config))
    started = time.perf_counter()
    stats = stage1_features.run(config_path=str(config), backend_override=backend)
    wall = time.perf_counter() - started
    snapshot = stats["phase_telemetry"]
    return {
        "cpu_workers": cpu_workers,
        "prefetch_batches": prefetch_batches,
        "stage1_wall_seconds": wall,
        "loop_seconds": snapshot["loop_seconds"],
        "wait_seconds": snapshot.get("wait_accounted_seconds"),
        "worker_seconds": snapshot.get("worker_seconds"),
        "overlap_seconds": snapshot.get("overlap_seconds"),
        "counters": dict(snapshot.get("counters") or {}),
        "processed": stats["processed"],
        "failed": stats["failed"],
        "batches": snapshot["batches"],
        "fingerprint": feature_fingerprint(directory / "inventory.sqlite"),
        "loop_settings": stats["loop_settings"],
    }


_STUB_NOTE = (
    "Timing on the stub backend is indicative only: the stub's 'inference' takes "
    "microseconds, so there is no GPU work for the producer to hide behind and the "
    "threaded run may be slower here. Real-weight numbers are in "
    "docs/STAGE1_THROUGHPUT.md."
)
_TORCH_NOTE = (
    "Timing is from the real model stack on this machine. It reflects this GPU, "
    "this CPU and this disk; a different box (especially one reading from a "
    "mechanical disk) will differ. Features were compared for equality regardless."
)


def compare(serial: Dict[str, Any], optimised: Dict[str, Any],
            backend: str = "stub") -> Dict[str, Any]:
    """Identical results, model-call counts, and honest timing."""
    identical = serial["fingerprint"] == optimised["fingerprint"]
    mismatches: List[int] = []
    if not identical:
        for left, right in zip(serial["fingerprint"], optimised["fingerprint"]):
            if left != right:
                mismatches.append(int(left["file_id"]))
    speedup = (serial["stage1_wall_seconds"] / optimised["stage1_wall_seconds"]
               if optimised["stage1_wall_seconds"] > 0 else None)
    return {
        "identical_features": identical,
        "mismatched_file_ids": mismatches,
        "processed": (serial["processed"], optimised["processed"]),
        "backend": backend,
        "serial_wall_seconds": serial["stage1_wall_seconds"],
        "optimised_wall_seconds": optimised["stage1_wall_seconds"],
        "wall_speedup": speedup,
        "serial_counters": serial["counters"],
        "optimised_counters": optimised["counters"],
        # The caveat must describe what actually ran, not what usually runs.
        "note": _STUB_NOTE if backend == "stub" else _TORCH_NOTE,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the serial Stage 1 pipeline with the optimised one")
    parser.add_argument("--images", type=int, default=24, help="synthetic images")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iqa-batch-size", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--backend", default="stub", choices=["stub", "torch"])
    parser.add_argument("--workdir", default=None,
                        help="scratch directory (default: a temp dir, removed after)")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write the comparison to this file")
    args = parser.parse_args(argv)

    import tempfile

    temporary = args.workdir is None
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(
        prefix="stage1-benchmark-"))
    try:
        root = workdir / "photos"
        print(f"[benchmark] building {args.images} synthetic images in {root}")
        build_library(root, args.images)
        print("[benchmark] serial run (cpu_workers=0, prefetch_batches=0)")
        serial = run_variant(workdir / "serial", root, cpu_workers=0, prefetch_batches=0,
                             batch_size=args.batch_size, iqa_batch_size=args.iqa_batch_size,
                             backend=args.backend)
        print(f"[benchmark] optimised run (cpu_workers={args.cpu_workers}, "
              f"prefetch_batches={args.prefetch_batches})")
        optimised = run_variant(workdir / "optimised", root,
                                cpu_workers=args.cpu_workers,
                                prefetch_batches=args.prefetch_batches,
                                batch_size=args.batch_size,
                                iqa_batch_size=args.iqa_batch_size, backend=args.backend)
        result = compare(serial, optimised, backend=args.backend)
        print("\n=== Stage 1 benchmark ===")
        print(f"backend                  : {args.backend}")
        print(f"identical features       : {result['identical_features']}")
        print(f"processed (serial, opt)  : {result['processed']}")
        print(f"serial wall              : {result['serial_wall_seconds']:.2f}s")
        print(f"optimised wall           : {result['optimised_wall_seconds']:.2f}s")
        if result["wall_speedup"] is not None:
            print(f"wall speedup             : {result['wall_speedup']:.2f}x")
        print(f"serial counters          : {result['serial_counters']}")
        print(f"optimised counters       : {result['optimised_counters']}")
        print(f"producer CPU seconds     : {optimised['worker_seconds']}")
        print(f"loop waited for producer : {optimised['wait_seconds']}s")
        print(f"overlapped (hidden) work : {optimised['overlap_seconds']}s")
        print(f"\nnote: {result['note']}")
        if args.json_path:
            payload = {k: v for k, v in result.items()}
            Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[benchmark] wrote {args.json_path}")
        return 0 if result["identical_features"] else 1
    finally:
        if temporary:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
