"""Observed pipeline performance, thumbnail disk accounting, and rough ETA.

Everything here is measured or derived from *this* run:

* wall time per stage (0/1/2/3) plus the total,
* Stage 0 files/second and Stage 1 images/second,
* a **clearly labelled rough linear** ETA for the whole library, scaled from
  the inventory this run actually saw and the images this run actually
  processed. No hardcoded library size (the old "1 TiB" projection was
  misleading); when there is no sample, the ETA is reported as ``unknown``.
* thumbnail cache: estimated total, actual current usage, and SSD free space.
  Below the advisory floor we warn and keep going -- never abort, and never
  demand tens of GB of headroom.
* the Stage 1 phase breakdown from :mod:`src.stage1_telemetry` (where the wall
  clock actually went: HDD read, decode, GPU inference, thumbnail, DB), so a
  "GPU only pulses to 81%" report can be answered with numbers.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from . import stage1_telemetry

GIB = 1024 ** 3
LOW_SPACE_WARN_GIB = 20.0
STAGES = ("stage0", "stage1", "stage2", "stage3")


def _rate(count: float, seconds: float) -> Optional[float]:
    return (count / seconds) if seconds > 0 else None


def _fmt_rate(value: Optional[float], unit: str) -> str:
    return f"{value:.2f} {unit}" if value is not None else f"unknown {unit}"


def _fmt_gib(value: Optional[float]) -> str:
    return f"{value:.2f} GiB" if value is not None else "unknown"


def _scope_lines(scope: Optional[Dict[str, Any]]) -> list[str]:
    """State which root/run the numbers below describe (empty when unbound)."""
    if not scope or not scope.get("root"):
        return []
    from . import root_scope

    return [root_scope.scope_note(scope)]


def free_bytes(path: Path) -> Optional[int]:
    """Free space on the volume holding ``path`` (its nearest existing parent)."""
    probe = Path(path)
    for candidate in (probe, *probe.parents):
        try:
            return int(shutil.disk_usage(candidate).free)
        except OSError:
            continue
    return None


def thumbnail_disk_report(
    output_dir: Path,
    still_images: int,
    thumb_stats: Optional[Dict[str, Any]] = None,
    warn_free_gib: float = LOW_SPACE_WARN_GIB,
) -> Dict[str, Any]:
    """Estimated vs actual thumbnail footprint and remaining SSD space.

    Actual usage is always measured from the cache directory (the only source of
    truth); ``thumb_stats`` merely supplies a better per-file average when this
    run wrote thumbnails.
    """
    from . import thumbnails

    stats = thumb_stats or {}
    directory = thumbnails.thumbs_dir(output_dir)
    cache_files, cache_bytes = thumbnails.directory_usage(directory)
    average = stats.get("average_bytes")
    if average is None and cache_files:
        average = cache_bytes / cache_files
    estimated = thumbnails.estimate_total_bytes(still_images, average)
    remaining = max(0, int(still_images) - cache_files)
    remaining_estimate = (
        int(round(float(average) * remaining)) if average is not None else None
    )
    free = free_bytes(directory)
    return {
        "directory": str(directory),
        "still_images": int(still_images),
        "cached_files": cache_files,
        "actual_bytes": cache_bytes,
        "average_bytes": average,
        "estimated_total_bytes": estimated,
        "estimated_remaining_bytes": remaining_estimate,
        "free_bytes": free,
        "warn_free_gib": float(warn_free_gib),
        "low_space": (free is not None and free < float(warn_free_gib) * GIB),
        "basis": ("measured sample average" if average is not None
                  else "unknown (no thumbnail written yet)"),
    }


def build_performance(
    timings: Dict[str, float],
    inventory: Dict[str, Any],
    features: Dict[str, Any],
    library_still_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble observed timings/throughput plus the rough linear ETA."""
    raw_seconds = {name: float(timings.get(name, 0.0)) for name in STAGES}
    stage_seconds = {name: round(value, 3) for name, value in raw_seconds.items()}
    total_seconds = round(sum(raw_seconds.values()), 3)
    files = int(inventory.get("files", 0) or 0)
    processed = int(features.get("processed", 0) or 0)
    skipped_oversize = int(features.get("skipped_oversize", 0) or 0)
    skipped_pixel = int(features.get("skipped_pixel_limit", 0) or 0)
    skipped_aspect = int(features.get("skipped_aspect_ratio", 0) or 0)
    stage0_rate = _rate(files, raw_seconds["stage0"])
    stage1_rate = _rate(processed, raw_seconds["stage1"])
    scope = int(library_still_images if library_still_images is not None
                else (inventory.get("still_images") or 0) or 0)
    eta_seconds: Optional[float] = None
    eta_basis = "unknown (no images processed in this run)"
    # Only Stage 1 has a representative per-image sample under --limit.
    # Scaling the whole pipeline would multiply Stage 0's already-completed
    # directory walk and produce a deceptively large number.
    sampled_seconds = raw_seconds["stage1"]
    if processed > 0 and scope > 0 and sampled_seconds > 0:
        eta_seconds = sampled_seconds * (scope / processed)
        eta_basis = (
            f"rough Stage 1 scale-up: {sampled_seconds:.1f}s for "
            f"{processed} processed image(s) -> {scope} discovered still image(s)"
        )
    return {
        "stage_seconds": stage_seconds,
        "total_seconds": total_seconds,
        "inventory_files": files,
        "inventory_still_images": scope,
        "stage1_processed": processed,
        "stage1_skipped_oversize": skipped_oversize,
        "stage1_skipped_pixel_limit": skipped_pixel,
        "stage1_skipped_aspect_ratio": skipped_aspect,
        "stage0_files_per_second": stage0_rate,
        "stage1_images_per_second": stage1_rate,
        "stage1_phase_telemetry": features.get("phase_telemetry") or {},
        "scope": features.get("scope") or inventory.get("scope") or {},
        "eta_seconds": eta_seconds,
        "eta_hours": (eta_seconds / 3600.0) if eta_seconds is not None else None,
        "eta_basis": eta_basis,
        "eta_disclaimer": (
            "ROUGH LINEAR ESTIMATE ONLY - extrapolated from this run's own sample. "
            "Real time depends on HDD throughput, photo sizes, GPU model speed, and "
            "how much work was already cached."
        ),
    }


def render_lines(performance: Dict[str, Any], disk: Optional[Dict[str, Any]] = None) -> list[str]:
    """Human-readable report lines shared by performance.txt and the console."""
    stage_seconds = performance["stage_seconds"]
    lines = [
        "Photo Dedup - observed performance (this run)",
        *_scope_lines(performance.get("scope")),
        *(f"{name} wall time: {stage_seconds[name]:.2f}s" for name in STAGES),
        f"total wall time: {performance['total_seconds']:.2f}s",
        f"stage0 processed: {performance['inventory_files']} files "
        f"({_fmt_rate(performance['stage0_files_per_second'], 'files/s')})",
        f"stage1 processed: {performance['stage1_processed']} images "
        f"({_fmt_rate(performance['stage1_images_per_second'], 'images/s')})",
        f"stage1 skipped: {performance.get('stage1_skipped_oversize', 0)} images "
        f"(PIXEL_LIMIT={performance.get('stage1_skipped_pixel_limit', 0)}, "
        f"ASPECT_RATIO={performance.get('stage1_skipped_aspect_ratio', 0)})",
        f"discovered still images used for projection: "
        f"{performance['inventory_still_images']}",
    ]
    if performance["eta_hours"] is not None:
        lines.append(
            f"ROUGH full-library Stage 1 ETA: {performance['eta_hours']:.2f}h "
            f"({performance['eta_seconds'] / 60.0:.1f} min) - {performance['eta_basis']}"
        )
    else:
        lines.append(f"ROUGH full-library Stage 1 ETA: unknown - {performance['eta_basis']}")
    lines.append(performance["eta_disclaimer"])
    telemetry = performance.get("stage1_phase_telemetry") or {}
    if telemetry:
        lines.append("")
        lines.extend(stage1_telemetry.render_lines(telemetry))
    if disk:
        lines.extend([
            "",
            "Thumbnail cache (SSD)",
            f"directory: {disk['directory']}",
            f"cached thumbnails: {disk['cached_files']} of {disk['still_images']} still images",
            f"actual usage: {_fmt_gib(disk['actual_bytes'] / GIB)}",
            "estimated full-library usage: "
            + (_fmt_gib(disk["estimated_total_bytes"] / GIB)
               if disk["estimated_total_bytes"] is not None else "unknown")
            + f" (basis: {disk['basis']})",
            "SSD free space: "
            + (_fmt_gib(disk["free_bytes"] / GIB) if disk["free_bytes"] is not None
               else "unknown"),
        ])
        if disk["low_space"]:
            lines.append(
                f"WARNING: less than {disk['warn_free_gib']:.0f} GiB free on the thumbnail "
                "volume. This is advisory only - the pipeline does not stop, and it never "
                "requires 50 GiB free."
            )
    return lines


def write_performance_file(output_dir: Path, performance: Dict[str, Any],
                           disk: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(output_dir) / "performance.txt"
    path.write_text("\n".join(render_lines(performance, disk)) + "\n", encoding="utf-8")
    return path


def print_report(performance: Dict[str, Any], disk: Optional[Dict[str, Any]] = None,
                 prefix: str = "[pipeline]") -> None:
    for line in render_lines(performance, disk):
        print(f"{prefix} {line}" if line else prefix)


def print_thumbnail_plan(output_dir: Path, still_images: int,
                         average_bytes: Optional[float] = None,
                         warn_free_gib: float = LOW_SPACE_WARN_GIB) -> Dict[str, Any]:
    """Startup notice: expected thumbnail footprint and current SSD free space."""
    from . import thumbnails

    directory = thumbnails.thumbs_dir(output_dir)
    existing_files, existing_bytes = thumbnails.directory_usage(directory)
    basis_average = average_bytes
    if basis_average is None and existing_files:
        basis_average = existing_bytes / existing_files
    estimate = thumbnails.estimate_total_bytes(still_images, basis_average)
    free = free_bytes(directory)
    print(f"[pipeline] thumbnail cache directory: {directory}")
    print(
        "[pipeline] estimated thumbnail usage for "
        f"{still_images} still image(s): "
        + (_fmt_gib(estimate / GIB) if estimate is not None
           else "unknown until the first thumbnails are written (rough guide: "
                "~25-45 KiB each at 320px)")
    )
    print(
        f"[pipeline] thumbnails already cached: {existing_files} "
        f"({_fmt_gib(existing_bytes / GIB)}); SSD free: "
        + (_fmt_gib(free / GIB) if free is not None else "unknown")
    )
    low = free is not None and free < float(warn_free_gib) * GIB
    if low:
        print(
            f"[pipeline][WARN] less than {warn_free_gib:.0f} GiB free on the thumbnail "
            "volume; continuing anyway (advisory only)"
        )
    return {
        "directory": str(directory), "existing_files": existing_files,
        "existing_bytes": existing_bytes, "estimated_total_bytes": estimate,
        "free_bytes": free, "low_space": low,
    }
