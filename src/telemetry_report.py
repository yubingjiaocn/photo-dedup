"""Rendering for the Stage 1 phase breakdown (log + performance.txt).

Split from :mod:`src.stage1_telemetry` so the collector stays about measuring and
this file stays about wording. The wording is the honesty contract:

* the ``unit`` column says whether a phase's calls were per image, per batch or a
  one-off step, so a per-image measurement is never read as per batch;
* ``host-wall`` is stated on every host measurement, with the async-CUDA caveat;
* CUDA-event numbers are printed separately, as ``samples`` with ms/sample, and
  the number of synchronisations is disclosed;
* both reconciliations are printed: in-loop phases against the measured batch
  total, and loop + setup against Stage 1's outer wall time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import stage1_telemetry as tel

_HEADER = f"{'phase':<32}{'unit':>6}{'calls':>8}{'seconds':>10}{'ms/img':>9}{'%':>7}  timing"


def render_lines(snapshot: Dict[str, Any]) -> List[str]:
    """Fixed-width, greppable summary shared by the log and performance.txt."""
    if not snapshot or not snapshot.get("enabled"):
        return ["Stage 1 phase breakdown: disabled (features.telemetry.enabled=false)"]
    lines = [
        "Stage 1 phase breakdown (cumulative; host-wall unless stated otherwise)",
        _counts_line(snapshot),
        _reconciliation_line(snapshot),
    ]
    loop = _phases(snapshot, tel.LOOP)
    if loop:
        lines.append("in-loop phases (percent of batch total)")
        lines.append(_HEADER)
        lines.extend(_phase_line(phase) for phase in loop)
        lines.append(_gap_line("unaccounted in-loop (loop overhead)",
                               snapshot.get("loop_unaccounted_seconds"),
                               snapshot.get("loop_unaccounted_percent"),
                               snapshot.get("images_attempted")))
    setup = _phases(snapshot, tel.SETUP)
    if setup:
        lines.append("setup / finalize phases (percent of Stage 1 outer wall time)")
        lines.append(_HEADER)
        lines.extend(_phase_line(phase) for phase in setup)
    outer_gap = snapshot.get("outer_unaccounted_seconds")
    if outer_gap is not None:
        lines.append(_gap_line("unaccounted outside loop/setup", outer_gap,
                               _percent(outer_gap, snapshot.get("outer_seconds")),
                               snapshot.get("images_attempted")))
    percentiles = snapshot.get("batch_percentiles") or {}
    lines.append(
        f"batch wall time p50={_seconds(percentiles.get('p50'))} "
        f"p95={_seconds(percentiles.get('p95'))}"
    )
    lines.extend(_gpu_lines(snapshot))
    lines.extend(f"note: {note}" for note in (snapshot.get("notes") or []))
    lines.append(
        "host-wall = measured on the CPU thread. With an async CUDA backend that "
        "includes submit + wait, so it is an upper bound on kernel time, not kernel time."
    )
    return lines


def _counts_line(snapshot: Dict[str, Any]) -> str:
    attempted = int(snapshot.get("images_attempted") or 0)
    loop_seconds = float(snapshot.get("loop_seconds") or 0.0)
    per_image = (f" ({1000.0 * loop_seconds / attempted:.1f} ms/image attempted)"
                 if attempted else "")
    return (
        f"images attempted={attempted} succeeded={snapshot.get('images_succeeded', 0)} "
        f"failed={snapshot.get('images_failed', 0)} batches={snapshot.get('batches', 0)} "
        f"batch_total={loop_seconds:.2f}s{per_image}"
    )


def _reconciliation_line(snapshot: Dict[str, Any]) -> str:
    outer = snapshot.get("outer_seconds")
    setup = float(snapshot.get("setup_accounted_seconds") or 0.0)
    loop = float(snapshot.get("loop_seconds") or 0.0)
    if outer is None:
        return (f"reconciliation: batch total {loop:.2f}s + timed setup {setup:.2f}s "
                "(Stage 1 outer wall time not supplied)")
    gap = snapshot.get("outer_unaccounted_seconds")
    return (
        f"reconciliation: Stage 1 outer {float(outer):.2f}s = batch total {loop:.2f}s "
        f"+ timed setup/finalize {setup:.2f}s + unaccounted {float(gap or 0.0):.2f}s"
    )


def _phases(snapshot: Dict[str, Any], scope: str) -> List[Dict[str, Any]]:
    rows = [phase for phase in (snapshot.get("phases") or [])
            if phase.get("scope") == scope]
    rows.sort(key=lambda item: float(item.get("seconds") or 0.0), reverse=True)
    return rows


def _phase_line(phase: Dict[str, Any]) -> str:
    ms = phase.get("ms_per_image")
    percent = phase.get("percent")
    timing = "host-wall"
    if phase.get("gpu_event_seconds") is not None:
        timing = "host-wall (+CUDA below)"
    return (
        f"{str(phase.get('label'))[:31]:<32}"
        f"{str(phase.get('unit')):>6}"
        f"{int(phase.get('calls') or 0):>8}"
        f"{float(phase.get('seconds') or 0.0):>10.2f}"
        f"{(f'{ms:.2f}' if ms is not None else '-'):>9}"
        f"{(f'{percent:.1f}' if percent is not None else '-'):>7}"
        f"  {timing}"
    )


def _gap_line(label: str, seconds: Optional[float], percent: Optional[float],
              images: Optional[int]) -> str:
    value = float(seconds or 0.0)
    count = int(images or 0)
    ms = (1000.0 * value / count) if count else None
    return (
        f"{label[:31]:<32}{'-':>6}{'-':>8}{value:>10.2f}"
        f"{(f'{ms:.2f}' if ms is not None else '-'):>9}"
        f"{(f'{percent:.1f}' if percent is not None else '-'):>7}  host-wall"
    )


def _gpu_lines(snapshot: Dict[str, Any]) -> List[str]:
    every = int(snapshot.get("gpu_event_sampling_every") or 0)
    sampled = [phase for phase in (snapshot.get("phases") or [])
               if phase.get("gpu_event_seconds") is not None]
    if not sampled:
        reason = ("disabled (features.telemetry.gpu_event_every=0)" if every == 0
                  else "enabled but no batch sampled yet, or CUDA unavailable")
        return [f"CUDA event timing: {reason}"]
    lines = [
        f"CUDA event timing: every {every}th batch sampled "
        f"({snapshot.get('gpu_event_sampled_batches', 0)} batch(es), "
        f"{snapshot.get('gpu_event_synchronisations', 0)} synchronisation(s) total, "
        "taken after each batch's wall clock)"
    ]
    for phase in sampled:
        samples = int(phase.get("gpu_event_samples") or 0)
        seconds = float(phase.get("gpu_event_seconds") or 0.0)
        per_sample = (seconds / samples) if samples else None
        unit = phase.get("unit")
        lines.append(
            f"  {str(phase.get('label'))[:31]:<32} samples={samples:<4} "
            f"gpu={seconds:.4f}s"
            + (f" ({per_sample * 1000.0:.2f} ms per sampled {unit})"
               if per_sample is not None else "")
        )
    lines.append(
        "  a sample is one instrumented call of that lane (see the unit column); "
        "at most one sample per lane per sampled batch"
    )
    return lines


def _percent(value: Optional[float], total: Optional[float]) -> Optional[float]:
    if value is None or not total:
        return None
    return 100.0 * float(value) / float(total)


def _seconds(value: Optional[float]) -> str:
    return f"{value:.3f}s" if value is not None else "unknown"


def print_summary(snapshot: Dict[str, Any], prefix: str = "[stage1]") -> None:
    for line in render_lines(snapshot):
        print(f"{prefix} {line}")
