"""Stage 1 loop settings: validated once, up front, with no silent correction.

Every knob here changes how much work is in flight, so a bad value must stop the
run with a message that names the key and the value — never get clamped into
something the user did not ask for. That rule is why this is a module and not a
handful of ``int(cfg.features.get(...))`` calls scattered through the loop.

The knobs
---------
``cpu_workers``
    Threads preparing images. ``0`` means the pre-change pipeline exactly: one
    thread, no queue, preparation billed to the loop. ``"auto"`` (the default)
    means ``min(4, os.cpu_count())``. Four is deliberate rather than "all cores":
    the measured scaling is 3.18x at four threads and only 4.28x at eight
    (``docs/STAGE1_THROUGHPUT.md``), while the main thread still needs a core for
    the GPU lanes, YuNet and SQLite. On the 8C/16T target box that leaves plenty
    of headroom and keeps the memory bound small.
``prefetch_batches``
    Batches of prepared images allowed to queue up. Default 2. Combined with the
    reader, at most ``prefetch_batches + 1`` batches of decoded frames exist at
    once, so the ceiling is
    ``(prefetch_batches + 1) x features.max_inflight_megapixels`` of decoded RGB
    plus one bounded IQA array per in-flight image. The IQA arrays are bounded by
    ``iqa_max_long_edge``, **not** by the source resolution, so they scale with the
    image *count* rather than with the megapixel guard. ``0`` disables prefetch.
``iqa_batch_size``
    Images per IQA model call, capped by the batch itself. Default 4, measured at
    2.99 GiB peak VRAM for MUSIQ at a 1920 px long edge (8 would reach 5.87 GiB
    for no kernel-time gain).

``memory_note`` states the resulting bound in MP and MiB so it lands in the run
log, where a Windows user can see what the settings actually cost before the
first batch is read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .stage1_backends import DEFAULT_IQA_BATCH_SIZE

AUTO = "auto"
DEFAULT_PREFETCH_BATCHES = 2
DEFAULT_MAX_CPU_WORKERS = 4
# Bytes per megapixel of decoded RGB (3 bytes/px) and of a float32 IQA array
# (3 channels x 4 bytes = 12 bytes/px).
_RGB_BYTES_PER_MP = 3 * 1_000_000
_IQA_BYTES_PER_MP = 12 * 1_000_000


def default_cpu_workers() -> int:
    """``min(4, cpu_count)``: conservative, leaves the main thread a core."""
    return max(1, min(DEFAULT_MAX_CPU_WORKERS, os.cpu_count() or 1))


def _require_int(features: Any, key: str, default: int, minimum: int) -> int:
    """Strict int read: wrong type or out of range raises, never clamps."""
    raw = features.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"features.{key} must be an integer, got {raw!r}")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"features.{key} must be an integer, got {raw!r}") from exc
    if isinstance(raw, float) and float(raw) != value:
        raise ValueError(f"features.{key} must be a whole number, got {raw!r}")
    if value < minimum:
        raise ValueError(f"features.{key} must be at least {minimum}, got {value}")
    return value


def _resolve_workers(features: Any, override: Optional[int]) -> int:
    """``cpu_workers`` as an int, honouring ``auto`` and rejecting nonsense."""
    if override is not None:
        value = int(override)
        if value < 0:
            raise ValueError(f"cpu_workers must be at least 0, got {value}")
        return value
    raw = features.get("cpu_workers", AUTO)
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == AUTO):
        return default_cpu_workers()
    return _require_int(features, "cpu_workers", default_cpu_workers(), 0)


@dataclass(frozen=True)
class LoopSettings:
    """Validated Stage 1 loop configuration."""

    batch_size: int
    max_inflight_pixels: int
    cpu_workers: int
    prefetch_batches: int
    iqa_batch_size: int
    commit_every: int
    iqa_max_long_edge: int = 1920

    @property
    def prefetch_enabled(self) -> bool:
        return self.cpu_workers > 0 and self.prefetch_batches > 0

    @property
    def inflight_batches(self) -> int:
        """Batches of decoded frames that can exist at once (reader + queue)."""
        return self.prefetch_batches + 1 if self.prefetch_enabled else 1

    def memory_note(self) -> str:
        """Human-readable statement of the decoded-image memory ceiling.

        Two independent bounds, because they scale differently: decoded frames are
        capped by ``max_inflight_megapixels`` per batch, while the float32 IQA
        arrays are capped by ``iqa_max_long_edge`` per *image*, however large the
        source was. Quoting one number for both would overstate the total on a
        library of 50 MP photos.
        """
        megapixels = self.inflight_batches * (self.max_inflight_pixels / 1_000_000)
        rgb_mib = megapixels * _RGB_BYTES_PER_MP / (1024 ** 2)
        images = self.inflight_batches * self.batch_size
        iqa_mp = (self.iqa_max_long_edge ** 2) / 1_000_000
        iqa_mib = images * iqa_mp * _IQA_BYTES_PER_MP / (1024 ** 2)
        return (
            f"memory bound: <= {self.inflight_batches} batch(es) in flight x "
            f"{self.max_inflight_pixels / 1_000_000:.0f} MP = {megapixels:.0f} MP decoded "
            f"(~{rgb_mib:.0f} MiB RGB), plus <= {images} bounded IQA arrays of "
            f"<= {self.iqa_max_long_edge}px (~{iqa_mib:.0f} MiB float32)"
        )

    def note(self) -> str:
        if not self.prefetch_enabled:
            return ("cpu_workers=0 or prefetch_batches=0: serial pipeline "
                    "(read, decode and prepare run in the batch loop)")
        return (f"prefetch: {self.cpu_workers} CPU worker(s), "
                f"{self.prefetch_batches} batch(es) queued, 1 sequential reader")


def resolve_loop_settings(cfg: Any, workers_override: Optional[int] = None) -> LoopSettings:
    """Read and validate every loop knob. Raises ``ValueError`` on bad input."""
    features = cfg.features
    batch_size = _require_int(features, "batch_size", 4, 1)
    max_inflight_mp = features.get("max_inflight_megapixels", 80)
    try:
        max_inflight = float(max_inflight_mp)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"features.max_inflight_megapixels must be a number, got {max_inflight_mp!r}"
        ) from exc
    if max_inflight <= 0:
        raise ValueError("features.max_inflight_megapixels must be greater than zero")
    workers = _resolve_workers(features, workers_override)
    prefetch = _require_int(features, "prefetch_batches", DEFAULT_PREFETCH_BATCHES, 0)
    iqa_batch = _require_int(features, "iqa_batch_size", DEFAULT_IQA_BATCH_SIZE, 1)
    commit_every = _require_int(cfg.scan, "commit_every", 100, 1)
    iqa_long_edge = _require_int(features, "iqa_max_long_edge", 1920, 1)
    return LoopSettings(
        batch_size=batch_size,
        max_inflight_pixels=int(max_inflight * 1_000_000),
        cpu_workers=workers,
        prefetch_batches=prefetch,
        iqa_batch_size=iqa_batch,
        commit_every=commit_every,
        iqa_max_long_edge=iqa_long_edge,
    )
