"""Stage 1 loop settings: validated once, up front.

Defaults for Ryzen 7 9700X 8C/16T, 32 GB RAM, RTX 5070 Ti 16 GB, single HDD:
  cpu_workers=4, prefetch_batches=2, iqa_batch_size=4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_CPU_WORKERS = 4
DEFAULT_PREFETCH_BATCHES = 2
DEFAULT_IQA_BATCH_SIZE = 4
_RGB_BYTES_PER_MP = 3 * 1_000_000
_IQA_BYTES_PER_MP = 12 * 1_000_000


def _int_or_default(section: Any, key: str, default: int, minimum: int = 0) -> int:
    """Read int with bounds check."""
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"features.{key} must be an integer, got {raw!r}")
    try:
        value = int(str(raw).strip()) if isinstance(raw, str) else int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"features.{key} must be an integer, got {raw!r}") from exc
    if isinstance(raw, float) and float(raw) != value:
        raise ValueError(f"features.{key} must be an integer, got {raw!r}")
    if value < minimum:
        raise ValueError(f"features.{key} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class LoopSettings:
    batch_size: int
    max_inflight_pixels: int
    cpu_workers: int
    prefetch_batches: int
    iqa_batch_size: int
    commit_every: int
    iqa_max_long_edge: int

    @property
    def prefetch_enabled(self) -> bool:
        return self.cpu_workers > 0 and self.prefetch_batches > 0

    @property
    def inflight_batches(self) -> int:
        return self.prefetch_batches + 1 if self.prefetch_enabled else 1

    def memory_note(self) -> str:
        megapixels = self.inflight_batches * (self.max_inflight_pixels / 1_000_000)
        rgb_mib = megapixels * _RGB_BYTES_PER_MP / (1024 ** 2)
        images = self.inflight_batches * self.batch_size
        iqa_mib = (images * (self.iqa_max_long_edge ** 2) / 1_000_000
                   * _IQA_BYTES_PER_MP / (1024 ** 2))
        return (f"memory bound: <= {self.inflight_batches} batch(es) in flight x "
                f"{self.max_inflight_pixels / 1_000_000:.0f} MP = {megapixels:.0f} MP decoded "
                f"(~{rgb_mib:.0f} MiB RGB), plus <= {images} bounded IQA arrays of "
                f"<= {self.iqa_max_long_edge}px (~{iqa_mib:.0f} MiB float32)")

    def note(self) -> str:
        if not self.prefetch_enabled:
            return "serial: read/decode/prepare in batch loop"
        return f"prefetch: {self.cpu_workers} CPU worker(s), {self.prefetch_batches} batch(es) queued"


def resolve_loop_settings(cfg: Any, workers_override: Optional[int] = None) -> LoopSettings:
    """Read and validate loop settings."""
    features = cfg.features
    max_inflight_mp = features.get("max_inflight_megapixels", 80)
    try:
        max_inflight = float(max_inflight_mp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"features.max_inflight_megapixels must be a number, got {max_inflight_mp!r}") from exc
    if max_inflight <= 0:
        raise ValueError("features.max_inflight_megapixels must be greater than zero")
    workers = int(workers_override) if workers_override is not None else _int_or_default(
        features, "cpu_workers", DEFAULT_CPU_WORKERS, 0)
    return LoopSettings(
        batch_size=_int_or_default(features, "batch_size", 4, 1),
        max_inflight_pixels=int(max_inflight * 1_000_000),
        cpu_workers=workers,
        prefetch_batches=_int_or_default(features, "prefetch_batches", DEFAULT_PREFETCH_BATCHES, 0),
        iqa_batch_size=_int_or_default(features, "iqa_batch_size", DEFAULT_IQA_BATCH_SIZE, 1),
        commit_every=_int_or_default(cfg.scan, "commit_every", 100, 1),
        iqa_max_long_edge=_int_or_default(features, "iqa_max_long_edge", 1920, 1),
    )
