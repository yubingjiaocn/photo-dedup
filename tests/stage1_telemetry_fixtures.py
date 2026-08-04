"""Shared helpers for the Stage 1 telemetry tests.

The fake CUDA module is the important part: it counts ``Event.record`` calls and
``synchronize`` calls, and can make a synchronisation *cost* measurable time. That
is what lets a test prove the two claims the design rests on -- one sync per
sampled batch, taken after the batch's wall clock -- without a GPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from PIL import Image

from src import telemetry_report


# --- fake CUDA -------------------------------------------------------------

class FakeEvent:
    """Records ordering only; ``synchronize`` is counted, never real.

    ``sync_seconds`` makes the (normally invisible) cost of a synchronisation
    measurable, so a test can prove *when* it happened relative to the batch
    clock rather than just how often.
    """

    def __init__(self, sink: "FakeCuda") -> None:
        self._sink = sink
        self._at = 0.0

    def record(self) -> None:
        self._sink.records += 1
        self._at = time.perf_counter()

    def synchronize(self) -> None:
        self._sink.syncs += 1
        if self._sink.sync_seconds:
            time.sleep(self._sink.sync_seconds)

    def elapsed_time(self, other: "FakeEvent") -> float:
        return max(0.0, (other._at - self._at)) * 1000.0 or 1.5


class FakeCuda:
    """Minimal torch.cuda stand-in that counts records and synchronisations."""

    def __init__(self, available: bool = True, sync_seconds: float = 0.0) -> None:
        self.available = available
        self.sync_seconds = sync_seconds
        self.records = 0
        self.syncs = 0
        cuda = self

        class _Namespace:
            @staticmethod
            def is_available() -> bool:
                return cuda.available

            @staticmethod
            def Event(enable_timing: bool = False) -> FakeEvent:  # noqa: N802
                return FakeEvent(cuda)

        self.cuda = _Namespace


def library(root: Path, count: int = 3) -> list[Path]:
    """``count`` decodable JPEGs, large enough that decode/IQA take real time."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"IMG_2026010{index}_120000.jpg"
        Image.new("RGB", (256, 192), (10 + index * 30, 60, 120)).save(path, "JPEG")
        paths.append(path)
    return paths


def config(tmp_path: Path, root: Path, **telemetry) -> str:
    """Stage config with telemetry on and CUDA sampling off; ``**telemetry`` overrides."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"root": str(root), "db": str(tmp_path / "inventory.sqlite"),
                  "output_dir": str(tmp_path / "out"),
                  "models_dir": str(tmp_path / "models"), "trash": str(tmp_path / "trash")},
        "features": {"backend": "stub", "batch_size": 2,
                     "thumbnails": {"enabled": True, "max_px": 64},
                     "telemetry": {"enabled": True, "gpu_event_every": 0, **telemetry}},
    }), encoding="utf-8")
    return str(path)


def rendered_line(snapshot: dict, prefix: str) -> str:
    """First rendered line containing ``prefix`` (assertion helper)."""
    for line in telemetry_report.render_lines(snapshot):
        if prefix in line:
            return line
    raise AssertionError(f"no rendered line contains {prefix!r}")
