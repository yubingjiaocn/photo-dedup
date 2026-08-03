"""Offline adapter for ``src.windows_benchmark --runner src.windows_siglip_runner:factory``.

It accepts only explicit local SigLIP identity/configuration and passes the
benchmark's already-decoded PIL image directly to ``SiglipShadowProvider``.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .routing_schema import validate_routing_record
from .siglip_runtime import SiglipShadowProvider

ENV_PREFIX = "PHOTO_DEDUP_SIGLIP_"
RUNNER_SCHEMA_VERSION = 1
_REQUIRED = {
    "model_path", "model_revision", "model_sha256", "prompt_bank_path",
    "prompt_bank_sha256", "device", "precision",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Windows SigLIP runner requires explicit {name}")
    return value.strip()


def _settings(values: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(name for name in _REQUIRED if not values.get(name))
    if missing:
        raise ValueError("Windows SigLIP runner missing identity field(s): " + ", ".join(missing))
    batch_size = values.get("batch_size", 16)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Windows SigLIP runner batch_size must be a positive integer")
    return {
        "model": {
            "path": _text(values["model_path"], "model_path"),
            "revision": _text(values["model_revision"], "model_revision"),
            "sha256": _text(values["model_sha256"], "model_sha256"),
        },
        "prompt": {
            "path": _text(values["prompt_bank_path"], "prompt_bank_path"),
            "sha256": _text(values["prompt_bank_sha256"], "prompt_bank_sha256"),
        },
        "runtime": {
            "device": _text(values["device"], "device"),
            "precision": _text(values["precision"], "precision"),
            "batch_size": batch_size,
        },
    }


def _config_values() -> dict[str, Any]:
    """Read a separate JSON runner config, then allow explicit env overrides."""
    values: dict[str, Any] = {}
    config_path = os.environ.get(ENV_PREFIX + "RUNNER_CONFIG")
    if config_path:
        try:
            loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read Windows SigLIP runner config") from exc
        if not isinstance(loaded, dict):
            raise ValueError("Windows SigLIP runner config must be a JSON object")
        unknown = set(loaded) - (_REQUIRED | {"batch_size"})
        if unknown:
            raise ValueError(f"Windows SigLIP runner config has unknown keys: {sorted(unknown)!r}")
        values.update(loaded)
    for name in _REQUIRED | {"batch_size"}:
        env_value = os.environ.get(ENV_PREFIX + name.upper())
        if env_value is not None:
            values[name] = int(env_value) if name == "batch_size" and env_value.isdigit() else env_value
    return values


class WindowsSiglipRunner:
    """Real local SigLIP plus explicitly labelled deterministic Stage 1 stub."""

    def __init__(self, settings: Mapping[str, Any], *, provider: SiglipShadowProvider | None = None) -> None:
        self._settings = _settings(settings)
        self.provider = provider or SiglipShadowProvider(self._settings)
        runtime = self.provider.runtime
        # Paths are deliberately excluded: checkpoints only bind public hashes/tokens.
        self.model_descriptor = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "siglip": {
                "family": "siglip", "model_revision": runtime.model_revision,
                "model_sha256": runtime.model_sha256,
                "prompt_bank_sha256": runtime.prompt_sha256,
                "prompt_bank_version": self.provider.bank["bank_version"],
                "device": runtime.device, "precision": runtime.precision,
                "batch_size": runtime.batch_size, "provider_schema_version": 1,
            },
            "stage1": {"backend": "StubBackend", "schema_version": 1, "deterministic": True},
        }

    def siglip(self, image: Image.Image) -> dict[str, Any]:
        """Process exactly this decoded image; invalid evidence is a hard error."""
        if not isinstance(image, Image.Image):
            raise ValueError("Windows SigLIP runner requires a decoded PIL.Image")
        record = validate_routing_record(self.provider(_routing_input(image)))
        for namespace in ("scene_context", "subject_protection"):
            if record[namespace]["state"] != "UNKNOWN":
                raise ValueError("SigLIP shadow provider must remain UNKNOWN")
            if "OUT_OF_CALIBRATION_DOMAIN" not in record[namespace]["reasons"]:
                raise ValueError("SigLIP shadow provider lacks calibration abstention reason")
        if not _complete_finite_raw_scores(record, set(self.provider.bank["tags"])):
            raise ValueError("SigLIP shadow provider returned incomplete or non-finite raw scores")
        return {"status": "PROCESSED", "routing_record": record}

    def stage1(self, image: Image.Image) -> dict[str, Any]:
        """Use the deterministic backend only: real Stage 1 may download models."""
        if not isinstance(image, Image.Image):
            raise ValueError("Windows Stage 1 runner requires a decoded PIL.Image")
        from .stage1_features import StubBackend

        backend = StubBackend()
        embedding = backend.embed_batch([image])[0]
        quality, _ = backend.quality(image)
        return {
            "status": "PROCESSED", "backend": "StubBackend",
            "embedding_dimensions": int(len(embedding)), "quality": float(quality),
        }


def _routing_input(image: Image.Image) -> Any:
    from .scene_router import RoutingInput
    return RoutingInput(metadata={}, image=image)


def _complete_finite_raw_scores(record: Mapping[str, Any], expected_tags: set[str]) -> bool:
    """Require the provider's full prompt audit before reporting throughput."""
    model = record.get("model")
    if not isinstance(model, Mapping):
        return False
    audit = model.get("shadow_prompt_audit")
    if not isinstance(audit, Mapping) or not isinstance(audit.get("tags"), Mapping):
        return False
    tags = audit["tags"]
    if set(tags) != expected_tags:
        return False
    for tag in tags.values():
        if not isinstance(tag, Mapping) or not isinstance(tag.get("raw_prompt_scores"), Mapping):
            return False
        values = tag["raw_prompt_scores"].values()
        if not values or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            return False
    return True


def factory() -> WindowsSiglipRunner:
    """Factory loaded by the benchmark; all identity is explicit and local."""
    return WindowsSiglipRunner(_config_values())
