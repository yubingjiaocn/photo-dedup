"""Offline-only SigLIP shadow inference provider.

The provider consumes the RGB image already present in ``RoutingInput`` and
returns only the existing raw-score audit record.  It never opens an image,
selects tags, calibrates confidence, or retries with altered inference
semantics.  Local model identity is pinned by path, revision, and SHA-256.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .scene_router import RoutingInput
from .siglip_router import build_shadow_routing_record, load_prompt_bank, prompt_bank_hash


class SiglipBackend(Protocol):
    """Injectable score backend used by tests and the local adapter."""

    def score(
        self, image: Any, prompts: Sequence[str], *, processor: Any, runtime: "RuntimeConfig"
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path
    model_revision: str
    model_sha256: str
    prompt_path: Path
    prompt_sha256: str
    device: str
    precision: str
    batch_size: int


class SiglipShadowProvider:
    """Callable provider producing an always-UNKNOWN shadow routing record."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        backend: SiglipBackend | None = None,
        processor: Any = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.runtime = _runtime_config(settings, Path(base_dir or Path.cwd()))
        _verify_local_artifacts(self.runtime)
        self.bank = load_prompt_bank(self.runtime.prompt_path)
        self.backend = backend or TransformersSiglipBackend()
        self.processor = processor
        self.prompts, self.locations = _ordered_prompts(self.bank)

    def __call__(self, routing_input: RoutingInput) -> Mapping[str, Any]:
        if routing_input.image is None:
            raise ValueError("RoutingInput.image is required")
        values = self.backend.score(
            routing_input.image,
            self.prompts,
            processor=self.processor,
            runtime=self.runtime,
        )
        scores = _strict_scores(values, len(self.prompts))
        raw: dict[str, dict[str, float]] = {code: {} for code in self.bank["tags"]}
        for (code, prompt), score in zip(self.locations, scores, strict=True):
            raw[code][prompt] = score
        return build_shadow_routing_record(
            self.bank,
            raw,
            model_name=self.runtime.model_path.name,
            model_revision=self.runtime.model_revision,
            model_audit={
                "model_sha256": self.runtime.model_sha256,
                "runtime": {
                    "device": self.runtime.device,
                    "precision": self.runtime.precision,
                    "batch_size": self.runtime.batch_size,
                    "oom_fallback": False,
                },
            },
        )


class TransformersSiglipBackend:
    """Thin, lazy adapter around a strictly local Transformers checkpoint."""

    def __init__(self) -> None:
        self._loaded: tuple[Any, Any, Any] | None = None

    def score(
        self, image: Any, prompts: Sequence[str], *, processor: Any, runtime: RuntimeConfig
    ) -> Sequence[float]:
        torch, loaded_processor, model = self._load(runtime, processor)
        output: list[float] = []
        # Prompt chunking is configured up front and only partitions an
        # otherwise identical score vector. OOM is propagated; no device,
        # precision, image size, or batch-size fallback is permitted.
        for start in range(0, len(prompts), runtime.batch_size):
            chunk = list(prompts[start : start + runtime.batch_size])
            inputs = loaded_processor(
                text=chunk,
                images=[image],
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
            with torch.inference_mode():
                result = model(**inputs)
            logits = result.logits_per_image
            if tuple(logits.shape) != (1, len(chunk)):
                raise ValueError(
                    f"SigLIP logits shape must be (1, {len(chunk)}), got {tuple(logits.shape)!r}"
                )
            output.extend(logits[0].float().cpu().tolist())
        return output

    def _load(self, runtime: RuntimeConfig, processor: Any) -> tuple[Any, Any, Any]:
        if self._loaded is not None:
            return self._loaded
        try:
            import torch
            from transformers import AutoProcessor, SiglipModel
        except ImportError as exc:
            raise RuntimeError("local SigLIP requires torch and transformers") from exc
        if runtime.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SigLIP configured for CUDA but CUDA is unavailable")
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[runtime.precision]
        # local_files_only blocks Hub resolution; the verified filesystem path
        # is the sole model source. Revision is retained as pinned audit data,
        # while the content hash is the executable identity check.
        loaded_processor = processor or AutoProcessor.from_pretrained(
            str(runtime.model_path), local_files_only=True, trust_remote_code=False
        )
        model = SiglipModel.from_pretrained(
            str(runtime.model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        ).to(runtime.device).eval()
        self._loaded = (torch, loaded_processor, model)
        return self._loaded


def build_siglip_provider(
    config: Any, *, backend: SiglipBackend | None = None, processor: Any = None
) -> SiglipShadowProvider:
    """Build from ``Config`` without enabling any implicit model discovery."""
    settings = config.features.get("scene_routing", {})
    base_dir = config.source.parent if getattr(config, "source", None) else Path.cwd()
    return SiglipShadowProvider(
        settings, backend=backend, processor=processor, base_dir=base_dir
    )


def artifact_sha256(path: str | Path) -> str:
    """Hash one file or a directory tree, rejecting symlinks and special files."""
    root = Path(path)
    if not root.exists() or root.is_symlink():
        raise ValueError("local model path must exist and must not be a symlink")
    if root.is_file():
        return hashlib.sha256(root.read_bytes()).hexdigest()
    if not root.is_dir():
        raise ValueError("local model path must be a file or directory")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("local model directory must contain files")
    for item in files:
        if item.is_symlink():
            raise ValueError("local model directory must not contain symlinks")
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _runtime_config(settings: Mapping[str, Any], base_dir: Path) -> RuntimeConfig:
    if not isinstance(settings, Mapping):
        raise ValueError("scene_routing settings must be an object")
    model = _object(settings.get("model"), "scene_routing.model")
    prompt = _object(settings.get("prompt"), "scene_routing.prompt")
    runtime = _object(settings.get("runtime"), "scene_routing.runtime")
    model_path = _local_path(model.get("path"), base_dir, "model.path")
    prompt_path = _local_path(prompt.get("path"), base_dir, "prompt.path")
    revision = _text(model.get("revision"), "model.revision")
    model_hash = _sha(model.get("sha256"), "model.sha256")
    prompt_hash = _sha(prompt.get("sha256"), "prompt.sha256")
    device = runtime.get("device")
    precision = runtime.get("precision")
    batch_size = runtime.get("batch_size")
    if device not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be 'cpu' or 'cuda'")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("runtime.precision must be float32, float16, or bfloat16")
    if device == "cpu" and precision != "float32":
        raise ValueError("CPU SigLIP requires float32 precision")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("runtime.batch_size must be a positive integer")
    return RuntimeConfig(
        model_path, revision, model_hash, prompt_path, prompt_hash,
        device, precision, batch_size,
    )


def _verify_local_artifacts(runtime: RuntimeConfig) -> None:
    if artifact_sha256(runtime.model_path) != runtime.model_sha256:
        raise ValueError("local model SHA-256 mismatch")
    if prompt_bank_hash(runtime.prompt_path) != runtime.prompt_sha256:
        raise ValueError("prompt bank SHA-256 mismatch")


def _ordered_prompts(bank: Mapping[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    prompts: list[str] = []
    locations: list[tuple[str, str]] = []
    for code, spec in bank["tags"].items():
        per_tag = (
            list(spec["positive_prompts"])
            + list(spec["hard_negative_prompts"])
            + [bank["generic_null_prompt"]]
        )
        prompts.extend(per_tag)
        locations.extend((code, prompt) for prompt in per_tag)
    return prompts, locations


def _strict_scores(value: Any, expected: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("SigLIP backend scores must be a flat sequence")
    if len(value) != expected:
        raise ValueError(f"SigLIP backend returned {len(value)} scores; expected {expected}")
    output: list[float] = []
    for score in value:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("SigLIP backend scores must be finite numbers")
        output.append(float(score))
    return output


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return text


def _local_path(value: Any, base_dir: Path, label: str) -> Path:
    text = _text(value, label)
    if "://" in text:
        raise ValueError(f"{label} must be a local filesystem path")
    path = Path(text)
    return path if path.is_absolute() else (base_dir / path)
