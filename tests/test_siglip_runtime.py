"""Safety and fake-backend tests for the offline SigLIP provider."""

from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from src.scene_router import RoutingInput, SceneRouter
from src.siglip_router import load_prompt_bank, prompt_bank_hash
from src.siglip_runtime import (
    SiglipShadowProvider,
    TransformersSiglipBackend,
    artifact_sha256,
)

BANK_PATH = Path(__file__).parents[1] / "research" / "siglip_prompt_bank_v1.yaml"


class FakeBackend:
    def __init__(self, values=None):
        self.values = values
        self.calls = []

    def score(self, image, prompts, *, processor, runtime):
        self.calls.append((image, list(prompts), processor, runtime))
        return self.values if self.values is not None else [index / 100 for index in range(len(prompts))]


def _settings(tmp_path, *, device="cpu", precision="float32", batch_size=8):
    model = tmp_path / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text('{"model_type":"siglip"}', encoding="utf-8")
    prompt = tmp_path / "bank.yaml"
    shutil.copyfile(BANK_PATH, prompt)
    return {
        "model": {
            "path": str(model),
            "revision": "local-revision-abc",
            "sha256": artifact_sha256(model),
        },
        "prompt": {"path": str(prompt), "sha256": prompt_bank_hash(prompt)},
        "runtime": {"device": device, "precision": precision, "batch_size": batch_size},
    }


def _expected_prompts(bank):
    result = []
    for spec in bank["tags"].values():
        result.extend(spec["positive_prompts"])
        result.extend(spec["hard_negative_prompts"])
        result.append(bank["generic_null_prompt"])
    return result


def test_single_image_uses_exact_object_fixed_prompt_order_and_unknown_record(tmp_path):
    settings = _settings(tmp_path)
    backend = FakeBackend()
    processor = object()
    provider = SiglipShadowProvider(settings, backend=backend, processor=processor)
    image = Image.new("RGB", (12, 8), "blue")

    record = provider(RoutingInput(metadata={}, image=image))

    assert len(backend.calls) == 1
    seen_image, prompts, seen_processor, runtime = backend.calls[0]
    assert seen_image is image
    assert seen_processor is processor
    assert prompts == _expected_prompts(load_prompt_bank(BANK_PATH))
    assert runtime.device == "cpu" and runtime.precision == "float32"
    assert record["scene_context"]["state"] == "UNKNOWN"
    assert record["subject_protection"]["state"] == "UNKNOWN"
    assert record["scene_context"]["reasons"] == ["OUT_OF_CALIBRATION_DOMAIN"]
    assert record["model"]["runtime"] == {
        "device": "cpu", "precision": "float32", "batch_size": 8, "oom_fallback": False
    }
    rendered = json.dumps(record).lower()
    assert not any(word in rendered for word in ("probability", "confidence", "decision", "action"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_backend_output_fails_closed_through_scene_router(tmp_path, bad):
    settings = _settings(tmp_path)
    provider = SiglipShadowProvider(settings, backend=FakeBackend([bad] * 67))
    router = SceneRouter({"scene_routing": {"enabled": True, "mode": "shadow"}}, provider)
    record = router.route(RoutingInput(metadata={}, image=object()))
    assert record["scene_context"]["reasons"] == ["ROUTER_CONFLICT"]


@pytest.mark.parametrize("values", [[], [0.0], [[0.0] * 67]])
def test_wrong_backend_shape_or_length_is_rejected(tmp_path, values):
    provider = SiglipShadowProvider(_settings(tmp_path), backend=FakeBackend(values))
    with pytest.raises(ValueError, match="scores|returned"):
        provider(RoutingInput(metadata={}, image=object()))


def test_missing_model_and_hash_mismatch_are_rejected_before_backend(tmp_path):
    settings = _settings(tmp_path)
    settings["model"]["path"] = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="exist"):
        SiglipShadowProvider(settings, backend=FakeBackend())

    settings = _settings(tmp_path / "other")
    settings["model"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        SiglipShadowProvider(settings, backend=FakeBackend())


def test_prompt_hash_mismatch_and_remote_paths_are_rejected(tmp_path):
    settings = _settings(tmp_path)
    settings["prompt"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prompt bank SHA-256 mismatch"):
        SiglipShadowProvider(settings, backend=FakeBackend())

    settings = _settings(tmp_path / "other")
    settings["model"]["path"] = "https://example.invalid/model"
    with pytest.raises(ValueError, match="local filesystem"):
        SiglipShadowProvider(settings, backend=FakeBackend())


def test_missing_transformers_dependency_has_explicit_error(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    provider = SiglipShadowProvider(settings, backend=TransformersSiglipBackend())
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="torch and transformers"):
        provider(RoutingInput(metadata={}, image=object()))


@pytest.mark.parametrize(
    ("device", "precision", "valid"),
    [
        ("cpu", "float32", True),
        ("cuda", "float16", True),
        ("cuda", "bfloat16", True),
        ("cuda", "float32", True),
        ("cpu", "float16", False),
        ("auto", "float32", False),
        ("cuda", "auto", False),
    ],
)
def test_cpu_cuda_precision_configuration_is_explicit(tmp_path, device, precision, valid):
    settings = _settings(tmp_path, device=device, precision=precision)
    if valid:
        provider = SiglipShadowProvider(settings, backend=FakeBackend())
        assert provider.runtime.device == device
        assert provider.runtime.precision == precision
    else:
        with pytest.raises(ValueError):
            SiglipShadowProvider(settings, backend=FakeBackend())


def test_fake_provider_makes_no_network_calls(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network API called")

    import socket
    import urllib.request

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    provider = SiglipShadowProvider(_settings(tmp_path), backend=FakeBackend())
    provider(RoutingInput(metadata={}, image=object()))


def test_backend_oom_is_not_retried_or_semantically_downgraded(tmp_path):
    class OomBackend:
        def __init__(self):
            self.calls = 0

        def score(self, image, prompts, *, processor, runtime):
            self.calls += 1
            raise RuntimeError("CUDA out of memory")

    backend = OomBackend()
    provider = SiglipShadowProvider(
        _settings(tmp_path, device="cuda", precision="float16", batch_size=16),
        backend=backend,
    )
    router = SceneRouter({"scene_routing": {"enabled": True, "mode": "shadow"}}, provider)
    record = router.route(RoutingInput(metadata={}, image=object()))
    assert backend.calls == 1
    assert record["scene_context"]["reasons"] == ["MODEL_UNAVAILABLE"]
