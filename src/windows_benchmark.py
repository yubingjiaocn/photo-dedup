"""Offline, opt-in Stage 1 benchmark harness; never scans a configured library."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from PIL import Image

from .config import load_config
from .stage1_features import StubBackend, _open_image_and_sha

STATE_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _path_token(path: Path) -> str:
    """A result identifier that deliberately does not disclose user paths."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def _samples(sample_dir: Optional[str], fixture_manifest: Optional[str], limit: Optional[int]) -> List[Path]:
    if bool(sample_dir) == bool(fixture_manifest):
        raise ValueError("provide exactly one of --sample-dir or --fixture-manifest")
    if sample_dir:
        root = Path(sample_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("--sample-dir must be an existing directory")
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        manifest = Path(str(fixture_manifest)).resolve()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [manifest.parent / item["file"] for item in data["fixtures"]]
        if any(not p.is_file() for p in paths):
            raise ValueError("fixture manifest references a missing file")
    if not paths:
        raise ValueError("sample selection contains no supported images")
    return paths[:limit] if limit is not None else paths


class BuiltinRunner:
    """Offline fallback: real decode and existing deterministic Stage 1 backend."""
    model_descriptor = {"siglip": "unavailable", "stage1": "StubBackend-v1"}

    def siglip(self, image: Image.Image) -> Any:
        del image
        return {"status": "SKIPPED", "reason": "NO_LOCAL_PROVIDER"}

    def stage1(self, image: Image.Image) -> Any:
        backend = StubBackend()
        embedding = backend.embed_batch([image])[0]
        quality, _ = backend.quality(image)
        return {"embedding_dimensions": int(len(embedding)), "quality": quality}


def _load_runner(spec: Optional[str]) -> Any:
    if not spec:
        return BuiltinRunner()
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError("--runner must be module:factory")
    factory: Callable[[], Any] = getattr(importlib.import_module(module), name)
    return factory()


def _gpu_peak_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**2)
    except Exception:
        pass
    return None


def _hardware() -> Dict[str, Any]:
    result: Dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["cuda"] = torch.version.cuda
    except Exception:
        result["torch"] = None
    return result


def _phase(fn: Callable[[], Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        fn()
        return {"seconds": time.perf_counter() - started, "errors": 0}
    except Exception as exc:
        return {"seconds": time.perf_counter() - started, "errors": 1, "error": type(exc).__name__}


def run(*, sample_dir: Optional[str] = None, fixture_manifest: Optional[str] = None,
        output: str = "benchmark-result.json", state: str = "benchmark-state.json",
        limit: Optional[int] = None, dry_run: bool = False, runner_spec: Optional[str] = None,
        config_path: Optional[str] = None) -> Dict[str, Any]:
    """Benchmark only explicitly supplied samples and atomically checkpoint progress."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    paths = _samples(sample_dir, fixture_manifest, limit)
    cfg = load_config(config_path)
    config_hash = _hash(cfg.as_dict())
    result_path, state_path = Path(output), Path(state)
    runner = _load_runner(runner_spec)
    header = {
        "schema_version": 1, "created_at": int(time.time()), "sample_count": len(paths),
        "config_hash": config_hash, "model_hash": _hash(getattr(runner, "model_descriptor", {})),
        "hardware": _hardware(), "dry_run": dry_run,
    }
    if dry_run:
        data = {**header, "status": "DRY_RUN", "samples": [{"id": _path_token(p)} for p in paths]}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
        return data
    completed: Dict[str, Any] = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") == config_hash:
            completed = previous.get("completed", {})
    for path in paths:
        token = _path_token(path)
        if token in completed:
            continue
        try:
            decode_started = time.perf_counter()
            image, _ = _open_image_and_sha(path)
            phases = {"decode": {"seconds": time.perf_counter() - decode_started, "errors": 0}}
            phases["siglip"] = _phase(lambda: runner.siglip(image))
            phases["stage1"] = _phase(lambda: runner.stage1(image))
        except KeyboardInterrupt:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"version": STATE_VERSION, "config_hash": config_hash, "completed": completed}), encoding="utf-8")
            raise
        except Exception as exc:
            phases = {name: {"seconds": 0.0, "errors": 1, "error": type(exc).__name__}
                      for name in ("decode", "siglip", "stage1")}
        completed[token] = phases
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"version": STATE_VERSION, "config_hash": config_hash, "completed": completed}), encoding="utf-8")
    summaries: Dict[str, Any] = {}
    for name in ("decode", "siglip", "stage1"):
        entries = [completed[_path_token(p)][name] for p in paths]
        seconds = sum(float(x["seconds"]) for x in entries)
        summaries[name] = {"seconds": seconds, "throughput_per_s": len(entries) / seconds if seconds else None,
                           "errors": sum(int(x["errors"]) for x in entries), "peak_gpu_mb": _gpu_peak_mb()}
    data = {**header, "status": "COMPLETE", "phases": summaries, "samples": completed}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    return data


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline opt-in Windows Stage 1 benchmark")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-dir")
    source.add_argument("--fixture-manifest")
    ap.add_argument("--output", default="benchmark-result.json")
    ap.add_argument("--state", default="benchmark-state.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runner", dest="runner_spec", help="offline injection: module:factory")
    ap.add_argument("--config", dest="config_path")
    args = ap.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
