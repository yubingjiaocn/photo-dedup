"""Offline, opt-in Stage 1 benchmark harness; it never scans a configured library."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from .config import load_config
from .stage1_features import StubBackend, _open_image_and_sha

STATE_VERSION = 2
FIXTURE_SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
    """An identifier that deliberately does not disclose user paths."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def _atomic_json(path: Path, data: Dict[str, Any]) -> None:
    """Durably replace JSON, leaving an older complete file intact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(data), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _fixture_paths(manifest: Path) -> Tuple[List[Path], str]:
    raw = manifest.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("fixture manifest must be valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError("fixture manifest has an unsupported schema_version")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixture manifest fixtures must be a non-empty list")
    root = manifest.parent.resolve()
    paths: List[Path] = []
    for item in fixtures:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("fixture entries require a string id")
        relative, buckets = item.get("file"), item.get("buckets")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("fixture file must be a non-empty relative path")
        if not isinstance(buckets, list) or not buckets or not all(isinstance(x, str) for x in buckets):
            raise ValueError("fixture buckets must be a non-empty string list")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("fixture file escapes manifest directory") from exc
        if candidate.suffix.lower() not in IMAGE_SUFFIXES or not candidate.is_file():
            raise ValueError("fixture manifest references a missing supported image")
        paths.append(candidate)
    return paths, hashlib.sha256(raw).hexdigest()


def _samples(sample_dir: Optional[str], fixture_manifest: Optional[str], limit: Optional[int]) -> Tuple[List[Path], Dict[str, Any]]:
    if bool(sample_dir) == bool(fixture_manifest):
        raise ValueError("provide exactly one of --sample-dir or --fixture-manifest")
    manifest_hash = None
    if sample_dir:
        root = Path(sample_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("--sample-dir must be an existing directory")
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        paths, manifest_hash = _fixture_paths(Path(str(fixture_manifest)).resolve())
    paths = paths[:limit] if limit is not None else paths
    if not paths:
        raise ValueError("sample selection contains no supported images")
    items = [{"id": _path_token(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    return paths, {"items": items, "manifest_hash": manifest_hash}


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


def _cuda() -> Any:
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _hardware() -> Dict[str, Any]:
    result: Dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        result.update(torch=torch.__version__, cuda_available=bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            result.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    except Exception:
        result["torch"] = None
    return result


def _phase(fn: Callable[[], Any]) -> Tuple[Dict[str, Any], Any]:
    """Time one phase and record a phase-local CUDA allocation peak."""
    torch = _cuda()
    if torch:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        value = fn()
        status = value.get("status", "PROCESSED") if isinstance(value, dict) else "PROCESSED"
        outcome: Dict[str, Any] = {"seconds": time.perf_counter() - started, "status": status,
                                   "processed": int(status != "SKIPPED"), "skipped": int(status == "SKIPPED"), "errors": 0}
    except Exception as exc:
        value = None
        outcome = {"seconds": time.perf_counter() - started, "status": "ERROR", "processed": 0,
                   "skipped": 0, "errors": 1, "error": type(exc).__name__}
    if torch:
        torch.cuda.synchronize()
        outcome["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2)
    else:
        outcome["peak_gpu_mb"] = None
    return outcome, value


def _skipped(reason: str) -> Dict[str, Any]:
    return {"seconds": 0.0, "status": "SKIPPED", "processed": 0, "skipped": 1, "errors": 0,
            "reason": reason, "peak_gpu_mb": None}


def run(*, sample_dir: Optional[str] = None, fixture_manifest: Optional[str] = None,
        output: str = "benchmark-result.json", state: str = "benchmark-state.json",
        limit: Optional[int] = None, dry_run: bool = False, runner_spec: Optional[str] = None,
        config_path: Optional[str] = None) -> Dict[str, Any]:
    """Benchmark only explicit samples; state identity mismatch safely restarts."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    paths, selection = _samples(sample_dir, fixture_manifest, limit)
    cfg, runner = load_config(config_path), _load_runner(runner_spec)
    identity = {"state_version": STATE_VERSION, "schema_version": 1, "config_hash": _hash(cfg.as_dict()),
                "sample_set_hash": _hash(selection), "model_hash": _hash(getattr(runner, "model_descriptor", {})),
                "runner_spec": runner_spec or "builtin"}
    result_path, state_path = Path(output), Path(state)
    header = {**identity, "created_at": int(time.time()), "sample_count": len(paths), "hardware": _hardware(), "dry_run": dry_run}
    if dry_run:
        data = {**header, "status": "DRY_RUN", "samples": [{"id": _path_token(p)} for p in paths]}
        _atomic_json(result_path, data)
        return data
    completed: Dict[str, Any] = {}
    resume_status = "NEW"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if all(previous.get(k) == v for k, v in identity.items()) and isinstance(previous.get("completed"), dict):
            completed, resume_status = previous["completed"], "RESUMED"
        else:
            resume_status = "STATE_MISMATCH_RESTARTED"
    for path in paths:
        token = _path_token(path)
        if token in completed:
            continue
        decode, image = _phase(lambda p=path: _open_image_and_sha(p)[0])
        if decode["errors"]:
            phases = {"decode": decode, "siglip": _skipped("DECODE_ERROR"), "stage1": _skipped("DECODE_ERROR")}
        else:
            siglip, _ = _phase(lambda: runner.siglip(image))
            stage1, _ = _phase(lambda: runner.stage1(image))
            phases = {"decode": decode, "siglip": siglip, "stage1": stage1}
        completed[token] = phases
        _atomic_json(state_path, {**identity, "completed": completed})
    summaries: Dict[str, Any] = {}
    for name in ("decode", "siglip", "stage1"):
        entries = [completed[_path_token(p)][name] for p in paths]
        seconds, processed = sum(float(x["seconds"]) for x in entries), sum(int(x["processed"]) for x in entries)
        summaries[name] = {"seconds": seconds, "processed": processed, "skipped": sum(int(x["skipped"]) for x in entries),
                           "errors": sum(int(x["errors"]) for x in entries),
                           "throughput_per_s": processed / seconds if processed and seconds else None,
                           "peak_gpu_mb": max((x["peak_gpu_mb"] for x in entries if x["peak_gpu_mb"] is not None), default=None)}
    data = {**header, "status": "COMPLETE", "resume_status": resume_status, "phases": summaries, "samples": completed}
    _atomic_json(result_path, data)
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
