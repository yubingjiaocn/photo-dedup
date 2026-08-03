"""Configuration loading for the photo-dedup pipeline.

Loads ``config.yaml`` into a lightweight, attribute-accessible object with
sane defaults. Every stage takes an optional ``--config`` path so alternate
configs (e.g. test fixtures) can be swapped in without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = parent of the src/ directory that contains this file.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


# Defaults mirror config.yaml so the pipeline still runs if a key is missing.
_DEFAULTS: Dict[str, Any] = {
    "paths": {
        "root": "E:/Photos",
        "db": "./inventory.sqlite",
        "trash": "E:/Photos/_trash",
        "output_dir": "./output",
        "models_dir": "./models",
    },
    "scan": {
        "extensions": [".jpg", ".jpeg", ".png", ".mp4", ".mp", ".mov"],
        "follow_symlinks": False,
        "commit_every": 100,
        "embedded_head_bytes": 262144,
        "embedded_full_scan_max_bytes": 0,
    },
    "features": {
        "backend": "auto",
        "device": "cuda",
        "batch_size": 4,
        "max_inflight_megapixels": 80,
        "dinov2_model": "facebook/dinov2-base",
        "dinov2_input": 224,
        "iqa_musiq": True,
        "iqa_clipiqa": True,
        "yunet_input_width": 640,
        "yunet_score_threshold": 0.6,
        "exposure_long_edge": 512,
        "yunet_url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
    },
    "cluster": {
        "burst_window_seconds": 30,
        "dinov2_threshold": 0.92,
        "phash_hamming_threshold": 2,
        "face_pose_shift_ratio": 0.30,
        "min_face_score": 0.6,
        "enable_loose_similar": False,
        "loose_window_seconds": 300,
        "loose_dinov2_threshold": 0.96,
    },
    "quality": {
        "weight_iqa": 0.6,
        "weight_face": 0.3,
        "weight_resolution": 0.1,
        "resolution_ref_mp": 12.0,
    },
    "decision": {
        "profile": "balanced",
    },
    "execute": {
        "mode": "move",
        "dry_run": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Section:
    """Attribute + item access wrapper around a config sub-dict."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class Config:
    """Top-level config object. Access sections as attributes: ``cfg.cluster``."""

    def __init__(self, data: Dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source
        for section in ("paths", "scan", "features", "cluster", "quality", "decision", "execute"):
            setattr(self, section, Section(data.get(section, {})))

    # -- resolved paths -----------------------------------------------------
    def _resolve(self, value: str) -> Path:
        """Resolve a path from config; relative paths are relative to REPO_ROOT."""
        p = Path(value)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def db_path(self) -> Path:
        return self._resolve(self.paths.get("db", "./inventory.sqlite"))

    @property
    def root_path(self) -> Path:
        return Path(self.paths.get("root", "E:/Photos"))

    @property
    def trash_path(self) -> Path:
        return Path(self.paths.get("trash", "E:/Photos/_trash"))

    @property
    def output_dir(self) -> Path:
        return self._resolve(self.paths.get("output_dir", "./output"))

    @property
    def models_dir(self) -> Path:
        return self._resolve(self.paths.get("models_dir", "./models"))

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from ``path`` (or ``config.yaml``), merged over defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user_data: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
    merged = _deep_merge(_DEFAULTS, user_data)
    return Config(merged, source=cfg_path)
