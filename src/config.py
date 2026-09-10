"""Configuration loading for the photo-dedup pipeline.

Loads ``config.yaml`` into a lightweight, attribute-accessible object with
sane defaults. Every stage takes an optional ``--config`` path so alternate
configs (e.g. test fixtures) can be swapped in without touching code.

Defaults vs. declarations
-------------------------
``_DEFAULTS`` exists so a config missing a key still runs. That is harmless for
thresholds and dangerous for exactly one key: ``paths.root``. The default
``E:/Photos`` is a placeholder, not a statement about this machine's library, so a
stage must never adopt it as the photo root an output directory is *bound* to
(see :mod:`src.root_scope`). :attr:`Config.declared_root` therefore reports a
root only when the config actually contained one, and ``None`` when the value
came from the default merge.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, Optional

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
        # Bounded CPU prefetch; 0 for either knob restores the single-threaded
        # pipeline. iqa_batch_size groups by identical bounded shape (padding
        # would change scores). See docs/STAGE1_THROUGHPUT.md.
        "cpu_workers": 4,
        "prefetch_batches": 2,
        "iqa_batch_size": 4,
        "max_process_megapixels": 64,
        "max_process_aspect_ratio": 3.0,
        "iqa_max_long_edge": 1920,
        "thumbnails": {
            "enabled": True,
            "max_px": 320,
            "jpeg_quality": 80,
        },
        # Stage 1 phase telemetry (cheap: one perf_counter pair per instrumented
        # call). CUDA-event sampling is opt-in (0 = off) so the default run has
        # no observer effect at all.
        "telemetry": {
            "enabled": True,
            "gpu_event_every": 0,
        },
        "dinov2_model": "facebook/dinov2-base",
        "dinov2_input": 224,
        "iqa_musiq": True,
        "iqa_clipiqa": True,
        "yunet_input_width": 640,
        "yunet_score_threshold": 0.6,
        "exposure_long_edge": 512,
        "eye_detection": {
            "enabled": False,
            "mode": "shadow",
            "model_path": "face_landmarker.task",
            "model_sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
            "max_faces": 10,
        },
        "face_identity": {
            "enabled": False,
            "mode": "shadow",
            "model_path": "face_recognition_sface_2021dec.onnx",
        },
        "scene_routing": {
            "enabled": False,
            "mode": "shadow",
            "model": {},
            "prompt": {},
            "runtime": {},
        },
        "yunet_url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
    },
    "cluster": {
        "burst_window_seconds": 30,  # Visual grouping time window (30s)
        "dinov2_threshold": 0.92,
        "phash_hamming_threshold": 2,
        "face_pose_shift_ratio": 0.30,
        "min_face_score": 0.6,
        "require_face_identity": False,
        "face_identity_cosine_threshold": 0.45,
        "face_identity_min_width_px": 80.0,
        "enable_loose_similar": False,
        "loose_window_seconds": 300,
        "loose_dinov2_threshold": 0.96,
        "phase_max_gap_seconds": 4,
        "phase_embedding_boundary": 0.93,
        "phase_position_shift": 0.22,
        "keepers_per_phase": 1,
        "max_group_keepers": 3,
        "keeper_diversity_similarity": 0.965,
        "keeper_mmr_quality_weight": 0.7,
        "keeper_score_policy": "per_member",
        "keeper_score_change_margin": 0.06,
        "keeper_diversity_policy": "mmr",
        "keeper_diversity_quality_slack": 0.04,
        "keeper_instance_recovery_policy": "off",
        "keeper_pose_policy": "off",
        "keeper_pose_displacement_threshold": 0.4,
        "keeper_local_quality_policy": "off",
        "keeper_local_quality_similarity": 0.9,
        "keeper_local_quality_penalty": 0.08,
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
    """Top-level config object. Access sections as attributes: ``cfg.cluster``.

    ``declared`` is the *unmerged* user data, which is how :attr:`declared_root`
    tells "the user wrote E:/Photos" apart from "nobody wrote anything, so the
    default filled in E:/Photos". Building a ``Config`` directly (tests, tooling)
    passes no ``declared``; then the data *is* the declaration, because no
    default merge happened that could be mistaken for one.
    """

    def __init__(self, data: Dict[str, Any], source: Path | None = None,
                 declared: Dict[str, Any] | None = None) -> None:
        self._data = data
        self._declared = data if declared is None else declared
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
        """Effective scan root, default included (Stage 0's ``--root`` fallback)."""
        return Path(self.paths.get("root", "E:/Photos"))

    @property
    def declared_root(self) -> Optional[Path]:
        """``paths.root`` as actually written in the config, else ``None``.

        Stages 1-3 use this: they may bind an unbound output directory to a root
        the user declared, but never to the ``E:/Photos`` placeholder that the
        default merge would otherwise hand them.
        """
        section = self._declared.get("paths")
        if not isinstance(section, dict):
            return None
        value = section.get("root")
        if value is None or not str(value).strip():
            return None
        return Path(str(value))

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

    def validate(self) -> None:
        """Fail closed on invalid config values before any stage runs.

        Call this after load_config() in CLI entry points to reject invalid
        thresholds early with clear error messages.
        """
        # phash_hamming_threshold must be in [0, 2] range
        threshold = self.cluster.get("phash_hamming_threshold")
        if threshold is not None:
            threshold_int = int(threshold)
            if threshold_int < 0 or threshold_int > 2:
                raise ValueError(
                    f"cluster.phash_hamming_threshold={threshold_int} is out of valid range [0, 2]. "
                    f"Current band-prefilter algorithm only supports hamming thresholds 0-2. "
                    f"Set to 2 (default) for near-duplicate detection."
                )
        identity = self.features.get("face_identity", {})
        if self.cluster.get("require_face_identity", False):
            if not isinstance(identity, dict) or identity.get("enabled") is not True:
                raise ValueError(
                    "cluster.require_face_identity=true requires "
                    "features.face_identity.enabled=true"
                )
        identity_threshold = float(self.cluster.get("face_identity_cosine_threshold", 0.45))
        if not -1.0 <= identity_threshold <= 1.0:
            raise ValueError("cluster.face_identity_cosine_threshold must be in [-1, 1]")
        if float(self.cluster.get("face_identity_min_width_px", 80.0)) <= 0:
            raise ValueError("cluster.face_identity_min_width_px must be > 0")
        score_policy = self.cluster.get("keeper_score_policy", "per_member")
        if not isinstance(score_policy, str) or score_policy not in {"per_member", "common_evidence", "guarded_common"}:
            raise ValueError("cluster.keeper_score_policy must be per_member, common_evidence or guarded_common")
        margin = self.cluster.get("keeper_score_change_margin", 0.06)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= 1:
            raise ValueError("cluster.keeper_score_change_margin must be a finite number in [0, 1]")
        diversity_policy = self.cluster.get("keeper_diversity_policy", "mmr")
        if not isinstance(diversity_policy, str) or diversity_policy not in {"mmr", "quality_banded"}:
            raise ValueError("cluster.keeper_diversity_policy must be mmr or quality_banded")
        slack = self.cluster.get("keeper_diversity_quality_slack", 0.04)
        if isinstance(slack, bool) or not isinstance(slack, (int, float)) or not 0 <= slack <= 1:
            raise ValueError("cluster.keeper_diversity_quality_slack must be finite and in [0, 1]")
        local_policy = self.cluster.get("keeper_local_quality_policy", "off")
        if not isinstance(local_policy, str) or local_policy not in {"off", "dominance", "region_set"}:
            raise ValueError("cluster.keeper_local_quality_policy must be off, dominance or region_set")
        for key, upper, default in [("keeper_local_quality_similarity", 1.0, 0.9),
                                    ("keeper_local_quality_penalty", 0.25, 0.08)]:
            value = self.cluster.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= upper):
                raise ValueError(f"cluster.{key} is outside its finite range")
        if local_policy == "region_set" and self.cluster.get("keeper_local_quality_similarity", 0.9) < 0.9:
            raise ValueError("region_set requires similarity >= 0.9")
        recovery_policy = self.cluster.get("keeper_instance_recovery_policy", "off")
        if not isinstance(recovery_policy, str) or recovery_policy not in {"off", "review_only"}:
            raise ValueError("cluster.keeper_instance_recovery_policy must be off or review_only")
        pose_policy = self.cluster.get("keeper_pose_policy", "off")
        if not isinstance(pose_policy, str) or pose_policy not in {"off", "consensus", "stable_actor"}:
            raise ValueError("cluster.keeper_pose_policy must be off, consensus or stable_actor")
        pose_threshold = self.cluster.get("keeper_pose_displacement_threshold", 0.4)
        if (isinstance(pose_threshold, bool) or not isinstance(pose_threshold, (int, float))
                or not math.isfinite(pose_threshold) or pose_threshold <= 0):
            raise ValueError("cluster.keeper_pose_displacement_threshold must be positive and finite")
        if int(self.cluster.get("phase_max_gap_seconds", 4)) <= 0:
            raise ValueError("cluster.phase_max_gap_seconds must be > 0")
        for key in ("phase_embedding_boundary", "phase_position_shift", "keeper_diversity_similarity",
                    "keeper_mmr_quality_weight"):
            value = float(self.cluster.get(key))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"cluster.{key} must be in [0, 1]")
        if int(self.cluster.get("keepers_per_phase", 1)) < 1:
            raise ValueError("cluster.keepers_per_phase must be >= 1")
        if int(self.cluster.get("max_group_keepers", 3)) < 1:
            raise ValueError("cluster.max_group_keepers must be >= 1")


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from ``path`` (or ``config.yaml``), merged over defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user_data: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
    merged = _deep_merge(_DEFAULTS, user_data)
    return Config(merged, source=cfg_path, declared=user_data)
