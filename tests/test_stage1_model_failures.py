"""Normal download/load failures must not masquerade as usable Stage 1 results."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Config
from src.stage1_features import _load_yunet, resolve_backend


def test_auto_does_not_hide_torch_model_load_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    cfg = Config({"features": {"backend": "auto"}})
    monkeypatch.setattr("src.stage1_features.TorchBackend", lambda _cfg: (_ for _ in ()).throw(RuntimeError("weights unavailable")))
    with pytest.raises(RuntimeError, match="weights unavailable"):
        resolve_backend(cfg)


def test_yunet_download_failure_is_clear_error(monkeypatch, tmp_path: Path):
    cfg = Config({"paths": {"models_dir": str(tmp_path)}, "features": {"yunet_url": "https://example.invalid/yunet.onnx"}})
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(FaceDetectorYN=object()))
    monkeypatch.setitem(sys.modules, "urllib.request", SimpleNamespace(urlretrieve=lambda *_: (_ for _ in ()).throw(OSError("offline"))))
    with pytest.raises(RuntimeError, match="YuNet download failed"):
        _load_yunet(cfg)
