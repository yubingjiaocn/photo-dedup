"""Stage 1 wiring tests for decode-once, shadow-only scene routing."""

from __future__ import annotations

import json

from PIL import Image

from src import stage1_features as stage1
from src.config import load_config
from src.scene_router import SceneRouter


def _image_row(tmp_path, *, file_id=1, file_kind="jpg", **extra):
    path = tmp_path / f"image-{file_id}.jpg"
    Image.new("RGB", (32, 32), "white").save(path)
    return {"id": file_id, "path": str(path), "file_kind": file_kind, **extra}


def _routing(result):
    return json.loads(result[0]["quality_meta"])["routing"]


def test_stage1_decodes_once_and_fans_out_existing_rgb_without_router_opening(tmp_path, monkeypatch):
    row = _image_row(tmp_path)
    original = stage1._open_image_and_sha
    calls = 0

    def counted(path, telemetry=None):
        nonlocal calls
        calls += 1
        return original(path, telemetry)

    monkeypatch.setattr(stage1, "_open_image_and_sha", counted)
    result = stage1._process_batch(stage1.StubBackend(), [row], load_config())
    assert calls == 1
    assert _routing(result)["scene_context"]["reasons"] == ["ROUTER_DISABLED"]


def test_stage1_motion_photo_uses_metadata_gate_and_preserves_existing_quality_meta(tmp_path):
    row = _image_row(tmp_path, file_kind="jpg_motion", motion_partner_id=99)

    class Detector:
        def analyze(self, image, expected_face_count=None):
            return {"status": "ALL_OPEN", "sentinel": True}

    result = stage1._process_batch(
        stage1.StubBackend(), [row], load_config(), eye_detector=Detector()
    )
    meta = json.loads(result[0]["quality_meta"])
    assert meta["routing"]["scene_context"]["reasons"] == ["MOTION_PHOTO_BOUND_ASSET"]
    assert meta["eye_detection"] == {"status": "ALL_OPEN", "sentinel": True}
    assert "face_quality" in meta and "exposure" in meta and "detectors" in meta


def test_stage1_enabled_without_provider_records_model_unavailable(tmp_path):
    cfg = load_config()
    cfg.features["scene_routing"]["enabled"] = True
    result = stage1._process_batch(stage1.StubBackend(), [_image_row(tmp_path)], cfg)
    assert _routing(result)["scene_context"]["reasons"] == ["MODEL_UNAVAILABLE"]


def test_stage1_supplies_same_decoded_image_to_reused_router(tmp_path):
    row = _image_row(tmp_path)

    class CapturingRouter:
        def __init__(self):
            self.images = []
            self.delegate = SceneRouter(load_config())

        def route(self, routing_input):
            self.images.append(routing_input.image)
            return self.delegate.route(routing_input)

    router = CapturingRouter()
    stage1._process_batch(stage1.StubBackend(), [row], load_config(), scene_router=router)
    assert len(router.images) == 1
    assert router.images[0].mode == "RGB"
