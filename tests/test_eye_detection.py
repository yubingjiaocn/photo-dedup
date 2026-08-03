import json

from PIL import Image

from src import eye_detection as eyes
from src import stage1_features as stage1
from src.config import load_config


def face(**overrides):
    values = {"face_index": 0, "face_width_px": 200, "iod_px": 60,
              "blink_left": 0.05, "blink_right": 0.08,
              "squint_left": 0.1, "squint_right": 0.1,
              "smile_left": 0.1, "smile_right": 0.1, "model_id": "test"}
    values.update(overrides)
    return eyes.classify_face(values)


def test_open():
    assert face()["state"] == "OPEN"


def test_closed():
    assert face(blink_left=0.7, blink_right=0.75)["state"] == "CLOSED"


def test_smile_squint_is_maybe():
    result = face(blink_left=0.7, blink_right=0.7, smile_left=0.6, squint_right=0.6)
    assert (result["state"], result["reasons"]) == ("MAYBE", ["SMILE_SQUINT"])


def test_wink_is_maybe():
    assert face(blink_left=0.8, blink_right=0.1)["reasons"] == ["WINK_OR_ASYMMETRIC"]


def test_small_face_is_unknown():
    assert face(face_width_px=80)["state"] == "UNKNOWN"


def test_missing_is_unknown():
    assert face(blink_left=None)["reasons"] == ["MISSING_EVIDENCE"]


def test_unavailable_without_model(tmp_path):
    detector = eyes.MediaPipeEyeDetector({"model_path": "missing.task"}, tmp_path)
    result = detector.analyze(Image.new("RGB", (10, 10)))
    assert result["status"] == "UNKNOWN"
    assert result["reasons"] == ["MODEL_NOT_FOUND"]


def test_stage1_metadata_integration(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)

    class Detector:
        def analyze(self, image, expected_face_count=None):
            assert image.size == (32, 32)
            return eyes.aggregate_faces([face()], model_id="test")

    rows = [{"id": 1, "path": str(image_path)}]
    result = stage1._process_batch(stage1.StubBackend(), rows, load_config(), eye_detector=Detector())
    metadata = json.loads(result[0]["quality_meta"])
    assert metadata["eye_detection"]["status"] == "ALL_OPEN"
