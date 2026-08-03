"""Optional MediaPipe eye detector; inference is isolated from pure rules."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


class EyeState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    MAYBE = "MAYBE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EyeThresholds:
    open_max: float = 0.15
    closed_min: float = 0.50
    asymmetry_max: float = 0.25
    min_iod_px: float = 24.0
    min_face_px: float = 96.0
    smile_high: float = 0.45
    squint_high: float = 0.45


def classify_face(values: Mapping[str, Any], thresholds: EyeThresholds | None = None) -> dict[str, Any]:
    """Classify scalar face evidence without any MediaPipe dependency."""
    t = thresholds or EyeThresholds()
    result = dict(values)
    required = ("blink_left", "blink_right", "face_width_px", "iod_px")
    if any(values.get(key) is None for key in required):
        state, reasons = EyeState.UNKNOWN, ["MISSING_EVIDENCE"]
    elif float(values["face_width_px"]) < t.min_face_px or float(values["iod_px"]) < t.min_iod_px:
        state, reasons = EyeState.UNKNOWN, ["SMALL_FACE"]
    else:
        left, right = float(values["blink_left"]), float(values["blink_right"])
        if left <= t.open_max and right <= t.open_max:
            state, reasons = EyeState.OPEN, ["BOTH_EYES_OPEN"]
        elif left >= t.closed_min and right >= t.closed_min and abs(left - right) <= t.asymmetry_max:
            smile = max(_number(values.get("smile_left")), _number(values.get("smile_right")))
            squint = max(_number(values.get("squint_left")), _number(values.get("squint_right")))
            if smile >= t.smile_high and squint >= t.squint_high:
                state, reasons = EyeState.MAYBE, ["SMILE_SQUINT"]
            else:
                state, reasons = EyeState.CLOSED, ["BOTH_EYES_CLOSED"]
        elif (left >= t.closed_min) != (right >= t.closed_min) or abs(left - right) > t.asymmetry_max:
            state, reasons = EyeState.MAYBE, ["WINK_OR_ASYMMETRIC"]
        else:
            state, reasons = EyeState.MAYBE, ["BLINK_GRAY_ZONE"]
    result.update(state=state.value, reasons=reasons)
    return result


def aggregate_faces(faces: Sequence[Mapping[str, Any]], *, expected_face_count: int | None = None,
                    model_id: str | None = None) -> dict[str, Any]:
    serial = [dict(face) for face in faces]
    disagreement = expected_face_count is not None and expected_face_count != len(serial)
    states = {face.get("state") for face in serial}
    reasons = ["DETECTOR_COUNT_DISAGREEMENT"] if disagreement else []
    if disagreement:
        status = "UNKNOWN"
    elif not serial:
        status = "NO_FACE"
    elif EyeState.CLOSED.value in states:
        status = "HAS_CLOSED_EYES"
    elif EyeState.UNKNOWN.value in states:
        status = "UNKNOWN"
    elif EyeState.MAYBE.value in states:
        status = "HAS_MAYBE"
    else:
        status = "ALL_OPEN"
    return {"status": status, "faces": serial, "detected_face_count": len(serial),
            "detector_count_disagreement": disagreement, "reasons": reasons, "model_id": model_id}


def unavailable_result(reason: str = "DETECTOR_UNAVAILABLE") -> dict[str, Any]:
    return {"status": "UNKNOWN", "faces": [], "detected_face_count": 0,
            "detector_count_disagreement": False, "reasons": [reason], "model_id": None}


class MediaPipeEyeDetector:
    """Lazy optional wrapper. Model assets are never downloaded."""
    def __init__(self, settings: Mapping[str, Any], models_dir: Path) -> None:
        self.settings = dict(settings)
        configured = Path(str(self.settings.get("model_path", "face_landmarker.task")))
        self.model_path = configured if configured.is_absolute() else models_dir / configured
        self.expected_sha256 = str(self.settings.get("model_sha256", "") or "").lower()
        self.model_id: str | None = None
        self._landmarker: Any = None
        self.unavailable_reason: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self.unavailable_reason = "MODEL_NOT_FOUND"
            return
        digest = _sha256(self.model_path)
        if self.expected_sha256 and digest != self.expected_sha256:
            self.unavailable_reason = "MODEL_SHA256_MISMATCH"
            return
        self.model_id = f"mediapipe-face-landmarker@sha256:{digest}"
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=int(self.settings.get("max_faces", 10)),
                min_face_detection_confidence=float(self.settings.get("min_detection_confidence", 0.5)),
                min_face_presence_confidence=float(self.settings.get("min_presence_confidence", 0.5)),
                output_face_blendshapes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except (ImportError, ModuleNotFoundError):
            self.unavailable_reason = "MEDIAPIPE_NOT_INSTALLED"
        except Exception:
            self.unavailable_reason = "DETECTOR_INIT_FAILED"

    def analyze(self, image: Image.Image, expected_face_count: int | None = None) -> dict[str, Any]:
        if self._landmarker is None:
            return unavailable_result(self.unavailable_reason or "DETECTOR_UNAVAILABLE")
        try:
            raw_faces = self._infer(image)
            defaults = asdict(EyeThresholds())
            thresholds = EyeThresholds(**{key: self.settings[key] for key in defaults if key in self.settings})
            faces = [classify_face(face, thresholds) for face in raw_faces]
            return aggregate_faces(faces, expected_face_count=expected_face_count, model_id=self.model_id)
        except Exception:
            return unavailable_result("DETECTOR_INFERENCE_FAILED")

    def _infer(self, image: Image.Image) -> list[dict[str, Any]]:
        """Translate MediaPipe output into evidence; policy does not live here."""
        import mediapipe as mp
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        detected = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        h, w = rgb.shape[:2]
        output = []
        for index, landmarks in enumerate(detected.face_landmarks):
            scores = {c.category_name: float(c.score) for c in detected.face_blendshapes[index]}
            xs, ys = [p.x for p in landmarks], [p.y for p in landmarks]
            bbox = [min(xs) * w, min(ys) * h, (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h]
            iod = float(np.hypot((landmarks[263].x - landmarks[33].x) * w,
                                 (landmarks[263].y - landmarks[33].y) * h))
            output.append({"face_index": index, "bbox_xywh": bbox, "face_width_px": bbox[2],
                           "iod_px": iod, "blink_left": scores.get("eyeBlinkLeft"),
                           "blink_right": scores.get("eyeBlinkRight"),
                           "squint_left": scores.get("eyeSquintLeft"),
                           "squint_right": scores.get("eyeSquintRight"),
                           "smile_left": scores.get("mouthSmileLeft"),
                           "smile_right": scores.get("mouthSmileRight"), "model_id": self.model_id})
        return output


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
