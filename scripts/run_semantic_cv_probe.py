"""Run the bounded local-CV semantic probe and build labelled contact sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exposure, quality
from src.eye_detection import MediaPipeEyeDetector
from src.semantic_observer import frame_observations

MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
RESOLUTION_GATE_GROUPS = {
    ("2026-03-29", "G012"),
    ("2026-05-27", "G005"),
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _select_groups(holdout: Path) -> list[dict]:
    evaluation = _json(holdout / "evaluation/summary.json")
    selected = []
    for row in evaluation["rows"]:
        metrics = row["metrics"]["A_off"]
        issue = any(metrics[key] for key in (
            "phase_misses", "quality_wrong", "selected_unacceptable",
        ))
        if issue or row["regime"] == "multi":
            selected.append({
                "day": row["day"], "group_alias": row["group_alias"],
                "regime": row["regime"], "provisional_issue": issue,
                "provisional_metrics": {
                    key: metrics[key] for key in (
                        "phase_misses", "quality_wrong", "selected_unacceptable",
                    )
                },
            })
    if len(selected) != 8:
        raise RuntimeError(f"expected bounded 8-group sample, got {len(selected)}")
    return selected


def _mappings(holdout: Path) -> dict[tuple[str, str], dict]:
    output = {}
    for day in ("2026-03-29", "2026-05-27"):
        for item in _json(holdout / f"panels/{day}/mapping.json"):
            output[(day, item["group_alias"])] = item
    return output


def _yunet(path: Path):
    import cv2

    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 5000)


def _detect_yunet(detector, image: Image.Image) -> list[dict]:
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    height, width = bgr.shape[:2]
    scale = 640 / max(height, width) if max(height, width) > 640 else 1.0
    if scale != 1.0:
        bgr = cv2.resize(bgr, (max(1, int(width * scale)), max(1, int(height * scale))))
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, detections = detector.detect(bgr)
    if detections is None:
        return []
    faces = []
    for detection in detections:
        faces.append({
            "bbox": [float(value) / scale for value in detection[:4]],
            "landmarks": [float(value) / scale for value in detection[4:14]],
            "score": float(detection[14]),
        })
    return faces


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), (size[0] - 12, size[1] - 42))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, 36))
    return canvas


def _face_crop(image: Image.Image, face: dict) -> Image.Image:
    x, y, width, height = map(float, face["bbox_xywh"])
    pad = 0.35 * max(width, height)
    box = (
        max(0, int(x - pad)), max(0, int(y - pad)),
        min(image.width, int(x + width + pad)),
        min(image.height, int(y + height + pad)),
    )
    return image.crop(box)


def _sheet(group_id: str, frames: list[dict], destination: Path, width: int = 2048) -> None:
    columns = min(3, max(1, len(frames)))
    cell_width = width // columns
    cell_height = 620
    rows = (len(frames) + columns - 1) // columns
    face_cell = min(330, width // max(1, len(frames)))
    height = 70 + rows * cell_height + face_cell + 65
    sheet = Image.new("RGB", (width, height), "#ececec")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=20)
    draw.text((20, 18), f"GROUP {group_id} | full frames + largest detected face crop", fill="black", font=font)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        tile = _fit(frame["image"], (cell_width - 12, cell_height - 12))
        x, y = column * cell_width + 6, 70 + row * cell_height + 6
        sheet.paste(tile, (x, y))
        draw.text((x + 10, y + 6), frame["alias"], fill="black", font=font,
                  stroke_width=3, stroke_fill="white")
    face_y = 70 + rows * cell_height + 8
    for index, frame in enumerate(frames):
        x = index * face_cell + 6
        faces = frame["eye_result"].get("faces") or []
        if faces:
            face = max(faces, key=lambda item: float(item.get("face_width_px") or 0))
            crop = _fit(_face_crop(frame["image"], face), (face_cell - 12, face_cell - 12))
            sheet.paste(crop, (x, face_y))
            state = "CROP"
        elif frame.get("yunet_faces"):
            region = max(frame["yunet_faces"], key=lambda item: item["bbox"][2] * item["bbox"][3])
            x0, y0, bw, bh = region["bbox"]
            face = {"bbox_xywh": [x0, y0, bw, bh]}
            crop = _fit(_face_crop(frame["image"], face), (face_cell - 12, face_cell - 12))
            sheet.paste(crop, (x, face_y))
            state = "CROP"
        else:
            state = "NO_FACE"
        draw.text((x + 8, face_y + 4), f"{frame['alias']} FACE {state}", fill="black",
                  font=small, stroke_width=2, stroke_fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=88, optimize=True)


def run(root: Path, model_dir: Path, output: Path) -> dict:
    holdout = root / "holdout"
    selected = _select_groups(holdout)
    mappings = _mappings(holdout)
    output.mkdir(parents=True, exist_ok=True)
    yunet = _yunet(Path("models/face_detection_yunet_2023mar.onnx"))
    started = time.perf_counter()
    total_frames = 0
    status_counts: dict[str, int] = {}
    group_results = []
    with MediaPipeEyeDetector({
        "model_path": "face_landmarker.task", "model_sha256": MODEL_SHA256,
        "max_faces": 10,
    }, model_dir) as eye_detector:
        if eye_detector._landmarker is None:
            raise RuntimeError(eye_detector.unavailable_reason or "eye detector unavailable")
        for selected_group in selected:
            mapping = mappings[(selected_group["day"], selected_group["group_alias"])]
            rendered_frames = []
            observations = []
            for frame in mapping["frames"]:
                path = Path(frame["path"])
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                faces = _detect_yunet(yunet, image)
                eye_result = eye_detector.analyze(
                    image, expected_face_count=len(faces), face_regions=faces
                )
                rgb = np.asarray(image, dtype=np.uint8)
                face_quality = quality.compute_face_quality(rgb, faces)
                gray = rgb.mean(axis=2)
                sharpness = quality.variance_of_laplacian(gray)
                exposure_metrics = exposure.extract(image, faces)
                exposure_class = exposure.classify(exposure_metrics)
                normalized = frame_observations(
                    frame_alias=frame["alias"], image_size=image.size,
                    eye_result=eye_result, yunet_faces=faces,
                    face_quality=face_quality, sharpness=sharpness,
                    exposure_metrics=exposure_metrics, exposure_class=exposure_class,
                )
                normalized["image_sha256"] = _sha256(path)
                observations.append(normalized)
                rendered_frames.append({
                    "alias": frame["alias"], "image": image,
                    "eye_result": eye_result, "yunet_faces": faces,
                })
                total_frames += 1
                status = eye_result["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            group_id = f"{selected_group['day']}-{selected_group['group_alias']}"
            high = output / "contact-sheets" / f"{group_id}-2048.jpg"
            _sheet(group_id, rendered_frames, high, 2048)
            low = None
            if (selected_group["day"], selected_group["group_alias"]) in RESOLUTION_GATE_GROUPS:
                low = output / "contact-sheets" / f"{group_id}-1024.jpg"
                with Image.open(high) as source:
                    resized = source.resize((1024, round(source.height * 1024 / source.width)), Image.Resampling.LANCZOS)
                    resized.save(low, "JPEG", quality=88, optimize=True)
            group_results.append({
                **selected_group, "member_count": len(observations),
                "frames": observations,
                "contact_sheet_2048": str(high.relative_to(output)),
                "contact_sheet_1024": str(low.relative_to(output)) if low else None,
            })
    report = {
        "schema_version": 1,
        "mode": "shadow_review_only",
        "sample_policy": "all 7 provisional-issue groups plus the only multi group",
        "groups": len(group_results), "frames": total_frames,
        "eye_status_counts": status_counts,
        "model": {
            "mediapipe_version": __import__("mediapipe").__version__,
            "face_landmarker_sha256": MODEL_SHA256,
            "yunet_sha256": _sha256(Path("models/face_detection_yunet_2023mar.onnx")),
        },
        "runtime_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "groups_detail": group_results,
        "invariants": {
            "changes_keeper": False, "changes_sorting": False,
            "changes_routing": False, "deletion_authority": "none",
        },
    }
    (output / "cv-observations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.model_dir, args.output)
    print(json.dumps({
        key: report[key] for key in (
            "groups", "frames", "eye_status_counts", "model", "runtime_seconds",
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
