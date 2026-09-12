"""Two-frame native DINO patch correspondence spike; no runtime admission.

Uses existing DINOv2 + YOLO weights. Tests local descriptors because the masked
CLS summary failed; does not lower the frozen v0 identity or keeper thresholds.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.config import Config, load_config  # noqa: E402
from src.stage1_backends import TorchBackend  # noqa: E402
from src.instance_correspondence import mask_for  # noqa: E402
from src.local_quality import _iou  # noqa: E402
from run_foreground_correspondence import detect_view, choose, save  # noqa: E402


def dense(backend, image, obj):
    torch = backend.torch
    box = tuple(round(x) for x in obj["box"])
    crop = image.crop(box)
    patch = int(backend.model.config.patch_size)
    gh, gw = crop.height // patch, crop.width // patch
    if gh * gw > 8192 or min(crop.size) < 96:
        return None, "NATIVE_DENSE_CAPACITY_OR_SIZE"
    pixels = backend.processor(
        images=[crop], do_resize=False, do_center_crop=False, return_tensors="pt"
    )["pixel_values"]
    assert tuple(pixels.shape[-2:]) == (crop.height, crop.width)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        values = backend.model(
            pixel_values=pixels.to(backend.device)
        ).last_hidden_state[0, 1:]
    assert len(values) == gh * gw
    vectors = values.float().cpu().numpy()
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    polygon = np.asarray(obj["polygon"]) - np.array(box[:2])
    mask = mask_for((crop.height, crop.width), polygon)
    gray = np.asarray(crop.convert("L"))
    keep = []
    points = []
    for y in range(gh):
        for x in range(gw):
            region = mask[y * patch : (y + 1) * patch, x * patch : (x + 1) * patch]
            texture = gray[y * patch : (y + 1) * patch, x * patch : (x + 1) * patch]
            if (
                np.count_nonzero(region) / region.size >= 0.75
                and float(texture.std()) >= 8
            ):
                keep.append(y * gw + x)
                points.append([box[0] + (x + 0.5) * patch, box[1] + (y + 0.5) * patch])
    if len(keep) < 24:
        return None, "NATIVE_FOREGROUND_LOW_TEXTURE"
    take = np.linspace(0, len(keep) - 1, min(512, len(keep))).astype(int)
    return {
        "vectors": vectors[np.asarray(keep)[take]],
        "points": np.asarray(points, dtype=np.float32)[take],
        "native_grid": [gw, gh],
        "native_patch_size": patch,
        "foreground_textured_patches": len(keep),
    }, None


def compare(a, b, oa, ob):
    import cv2

    if a is None or b is None:
        return {"eligible": False, "reason": "NO_NATIVE_DESCRIPTORS", "score": 0.0}
    sim = a["vectors"] @ b["vectors"].T
    columns = np.argmax(sim, axis=1)
    rows = np.argmax(sim, axis=0)
    source = []
    target = []
    scores = []
    for i, j in enumerate(columns):
        value = float(sim[i, j])
        alternatives_row = np.partition(sim[i], -2)[-2]
        alternatives_col = np.partition(sim[:, j], -2)[-2]
        if (
            rows[j] == i
            and value >= 0.9
            and value - alternatives_row >= 0.01
            and value - alternatives_col >= 0.01
        ):
            source.append(i)
            target.append(j)
            scores.append(value)
    result = {
        "eligible": False,
        "reason": "PATCH_RECIPROCAL_SUPPORT_LOW",
        "score": 0.0,
        "matches": len(source),
        "source_patches": len(a["vectors"]),
        "target_patches": len(b["vectors"]),
    }
    if len(source) < 12:
        return result
    p = a["points"][source]
    q = b["points"][target]
    matrix, inliers = cv2.estimateAffinePartial2D(
        p, q, method=cv2.RANSAC, ransacReprojThreshold=28, maxIters=2000
    )
    if matrix is None or inliers is None:
        return {**result, "reason": "PATCH_GEOMETRY_UNRESOLVED"}
    valid = inliers.ravel().astype(bool)
    n = int(valid.sum())
    areas = [
        (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) for o in [oa, ob]
    ]
    coverage = [
        float(np.prod(np.ptp(points[valid], axis=0)) / area) if n else 0.0
        for points, area in zip([p, q], areas)
    ]
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    fraction = n / min(len(a["vectors"]), len(b["vectors"]))
    eligible = (
        n >= 12
        and n / len(source) >= 0.65
        and min(coverage) >= 0.15
        and 0.8 <= scale <= 1.25
        and fraction >= 0.08
    )
    return {
        **result,
        "eligible": bool(eligible),
        "reason": "NATIVE_PATCH_SUPPORT"
        if eligible
        else "PATCH_DRIFT_OR_LOCAL_FRAGMENT",
        "inliers": n,
        "inlier_ratio": n / len(source),
        "coverage": coverage,
        "scale": scale,
        "score": fraction,
        "median_similarity": float(np.median(scores)),
        "source_inlier_centers": p[valid][:64].tolist(),
        "target_inlier_centers": q[valid][:64].tolist(),
        "whole_body_completeness": "unknown",
        "identity_authority": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    a = p.parse_args()
    if not a.model.is_file():
        raise ValueError("Existing model required")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "summary.json").exists():
        raise ValueError("Completed spike exists; inspect it")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YOLO_CONFIG_DIR=str(a.output / "ultralytics-settings"),
    )
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    detector = YOLO(str(a.model))
    group = next(
        g
        for g in json.loads((a.root / "slice.json").read_text())["groups"]
        if g["group_alias"] == "G018"
    )
    cfg = load_config(group["config"]).as_dict()
    cfg["features"].update(iqa_musiq=False, iqa_clipiqa=False)
    backend = TorchBackend(Config(cfg))
    objects = []
    descriptors = []
    for frame in group["frames"][:2]:
        with Image.open(frame["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = image.size
        windows = [[0, 0, w, h]]
        for y1, y2 in [(0, int(h * 0.6)), (int(h * 0.4), h)]:
            for x1, x2 in [(0, int(w * 0.6)), (int(w * 0.4), w)]:
                windows.append([x1, y1, x2, y2])
        original = json.loads(
            (
                a.root / f"next-class-scale/2026-07-25-G018-{frame['alias']}.json"
            ).read_text()
        )
        obs = []
        desc = []
        frame_out = a.output / frame["alias"]
        frame_out.mkdir(exist_ok=True)
        for index, seed in enumerate(original["observations"]):
            if seed["class_id"] not in (0, 77):
                continue
            candidates = detect_view(
                detector,
                image,
                windows[seed["window"]],
                frame_out / f"window-{seed['window']}.json",
            )
            selected, reason = choose(seed, candidates)
            if selected is None:
                continue
            selected = {
                **selected,
                "source_observation_index": index,
                "detector_confirmed": True,
                "whole_body_completeness": "unknown",
                "frame": frame["alias"],
            }
            cache = frame_out / f"dense-{index}.npz"
            if cache.exists():
                with np.load(cache) as loaded:
                    value = {k: loaded[k] for k in loaded.files}
                why = None
            else:
                value, why = dense(backend, image, selected)
                if value is not None:
                    np.savez_compressed(cache, **value)
            selected["descriptor_refusal"] = why
            obs.append(selected)
            desc.append(value)
            print(
                frame["alias"],
                index,
                "class",
                selected["class_id"],
                "patches",
                None if value is None else len(value["vectors"]),
                why,
                flush=True,
            )
        objects.append(obs)
        descriptors.append(desc)
    pairs = []
    for i, oa in enumerate(objects[0]):
        for j, ob in enumerate(objects[1]):
            measured = compare(descriptors[0][i], descriptors[1][j], oa, ob)
            pairs.append({"source_index": i, "target_index": j, **measured})
    eligible = [p for p in pairs if p["eligible"]]
    # Score must uniquely prefer the same counterpart at the object level too.
    admitted = []
    for pair in eligible:
        competitors = [
            p["score"]
            for p in eligible
            if p is not pair
            and (
                p["source_index"] == pair["source_index"]
                or p["target_index"] == pair["target_index"]
            )
        ]
        if (not competitors or pair["score"] >= max(competitors) * 1.5) and _iou(
            objects[0][pair["source_index"]]["box"],
            objects[1][pair["target_index"]]["box"],
        ) >= 0.5:
            admitted.append(pair)
    result = {
        "group": "G018",
        "frames": ["F01", "F02"],
        "objects": objects,
        "pairs": pairs,
        "review_tracklet_candidates": admitted,
        "candidate_count": len(admitted),
        "class_or_human_identity_authority": False,
        "runtime_admitted": False,
        "full_four_frame_track_confirmed": False,
        "native_no_resize": True,
        "refusals": dict(Counter(p["reason"] for p in pairs if not p["eligible"])),
    }
    save(a.output / "result.json", result)
    summary = {
        k: v
        for k, v in result.items()
        if k not in ["objects", "pairs", "review_tracklet_candidates"]
    }
    save(a.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
