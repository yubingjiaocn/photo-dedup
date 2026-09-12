"""Resumable multi-view native foreground correspondence on exposed groups.

Originals, old feature caches and frozen v0 are read-only. Search priors can come
from other frames, but every accepted observation needs its own two detector
views. No labels or semantic actor names are runtime inputs.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.config import Config, load_config  # noqa: E402
from src.stage1_backends import TorchBackend  # noqa: E402
from src.local_quality import _iou  # noqa: E402
from src.instance_recovery import novel_detection  # noqa: E402
from src.instance_correspondence import (  # noqa: E402
    view_agreement,
    masked_crop,
    foreground_keypoints,
    foreground_motion,
    match_appearance,
    track_components,
)


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def choose(seed, observations):
    scores = sorted(
        [
            (_iou(seed["box"], o["box"]), i)
            for i, o in enumerate(observations)
            if o["class_id"] in (0, 77)
            and min(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) >= 96
        ],
        reverse=True,
    )
    if not scores or scores[0][0] < 0.5:
        return None, "NO_INDEPENDENT_TARGET"
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.15:
        return None, "TARGET_VIEW_AMBIGUOUS"
    return observations[scores[0][1]], None


def detect_view(detector, image, window, path):
    if path.exists():
        return json.loads(path.read_text())["observations"]
    crop = image.crop(window)
    prediction = detector.predict(
        crop,
        imgsz=960,
        conf=0.6,
        agnostic_nms=True,
        max_det=64,
        device=0,
        save=False,
        verbose=False,
    )[0]
    observations = []
    if prediction.masks is not None:
        for box, cls, confidence, polygon in zip(
            prediction.boxes.xyxy.cpu().tolist(),
            prediction.boxes.cls.cpu().tolist(),
            prediction.boxes.conf.cpu().tolist(),
            prediction.masks.xy,
        ):
            edge = min(box[0], box[1], crop.width - box[2], crop.height - box[3])
            observations.append(
                {
                    "box": [
                        box[0] + window[0],
                        box[1] + window[1],
                        box[2] + window[0],
                        box[3] + window[1],
                    ],
                    "class_id": int(cls),
                    "confidence": float(confidence),
                    "polygon": (polygon + np.array(window[:2])).tolist(),
                    "crop_edge": edge < max(8, 0.01 * min(crop.size)),
                    "edge_distance_native_px": edge,
                    "window": window,
                }
            )
    save(path, {"window": window, "observations": observations})
    return observations


def observe_frame(frame, image, seeds, prior, detector, out):
    path = out / "observations.json"
    if path.exists():
        return json.loads(path.read_text())
    w, h = image.size
    observations = []
    attempts = []
    potential = []
    for i, seed in enumerate(seeds):
        box = seed["box"]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        views = []
        reason = None
        for j, (mx, my) in enumerate([(0.35, 0.25), (0.7, 0.5)]):
            window = [
                max(0, int(box[0] - bw * mx)),
                max(0, int(box[1] - bh * my)),
                min(w, int(box[2] + bw * mx)),
                min(h, int(box[3] + bh * my)),
            ]
            detections = detect_view(
                detector, image, window, out / f"view-{i:02d}-{j}.json"
            )
            potential.extend(
                o
                for o in detections
                if o["class_id"] in (0, 77)
                and min(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) >= 96
            )
            selected, reason = choose(seed, detections)
            if selected is None:
                break
            views.append(selected)
        evidence = (
            {"eligible": False, "reason": reason}
            if reason
            else view_agreement(*views, (h, w))
        )
        attempts.append(
            {
                "seed_index": i,
                "source_frame": seed["source_frame"],
                "seed_box": box,
                "views": views,
                "agreement": evidence,
            }
        )
        if not evidence["eligible"]:
            continue
        # Same hypothesized object in two views: use the broader confirmed extent,
        # never choose the largest actor in a scene.
        selected = max(
            views,
            key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]),
        )
        selected = {
            **selected,
            "view_confirmed": True,
            "view_evidence": evidence,
            "seed_index": i,
            "view_classes": [v["class_id"] for v in views],
            "new_vs_v0_catalog": novel_detection(selected, prior),
        }
        if novel_detection(selected, observations):
            observations.append(selected)
    # Unconfirmed detector peers compete for identity too. Removing them before
    # the reciprocal appearance margin would manufacture uniqueness.
    for obj in sorted(potential, key=lambda o: o["confidence"], reverse=True):
        if novel_detection(obj, observations):
            observations.append(
                {
                    **obj,
                    "view_confirmed": False,
                    "new_vs_v0_catalog": novel_detection(obj, prior),
                }
            )
    result = {
        "frame": frame,
        "shape": [w, h],
        "observations": observations,
        "attempts": attempts,
        "refusals": dict(
            Counter(
                x["agreement"]["reason"]
                for x in attempts
                if not x["agreement"]["eligible"]
            )
        ),
    }
    save(path, result)
    return result


def group_run(group, root, out, detector, backend):
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    c = sqlite3.connect(f"file:{group['source_db']}?mode=ro&immutable=1", uri=True)
    images = []
    seeds = []
    prior = json.loads(
        (
            root / f"candidate/{group['date']}-{group['group_alias']}/result.json"
        ).read_text()
    )
    for frame in group["frames"]:
        row = c.execute(
            "SELECT path,size_bytes,mtime_ns FROM files WHERE id=?", (frame["id"],)
        ).fetchone()
        stat = Path(row[0]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row[1], row[2]):
            raise ValueError("Native image drift")
        with Image.open(row[0]) as raw:
            images.append(ImageOps.exif_transpose(raw).convert("RGB"))
        packet = json.loads(
            (
                root
                / f"next-class-scale/{group['date']}-{group['group_alias']}-{frame['alias']}.json"
            ).read_text()
        )
        for o in packet["observations"]:
            if o["class_id"] in (0, 77) and not any(
                _iou(o["box"], s["box"]) >= 0.8 for s in seeds
            ):
                seeds.append({**o, "source_frame": frame["alias"]})
    c.close()
    if len({im.size for im in images}) != 1:
        raise ValueError("Cross-frame search priors require identical frame geometry")
    if len(seeds) > 40:
        raise ValueError("Bounded search seed capacity exceeded")
    records = []
    vectors = []
    points = []
    for index, (frame, image) in enumerate(zip(group["frames"], images)):
        frame_out = out / frame["alias"]
        frame_out.mkdir(exist_ok=True)
        record = observe_frame(
            frame, image, seeds, prior["catalogs"][index], detector, frame_out
        )
        observations = record["observations"]
        vector_path = frame_out / "foreground-vectors.npy"
        if vector_path.exists():
            v = np.load(vector_path)
            if v.shape != (len(observations), 768):
                raise ValueError("Cached foreground vector count mismatch")
        else:
            values = []
            for obj in observations:
                prepared = backend.prepare_cpu(masked_crop(image, obj))
                value = backend.embed_prepared([prepared])[0].astype(np.float32)
                value /= np.linalg.norm(value)
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite foreground embedding")
                values.append(value)
            v = np.asarray(values, dtype=np.float32).reshape(-1, 768)
            np.save(vector_path, v)
        gray = np.asarray(image.convert("L"))
        points.append([foreground_keypoints(gray, obj) for obj in observations])
        vectors.append(v)
        records.append(record)
    links = []
    refusals = Counter()
    edges = []
    for s in range(len(records)):
        for t in range(s + 1, len(records)):
            left, right = records[s]["observations"], records[t]["observations"]
            pairs, rejected = match_appearance(left, right, vectors[s], vectors[t])
            for r in rejected:
                refusals[r["reason"]] += 1
                links.append(
                    {"source_frame": s, "target_frame": t, **r, "eligible": False}
                )
            for pair in pairs:
                i, j = pair["source_index"], pair["target_index"]
                support = foreground_motion(
                    points[s][i], points[t][j], left[i], right[j]
                )
                links.append(
                    {
                        "source_frame": s,
                        "target_frame": t,
                        **pair,
                        "support": support,
                        "eligible": support["eligible"],
                    }
                )
                if support["eligible"]:
                    edges.extend([((s, i), (t, j)), ((t, j), (s, i))])
                else:
                    refusals[support["reason"]] += 1
    tracks = track_components([len(r["observations"]) for r in records], edges)
    for track in tracks:
        track["new_observations"] = [
            {"frame": f, "index": i, "file_id": group["frames"][f]["id"]}
            for f, i in enumerate(track["slots"])
            if i is not None and records[f]["observations"][i]["new_vs_v0_catalog"]
        ]
    proposed = [t for t in tracks if t["new_observations"]]
    recovered = {
        (x["frame"], x["index"]) for t in proposed for x in t["new_observations"]
    }
    for f, (image, record) in enumerate(zip(images, records)):
        panel = image.copy()
        panel.thumbnail((1000, 800))
        draw = ImageDraw.Draw(panel)
        sx, sy = panel.width / image.width, panel.height / image.height
        for i, obj in enumerate(record["observations"]):
            color = "#00ff80" if (f, i) in recovered else "#ffc040"
            polygon = [(x * sx, y * sy) for x, y in obj["polygon"]]
            if polygon:
                draw.line(polygon + [polygon[0]], fill=color, width=2)
            box = [v * (sx if k % 2 == 0 else sy) for k, v in enumerate(obj["box"])]
            draw.rectangle(box, outline=color, width=2)
            draw.text(
                (box[0], box[1]),
                f"{i} cls{obj['class_id']}",
                fill=color,
                stroke_width=1,
                stroke_fill="black",
            )
        panel.save(out / f"{group['frames'][f]['alias']}.jpg", quality=90)
    result = {
        "day": group["date"],
        "group_alias": group["group_alias"],
        "group": group,
        "method": "independent_views_masked_dino_sift",
        "records": records,
        "links": links,
        "tracks": tracks,
        "proposed_tracks": proposed,
        "new_frame_observations": len(recovered),
        "refusals": dict(refusals),
        "complete_semantic_subject_set": "unknown",
        "keeper_authority": False,
        "runtime_admitted": False,
    }
    save(result_path, result)
    print(
        group["date"],
        group["group_alias"],
        "observations",
        [len(r["observations"]) for r in records],
        "tracks",
        len(tracks),
        "new tracks",
        len(proposed),
        "new frames",
        len(recovered),
        dict(refusals),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=["G012", "G018"])
    a = p.parse_args()
    if not a.model.is_file():
        raise ValueError("Existing local model required")
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YOLO_CONFIG_DIR=str(a.output / "ultralytics-settings"),
    )
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    groups = [
        g
        for g in json.loads((a.root / "slice.json").read_text())["groups"]
        if g["group_alias"] in a.groups
    ]
    if len(groups) != len(set(a.groups)):
        raise ValueError(
            "Every requested group must match the frozen slice exactly once"
        )
    detector = YOLO(str(a.model))
    cfg = load_config(groups[0]["config"]).as_dict()
    cfg["features"].update(iqa_musiq=False, iqa_clipiqa=False)
    backend = TorchBackend(Config(cfg))
    started = time.perf_counter()
    results = [
        group_run(
            g, a.root, a.output / f"{g['date']}-{g['group_alias']}", detector, backend
        )
        for g in groups
    ]
    summary = {
        "groups": len(results),
        "frames": sum(len(g["group"]["frames"]) for g in results),
        "proposed_groups": sum(bool(g["proposed_tracks"]) for g in results),
        "proposed_tracks": sum(len(g["proposed_tracks"]) for g in results),
        "new_frame_observations": sum(g["new_frame_observations"] for g in results),
        "complete_semantic_subject_sets_confirmed": 0,
        "keeper_changed_groups": 0,
        "seconds": time.perf_counter() - started,
        "runtime_admitted": False,
        "refusals": dict(sum((Counter(g["refusals"]) for g in results), Counter())),
    }
    save(a.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
