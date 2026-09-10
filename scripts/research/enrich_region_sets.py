"""Bounded all-instance local observations; no regime/label inputs."""

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import db  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.stage1_backends import TorchBackend  # noqa: E402


def enrich(source, output, config, model_path):
    if output.exists():
        raise ValueError("Existing output; inspect, never overwrite")
    if not source.is_file() or not model_path.is_file():
        raise ValueError("Existing input/model required")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output.parent / "ultralytics-settings")
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    detector = YOLO(str(model_path))
    cfg = load_config(str(config)).as_dict()
    cfg["features"]["iqa_max_long_edge"] = 960
    backend = TorchBackend(Config(cfg))
    c = sqlite3.connect(f"file:{source.resolve()}?mode=ro&immutable=1", uri=True)
    t = sqlite3.connect(output)
    c.backup(t)
    t.close()
    c.close()
    c = db.open_db(output)
    rows = c.execute(
        "SELECT DISTINCT f.*,fe.quality_meta,fe.faces_json FROM files f JOIN features fe ON f.id=fe.file_id JOIN group_members gm ON f.id=gm.file_id JOIN groups g ON gm.group_id=g.id WHERE g.group_type!='sha_exact' ORDER BY f.id"
    ).fetchall()
    counts = Counter()
    start = time.perf_counter()
    for row in rows:
        p = Path(row["path"])
        stat = p.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
            raise ValueError("Native input drift")
        with Image.open(p) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            w, h = image.size
            pred = detector.predict(
                image,
                imgsz=960,
                classes=[0, 15, 16],
                conf=0.6,
                max_det=64,
                device=0,
                save=False,
                verbose=False,
            )[0]
            objects = [
                {"box": box, "class_id": int(cls), "confidence": conf}
                for box, cls, conf in zip(
                    pred.boxes.xyxy.cpu().tolist(),
                    pred.boxes.cls.cpu().tolist(),
                    pred.boxes.conf.cpu().tolist(),
                )
            ]
            objects.sort(key=lambda o: (o["class_id"], o["box"][0], o["box"][1]))
            status = "complete"
            reason = None
            if not objects:
                status = "no_supported_regions"
                reason = "NO_RELIABLE_REGION_SET"
            elif len(objects) > 12:
                status = "incomplete"
                reason = "REGION_SET_OVER_CAPACITY"
            elif any(
                min(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) < 96
                for o in objects
            ):
                status = "incomplete"
                reason = "SMALL_MEMBER_UNASSESSABLE"
            faces = {i: [] for i in range(len(objects))}
            if status == "complete":
                for face in json.loads(row["faces_json"] or "[]"):
                    x, y, fw, fh = face["bbox"]
                    if face.get("score", 0) < 0.9 or min(fw, fh) < 96:
                        continue
                    parents = [
                        i
                        for i, o in enumerate(objects)
                        if o["class_id"] == 0
                        and o["box"][0] <= x + fw / 2 <= o["box"][2]
                        and o["box"][1] <= y + fh / 2 <= o["box"][3]
                    ]
                    if len(parents) != 1:
                        status = "incomplete"
                        reason = "FACE_PARENT_AMBIGUOUS"
                        break
                    faces[parents[0]].append(
                        (
                            face,
                            [
                                max(0, x - fw * 0.15),
                                max(0, y - fh * 0.15),
                                min(w, x + fw * 1.15),
                                min(h, y + fh * 1.15),
                            ],
                        )
                    )
                if any(len(v) > 1 for v in faces.values()):
                    status = "incomplete"
                    reason = "MULTIPLE_FACES_IN_INSTANCE"
            regions = []
            blob = bytearray()
            if status == "complete":
                for index, obj in enumerate(objects):
                    parent = f"S{index}"
                    rois = [("subject", parent, None, obj["box"], obj["confidence"])]
                    for face, box in faces[index]:
                        rois.append(("face", f"F{index}", parent, box, face["score"]))
                    for kind, rid, parent_id, box, confidence in rois:
                        x1, y1, x2, y2 = box
                        crop = image.crop(tuple(int(v) for v in box))
                        prepared = backend.prepare_cpu(crop)
                        _, q = backend.quality_prepared([prepared])[0]
                        v = backend.embed_prepared([prepared])[0]
                        regions.append(
                            {
                                "region_id": rid,
                                "parent_id": parent_id,
                                "kind": kind,
                                "class_id": obj["class_id"],
                                "confidence": confidence,
                                "box_normalized": [x1 / w, y1 / h, x2 / w, y2 / h],
                                "native_size": list(crop.size),
                                "musiq": q["musiq"],
                                "clipiqa": q.get("clipiqa"),
                                "embedding_offset": len(blob),
                            }
                        )
                        blob.extend(v.astype("<f2").tobytes())
        meta = json.loads(row["quality_meta"] or "{}")
        meta.pop("local_quality", None)
        meta["local_region_set"] = {
            "schema_version": 2,
            "status": status,
            "reason": reason,
            "subject_count": len(objects),
            "regions": regions,
            "catalog": objects,
            "catalog_semantics": "all supported detections, not intended-subject truth",
            "eye_state": "unknown",
            "amodal_completeness": "unknown",
            "regime_label_input": False,
        }
        c.execute(
            "UPDATE features SET quality_meta=?,local_quality_embedding=? WHERE file_id=?",
            (json.dumps(meta), bytes(blob) if blob else None, row["id"]),
        )
        counts.update(images=1, regions=len(regions))
        counts[status] += 1
        if status == "complete":
            counts["multi_sets" if len(objects) > 1 else "single_sets"] += 1
        if counts["images"] % 20 == 0:
            c.commit()
            print(dict(counts), flush=True)
    c.commit()
    c.close()
    summary = {
        **counts,
        "seconds_excluding_setup": time.perf_counter() - start,
        "source": str(source),
        "output": str(output),
        "original_changed": False,
        "label_input": False,
        "model_installed": str(model_path),
    }
    output.with_suffix(".region-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ["db", "output-db", "config", "model"]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    enrich(a.db, a.output_db, a.config, a.model)
