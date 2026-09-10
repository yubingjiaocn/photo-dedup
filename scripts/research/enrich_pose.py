"""Add optional local pose observations to a NEW feature database copy.

Loads an explicit already-installed YOLO pose model. No implicit downloads,
photo changes, external model requests, semantic labels, or deletion execution.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from PIL import Image, ImageOps


def enrich(source: Path, output: Path, model_path: Path):
    if not model_path.is_file() or not source.is_file():
        raise ValueError('Explicit existing source DB and local model are required')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing feature database')
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(output.parent / 'ultralytics-settings')
    from ultralytics import YOLO
    started = time.perf_counter()
    model = YOLO(str(model_path.resolve()))
    source_conn = sqlite3.connect(f'file:{source.resolve()}?mode=ro&immutable=1', uri=True)
    conn = sqlite3.connect(output)
    source_conn.backup(conn)
    source_conn.close()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT DISTINCT f.id,f.path,fe.quality_meta FROM files f JOIN features fe ON f.id=fe.file_id '
        'JOIN group_members gm ON gm.file_id=f.id JOIN groups g ON g.id=gm.group_id '
        'WHERE g.group_type != ? ORDER BY f.id', ('sha_exact',)).fetchall()
    counts = {'images': 0, 'images_with_people': 0, 'detected_people': 0}
    for row in rows:
        with Image.open(row['path']) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
            result = model.predict(image, imgsz=960, conf=0.25, max_det=30,
                                   device=0, save=False, verbose=False)[0]
            record = {'schema_version': 1, 'status': 'ok', 'producer': model_path.name,
                      'pose_format': 'coco17', 'shape': list(image.size), 'people': []}
            if result.keypoints is not None:
                coordinates = result.keypoints.xyn.cpu().tolist()
                confidence = result.keypoints.conf.cpu().tolist()
                boxes = result.boxes.xyxyn.cpu().tolist()
                box_conf = result.boxes.conf.cpu().tolist()
                for xy, scores, box, score in zip(coordinates, confidence, boxes, box_conf):
                    record['people'].append({'box': box, 'box_confidence': score,
                                             'keypoints': [[x, y, c] for (x, y), c in zip(xy, scores)]})
        meta = json.loads(row['quality_meta'] or '{}')
        meta['pose_evidence'] = record
        conn.execute('UPDATE features SET quality_meta=? WHERE file_id=?',
                     (json.dumps(meta), row['id']))
        counts['images'] += 1
        counts['images_with_people'] += bool(record['people'])
        counts['detected_people'] += len(record['people'])
        if counts['images'] % 25 == 0:
            conn.commit()
            print('pose images', counts['images'], flush=True)
    conn.commit()
    conn.close()
    report = {'source': str(source), 'output': str(output), 'model': str(model_path),
              'seconds': time.perf_counter() - started, **counts,
              'native_pixels_local_only': True, 'source_database_modified': False,
              'photo_modified': False, 'label_input': False, 'cloud_write': False}
    output.with_suffix('.pose-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--output-db', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    enrich(args.db, args.output_db, args.model)
