"""Native local-quality observations in a NEW feature DB; no label input."""
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import db  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.stage1_backends import TorchBackend  # noqa: E402


def enrich(source, output, config, model_path):
    if output.exists():
        raise ValueError('Refusing to overwrite existing local-quality DB')
    if not source.is_file() or not model_path.is_file():
        raise ValueError('Existing source DB and installed model are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(output.parent / 'ultralytics-settings')
    from ultralytics import YOLO, settings as yolo_settings
    yolo_settings.update({'sync': False})
    detector = YOLO(str(model_path))
    settings = load_config(str(config)).as_dict()
    settings['features']['iqa_max_long_edge'] = 960
    backend = TorchBackend(Config(settings))
    original = sqlite3.connect(f'file:{source.resolve()}?mode=ro&immutable=1', uri=True)
    destination = sqlite3.connect(output)
    original.backup(destination)
    original.close()
    destination.close()
    conn = db.open_db(output)
    rows = conn.execute("SELECT DISTINCT f.*,fe.quality_meta,fe.faces_json FROM files f JOIN features fe ON fe.file_id=f.id JOIN group_members gm ON gm.file_id=f.id JOIN groups g ON g.id=gm.group_id WHERE g.group_type!='sha_exact' ORDER BY f.id").fetchall()
    counts = Counter(images=0, regions=0, abstained_images=0)
    start = time.perf_counter()
    for row in rows:
        file = Path(row['path'])
        stat = file.stat()
        if stat.st_size != row['size_bytes'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError('Source photo changed relative to frozen feature cache')
        regions, blob = [], bytearray()
        with Image.open(file) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
            w, h = image.size
            pred = detector.predict(image, imgsz=960, classes=[0,1,2,3,15,16], conf=0.6,
                                    device=0, save=False, verbose=False)[0]
            objects = []
            for box, cls, confidence in zip(pred.boxes.xyxy.cpu().tolist(), pred.boxes.cls.cpu().tolist(), pred.boxes.conf.cpu().tolist()):
                x1,y1,x2,y2 = box
                if (x2-x1)*(y2-y1)/(w*h) >= 0.05 and min(x2-x1,y2-y1)>=96:
                    objects.append({'box':box,'class_id':int(cls),'confidence':confidence})
            frequencies = Counter(o['class_id'] for o in objects)
            for obj in objects:
                if frequencies[obj['class_id']] != 1:
                    continue
                rois = [('subject',obj['box'],obj['confidence'])]
                if obj['class_id']==0 and not any(o['class_id'] in [15,16] for o in objects):
                    eligible = []
                    for face in json.loads(row['faces_json'] or '[]'):
                        x,y,fw,fh = face['bbox']
                        x1,y1,x2,y2 = obj['box']
                        if face.get('score',0)>=0.9 and min(fw,fh)>=96 and x1<=x+fw/2<=x2 and y1<=y+fh/2<=y2:
                            eligible.append(([max(0,x-fw*.15),max(0,y-fh*.15),min(w,x+fw*1.15),min(h,y+fh*1.15)],face['score']))
                    if len(eligible)==1:
                        rois.append(('face',*eligible[0]))
                for kind, box, confidence in rois:
                    x1,y1,x2,y2 = box
                    crop = image.crop((int(x1),int(y1),int(x2),int(y2)))
                    prepared = backend.prepare_cpu(crop)
                    _, quality = backend.quality_prepared([prepared])[0]
                    vector = backend.embed_prepared([prepared])[0]
                    if min(crop.size)<96 or not np.all(np.isfinite(vector)):
                        continue
                    regions.append({'kind':kind,'class_id':obj['class_id'],'confidence':confidence,
                                    'box_normalized':[x1/w,y1/h,x2/w,y2/h], 'native_size':list(crop.size),
                                    'musiq':quality['musiq'],'clipiqa':quality.get('clipiqa'),
                                    'embedding_offset':len(blob)})
                    blob.extend(vector.astype('<f2').tobytes())
        meta = json.loads(row['quality_meta'] or '{}')
        meta['local_quality'] = {'schema_version':1,'method':'native_coco_crop_v1',
                                 'producer':model_path.name,'appearance_model':'dinov2-base',
                                 'regions':regions,'status':'observed' if regions else 'abstain',
                                 'catalog_complete_single':len(pred.boxes)==1 and len(objects)==1,
                                 'detected_subject_count':len(pred.boxes),
                                 'eye_state':'not_measured','amodal_completeness':'not_measured'}
        conn.execute('UPDATE features SET quality_meta=?,local_quality_embedding=? WHERE file_id=?',
                     (json.dumps(meta),bytes(blob) if blob else None,row['id']))
        counts.update(images=1,regions=len(regions),abstained_images=int(not regions))
        if counts['images']%20==0:
            conn.commit()
            print(dict(counts),flush=True)
    conn.commit()
    conn.close()
    summary = {'source':str(source),'output':str(output),'model':str(model_path),
               'seconds':time.perf_counter()-start,**counts,'native_pixels_local_only':True,
               'embeddings_binary_db_only':True,'label_input':False,'photo_modified':False}
    output.with_suffix('.local-quality-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--output-db',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    a=p.parse_args()
    enrich(a.db,a.output_db,a.config,a.model)
