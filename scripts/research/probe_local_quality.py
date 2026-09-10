"""Local-only thin slice: actual object/face crops, not semantic eye labels."""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
OLD = Path('/home/ubuntu/photo-dedup-eval/astra-real-events-20260910')
OUT = Path('/home/ubuntu/photo-dedup-eval/astra-local-quality-20260910/thin-slice')
sys.path.insert(0, str(REPO))
from src.config import Config, load_config  # noqa: E402
from src.stage1_backends import TorchBackend  # noqa: E402


def run():
    if (OUT / 'observations.json').exists():
        raise ValueError('Completed observations exist; inspect instead of overwriting')
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'ultralytics-settings')
    from ultralytics import YOLO
    detector_path = OLD.parent / 'astra-usable-algorithm-20260909/person-instance-probe/models/yolo26s-seg.pt'
    assert detector_path.is_file()
    detector = YOLO(str(detector_path))
    cfg_data = load_config(str(OLD / 'cache/2026-04-04/config.yaml')).as_dict()
    cfg_data['features']['iqa_max_long_edge'] = 960
    backend = TorchBackend(Config(cfg_data))
    selected = {'2026-04-05': {'G018', 'G015'}, '2026-04-04': {'G008', 'G051', 'G057', 'G062'}}
    # Add all default-baseline quality failures. Selection is development only.
    for day in selected:
        ab = json.loads((OLD / ('HOLDOUT-APR5-AB.json' if day.endswith('05') else 'HOLDOUT-APR4-AB.json')).read_text())
        selected[day].update(g['group_alias'] for g in ab['rows'] if g['variants']['baseline']['metrics']['quality_wrong'])
    observations, vectors = [], {}
    start = time.perf_counter()
    for day, aliases in selected.items():
        groups = json.loads((OLD / f'panels/{day}/mapping.json').read_text())
        conn = sqlite3.connect(f'file:{OLD / f"cache/{day}/inventory.sqlite"}?mode=ro&immutable=1', uri=True)
        conn.row_factory = sqlite3.Row
        for group in groups:
            if group['group_alias'] not in aliases:
                continue
            for frame in group['frames']:
                row = conn.execute('SELECT f.path,fe.faces_json,fe.quality_meta FROM files f JOIN features fe ON fe.file_id=f.id WHERE f.id=?', (frame['id'],)).fetchone()
                key = f'{day}/{group["group_alias"]}/{frame["alias"]}'
                record = {'key': key, 'day': day, 'group_alias': group['group_alias'], 'frame_alias': frame['alias'],
                          'file_id': frame['id'], 'regions': [], 'status': 'abstain'}
                with Image.open(row['path']) as raw:
                    image = ImageOps.exif_transpose(raw).convert('RGB')
                    w, h = image.size
                    pred = detector.predict(image, imgsz=960, classes=[0, 15, 16], conf=0.6,
                                            device=0, save=False, verbose=False)[0]
                    objects = []
                    for index, (box, cls, confidence) in enumerate(zip(pred.boxes.xyxy.cpu().tolist(), pred.boxes.cls.cpu().tolist(), pred.boxes.conf.cpu().tolist())):
                        x1, y1, x2, y2 = box
                        area = (x2-x1)*(y2-y1)/(w*h)
                        if area >= 0.05:
                            objects.append({'box': box, 'class_id': int(cls), 'confidence': confidence, 'area': area, 'index': index})
                    objects.sort(key=lambda o: o['area'], reverse=True)
                    record['detections'] = [{k:v for k,v in o.items() if k != 'index'} for o in objects]
                    if not objects or (len(objects)>1 and objects[0]['area'] < 2*objects[1]['area']):
                        record['reason'] = 'NO_UNAMBIGUOUS_DOMINANT_DETECTION'
                    else:
                        obj = objects[0]
                        rois = [('subject', obj['box'])]
                        if obj['class_id'] == 0 and not any(o['class_id'] in [15,16] for o in objects):
                            faces = json.loads(row['faces_json'] or '[]')
                            eligible = []
                            for face in faces:
                                x,y,fw,fh = face['bbox']
                                x1,y1,x2,y2 = obj['box']
                                if face.get('score',0)>=0.9 and min(fw,fh)>=96 and x1<=x+fw/2<=x2 and y1<=y+fh/2<=y2:
                                    eligible.append([max(0,x-fw*.15),max(0,y-fh*.15),min(w,x+fw*1.15),min(h,y+fh*1.15)])
                            if len(eligible)==1:
                                rois.append(('face',eligible[0]))
                        for name, box in rois:
                            x1,y1,x2,y2 = box
                            if min(x2-x1,y2-y1)<96:
                                continue
                            crop = image.crop((int(x1),int(y1),int(x2),int(y2)))
                            prepared = backend.prepare_cpu(crop)
                            _, quality = backend.quality_prepared([prepared])[0]
                            vector = backend.embed_prepared([prepared])[0]
                            vector_key = key.replace('/','_')+'_'+name
                            vectors[vector_key] = vector
                            bounded = crop.copy()
                            bounded.thumbnail((512,512),Image.Resampling.LANCZOS)
                            from src.quality import variance_of_laplacian
                            sharpness_512 = variance_of_laplacian(np.asarray(bounded.convert('L'),dtype=np.float32))
                            record['regions'].append({'kind':name,'class_id':obj['class_id'],
                              'box_normalized':[x1/w,y1/h,x2/w,y2/h], 'native_size':list(crop.size),
                              'musiq':quality['musiq'],'clipiqa':quality.get('clipiqa'),
                              'sharpness_512':sharpness_512,'embedding_key':vector_key,
                              'semantics':'local perceptual quality/detail; not eye state or amodal completeness'})
                        record['status'] = 'observed' if record['regions'] else 'abstain'
                observations.append(record)
                (OUT / 'checkpoint.json').write_text(json.dumps({'frames':observations},indent=2))
                print(key,record['status'],[(q['kind'],round(q['musiq'],2),round(q['clipiqa'],3),round(q['sharpness_512'],1)) for q in record['regions']],flush=True)
        conn.close()
    np.savez_compressed(OUT / 'private-local-embeddings.npz',**vectors)
    result = {'status':'development_only_probe','model':str(detector_path),'frames':observations,
              'seconds':time.perf_counter()-start,'native_pixels_local_only':True,'original_modified':False}
    (OUT / 'observations.json').write_text(json.dumps(result,indent=2))
    print('complete frames',len(observations),'seconds',result['seconds'])


if __name__ == '__main__':
    run()
