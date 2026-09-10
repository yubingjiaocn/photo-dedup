"""Bounded upstream class/scale diagnosis with the already-installed detector.

Not a runtime candidate: no keeper, identity, phase or semantic-subject authority.
Uses the frozen exposed slice only; never reads unseen labels or changes v0.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from PIL import Image,ImageDraw,ImageOps

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.instance_recovery import novel_detection


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--slice',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.model.is_file():
        raise ValueError('Existing local model required')
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(a.output/'ultralytics-settings')
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    model=YOLO(str(a.model))
    groups=json.loads(a.slice.read_text())['groups']
    rows=[]
    for group in groups:
        c=sqlite3.connect(f'file:{group["source_db"]}?mode=ro&immutable=1',uri=True)
        for frame in group['frames']:
            stem=f'{group["date"]}-{group["group_alias"]}-{frame["alias"]}'
            result_path=a.output/f'{stem}.json'
            if result_path.exists():
                rows.append(json.loads(result_path.read_text()))
                continue
            row=c.execute('SELECT f.path,f.size_bytes,f.mtime_ns,fe.quality_meta FROM files f JOIN features fe ON f.id=fe.file_id WHERE f.id=?',(frame['id'],)).fetchone()
            path=Path(row[0]); stat=path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(row[1],row[2]):
                raise ValueError('Native photo drift')
            original=json.loads(row[3])['local_region_set']['catalog']
            with Image.open(path) as raw:
                image=ImageOps.exif_transpose(raw).convert('RGB')
            w,h=image.size
            windows=[[0,0,w,h]]
            for y1,y2 in [(0,int(h*.6)),(int(h*.4),h)]:
                for x1,x2 in [(0,int(w*.6)),(int(w*.4),w)]:
                    windows.append([x1,y1,x2,y2])
            observations=[]
            started=time.perf_counter()
            for index,window in enumerate(windows):
                crop=image.crop(window)
                prediction=model.predict(crop,imgsz=960,conf=.6,agnostic_nms=True,
                                         max_det=64,device=0,save=False,verbose=False)[0]
                for box,cls,confidence in zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.cls.cpu().tolist(),prediction.boxes.conf.cpu().tolist()):
                    if index and (box[0]<=2 or box[1]<=2 or box[2]>=crop.width-2 or box[3]>=crop.height-2):
                        continue
                    b=[box[0]+window[0],box[1]+window[1],box[2]+window[0],box[3]+window[1]]
                    if min(b[2]-b[0],b[3]-b[1])<96:
                        continue
                    candidate={'box':b,'class_id':int(cls),'class_name':model.names[int(cls)],
                               'confidence':float(confidence),'window':index,
                               'outside_v0_classes':int(cls) not in [0,15,16]}
                    if novel_detection(candidate,observations):
                        observations.append(candidate)
            novel=[o for o in observations if novel_detection(o,original)]
            result={'day':group['date'],'group_alias':group['group_alias'],'frame_alias':frame['alias'],
                    'file_id':frame['id'],'original_count':len(original),'observations':observations,
                    'novel_box_candidates':novel,'identity_authority':False,'keeper_authority':False,
                    'semantic_subject_authority':False,'seconds':time.perf_counter()-started}
            panel=image.copy();panel.thumbnail((1000,800));draw=ImageDraw.Draw(panel)
            sx,sy=panel.width/w,panel.height/h
            for index,o in enumerate(novel):
                box=[v*(sx if k%2==0 else sy) for k,v in enumerate(o['box'])]
                color='#ffc040' if o['outside_v0_classes'] else '#80c0ff'
                draw.rectangle(box,outline=color,width=3)
                draw.text((box[0],box[1]),f'{index} {o["class_name"]} {o["confidence"]:.2f}',fill=color,stroke_width=1,stroke_fill='black')
            panel.save(a.output/f'{stem}.jpg',quality=90)
            result_path.write_text(json.dumps(result,indent=2))
            rows.append(result)
            print(stem,'novel boxes',Counter(o['class_name'] for o in novel),flush=True)
        c.close()
    summary={'status':'bounded_exposed_class_scale_probe_complete','frames':len(rows),'groups':len(groups),
             'novel_box_classes':dict(Counter(o['class_name'] for r in rows for o in r['novel_box_candidates'])),
             'outside_v0_classes':sum(o['outside_v0_classes'] for r in rows for o in r['novel_box_candidates']),
             'identity_or_semantic_truth':False,'v0_modified':False,'model_installed':str(a.model),
             'next_step':'Inspect whether novel boxes cover missed stage characters, not unrelated scene objects; no runtime expansion without confirmation.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
