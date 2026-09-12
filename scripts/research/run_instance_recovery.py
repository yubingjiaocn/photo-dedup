"""Offline native-pixel proposals, explicit local model, resumable by group.

No labels, network acquisition, source DB writes or keeper changes. Outputs are
review packets, never replacements for the original region observation catalog.
"""
from __future__ import annotations
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
from src.instance_recovery import (PRODUCER, track_box, sift_box, confirm_target,  # noqa: E402
                                   appearance_unique, novel_detection, pose_pair_support, stable_tracks)
from src.local_quality import _iou  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.stage1_backends import TorchBackend  # noqa: E402


def read_group(group):
    conn = sqlite3.connect(f"file:{group['source_db']}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    rows, images, catalogs = [], [], []
    for frame in group['frames']:
        row = dict(conn.execute('SELECT f.*,fe.quality_meta FROM files f JOIN features fe ON f.id=fe.file_id WHERE f.id=?', (frame['id'],)).fetchone())
        p = Path(row['path'])
        stat = p.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['size_bytes'], row['mtime_ns']):
            raise ValueError('Native input drift')
        with Image.open(p) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        catalog = json.loads(row['quality_meta']).get('local_region_set', {}).get('catalog', [])
        rows.append(row)
        images.append(image)
        catalogs.append([dict(o) for o in catalog])
    conn.close()
    return rows, images, catalogs


def detect_crop(detector, image, box, cls):
    w,h = image.size
    x1,y1,x2,y2 = box
    bw,bh = x2-x1,y2-y1
    roi = [max(0,int(x1-bw*.35)), max(0,int(y1-bh*.25)),
           min(w,int(x2+bw*.35)), min(h,int(y2+bh*.25))]
    crop = image.crop(roi)
    prediction = detector.predict(crop, imgsz=960, classes=[0,15,16], conf=.6,
                                  agnostic_nms=True, max_det=64, device=0, save=False, verbose=False)[0]
    # A crop-cut fragment cannot confirm the full target. Detect all classes;
    # class filtering must not hide a stronger conflicting dog/person result.
    return [{'box':[b[0]+roi[0],b[1]+roi[1],b[2]+roi[0],b[3]+roi[1]],
             'class_id':int(c), 'confidence':float(s), 'origin':'target_crop_redetection'}
            for b,c,s in zip(prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.cls.cpu().tolist(), prediction.boxes.conf.cpu().tolist())
            if int(c)==cls and b[0]>2 and b[1]>2 and b[2]<crop.width-2 and b[3]<crop.height-2]


def vectors_for(backend, image, objects):
    vectors = []
    for obj in objects:
        crop = image.crop(tuple(int(v) for v in obj['box']))
        prepared = backend.prepare_cpu(crop)
        v = backend.embed_prepared([prepared])[0].astype(np.float32)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n <= 0:
            raise ValueError('Invalid local appearance embedding')
        vectors.append(v/n)
    return np.array(vectors)


def process_group(group, detector, backend, out, method='flow'):
    rows, images, original = read_group(group)
    catalogs = [[dict(o) for o in objects] for objects in original]
    gray = [np.asarray(im.convert('L')) for im in images]
    links, refusals = [], Counter()
    if method == 'pose_anchor':
        for n,(row,image) in enumerate(zip(rows,images)):
            record=json.loads(row['quality_meta']).get('pose_evidence',{})
            if record.get('shape') != list(image.size) or record.get('status') != 'ok':
                refusals['POSE_CACHE_UNAVAILABLE'] += 1
                continue
            w,h=image.size
            for person in record.get('people',[]):
                if person['box_confidence']<.6 or sum(k[2]>=.5 for k in person['keypoints'])<8:
                    continue
                candidate={'box':[v*(w if k%2==0 else h) for k,v in enumerate(person['box'])],
                           'class_id':0,'confidence':person['box_confidence'],
                           'origin':'independent_cached_pose_detection','pose':person}
                overlaps=sorted([(_iou(candidate['box'],obj['box']),i) for i,obj in enumerate(catalogs[n]) if obj['class_id']==0],reverse=True)
                if overlaps and overlaps[0][0]>=.5:
                    if len(overlaps)>1 and overlaps[0][0]-overlaps[1][0]<.1:
                        refusals['POSE_INSTANCE_AMBIGUOUS']+=1
                        continue
                    obj=catalogs[n][overlaps[0][1]]
                    if 'pose' in obj:
                        obj['pose_ambiguous']=True
                    else:
                        obj['pose']=person
                elif novel_detection(candidate,catalogs[n]):
                    catalogs[n].append(candidate)
            for obj in catalogs[n]:
                if obj.pop('pose_ambiguous',False):
                    obj.pop('pose',None)
    if method == 'redetect_sift':
        # Search priors from other frames do not count as detections. Target
        # model evidence is obtained before any correspondence is attempted.
        for target,image in enumerate(images):
            for source,objects in enumerate(original):
                if source == target:
                    continue
                for obj in objects:
                    if min(obj['box'][2]-obj['box'][0],obj['box'][3]-obj['box'][1]) < 96:
                        continue
                    for candidate in detect_crop(detector,image,obj['box'],obj['class_id']):
                        if novel_detection(candidate,catalogs[target]) and min(candidate['box'][2]-candidate['box'][0],candidate['box'][3]-candidate['box'][1]) >= 96:
                            catalogs[target].append(candidate)
        if any(len(c)>24 for c in catalogs):
            raise ValueError('Candidate catalog exceeds bounded capacity; inspect crowd before expansion')
    transport = sift_box if method == 'redetect_sift' else track_box
    seeds = [[dict(o) for o in c] for c in catalogs] if method != 'flow' else original
    for source, objects in enumerate(seeds):
        for target in range(len(images)):
            if source == target:
                continue
            for index, obj in enumerate(objects):
                if method == 'pose_anchor':
                    confirmation=confirm_target(obj['box'],obj['class_id'],catalogs[target])
                    motion=(pose_pair_support(gray[source],gray[target],obj,catalogs[target][confirmation['target_index']])
                            if confirmation['eligible'] else confirmation)
                else:
                    motion = transport(gray[source], gray[target], obj['box'])
                link = {'source':source, 'target':target, 'source_index':index, 'motion':motion}
                if not motion['eligible']:
                    refusals[motion['reason']] += 1
                    links.append(link)
                    continue
                confirmation = confirm_target(motion['box'], obj['class_id'], catalogs[target])
                if method == 'flow' and not confirmation['eligible'] and confirmation['reason'] == 'TARGET_NOT_CONFIRMED':
                    candidates = detect_crop(detector, images[target], motion['box'], obj['class_id'])
                    local = confirm_target(motion['box'], obj['class_id'], candidates)
                    if local['eligible']:
                        candidate = candidates[local['target_index']]
                        # Suppress duplicates, but never replace a detected box.
                        if novel_detection(candidate,catalogs[target]):
                            catalogs[target].append(candidate)
                        confirmation = confirm_target(motion['box'], obj['class_id'], catalogs[target])
                link['confirmation'] = confirmation
                if not confirmation['eligible']:
                    refusals[confirmation['reason']] += 1
                else:
                    link['target_index'] = confirmation['target_index']
                links.append(link)
    vectors = [vectors_for(backend, image, objects) for image,objects in zip(images,catalogs)]
    proposals = {}
    valid_edges = {}
    for link in links:
        if 'target_index' not in link:
            continue
        s,t,i,j = link['source'],link['target'],link['source_index'],link['target_index']
        a = appearance_unique(i,j,catalogs[s],catalogs[t],vectors[s],vectors[t])
        link['appearance'] = a
        if not a['eligible']:
            refusals[a['reason']] += 1
            continue
        valid_edges[s,t,i] = j
        if j >= len(original[t]):
            w,h = images[t].size
            obj = catalogs[t][j]
            key = (t,j)
            proposal = {'status':'confirmed_review_proposal', 'source_file_id':rows[s]['id'],
                        'target_file_id':rows[t]['id'], 'source_frame':group['frames'][s]['alias'],
                        'target_frame':group['frames'][t]['alias'], 'source_index':i, 'target_index':j,
                        'class_id':obj['class_id'], 'box_normalized':[v/(w if k%2==0 else h) for k,v in enumerate(obj['box'])],
                        'target_confidence':obj['confidence'], 'target_confirmation':True,
                        'appearance_unique':True, 'motion_consistent':True, 'appearance':a,
                        'motion':link['motion'], 'keeper_authority':False,
                        'novelty_checked':novel_detection(obj,original[t])}
            if key not in proposals or a['similarity'] > proposals[key]['appearance']['similarity']:
                proposals[key] = proposal
    # A full observed-set claim needs every catalog node, not only the easy
    # persistent subset. Report subset tracks separately and never as coverage.
    tracks = stable_tracks(catalogs,valid_edges)
    stable = bool(tracks) and all(len(c)==len(tracks) for c in catalogs)
    packet = {'schema_version':1, 'producer':PRODUCER, 'review_only':True, 'keeper_authority':False,
              'group_member_ids':sorted(r['id'] for r in rows), 'stable_observed_set':stable,
              'semantic_subject_set_authority':False, 'proposals':list(proposals.values())}
    result = {'day':group['date'], 'group_alias':group['group_alias'], 'group':group,
              'original_counts':[len(c) for c in original], 'candidate_counts':[len(c) for c in catalogs],
              'catalogs':catalogs, 'links':links, 'refusals':dict(refusals), 'packet':packet,
              'stable_tracks':tracks, 'stable_track_count':len(tracks),
              'proposal_count':len(proposals), 'stable_observed_set':stable,
              'changed_keeper_groups':0, 'original_photos_modified':False}
    # Local review panels only, no app/UI modifications and no external media.
    panels = []
    for n,(image,objects) in enumerate(zip(images,catalogs)):
        panel = image.copy()
        panel.thumbnail((1000,800))
        draw = ImageDraw.Draw(panel)
        sx,sy = panel.width/image.width, panel.height/image.height
        for j,obj in enumerate(objects):
            color = '#00ff80' if (n,j) in proposals else ('#ffc040' if j>=len(original[n]) else '#80c0ff')
            box = [v*(sx if k%2==0 else sy) for k,v in enumerate(obj['box'])]
            draw.rectangle(box,outline=color,width=3)
            draw.text((box[0],box[1]),f'{j} {obj["confidence"]:.2f}',fill=color,stroke_width=1,stroke_fill='black')
        filename = f'{group["frames"][n]["alias"]}.jpg'
        panel.save(out/filename, quality=90)
        panels.append(filename)
    result['panels'] = panels
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--method',choices=['flow','redetect_sift','pose_anchor'],default='flow')
    args = parser.parse_args()
    if not args.model.is_file():
        raise ValueError('Explicit preinstalled model file required')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', YOLO_CONFIG_DIR=str(args.output/'ultralytics-settings'))
    from ultralytics import YOLO, settings
    settings.update({'sync':False})
    groups = json.loads(args.slice.read_text())['groups']
    pending = [g for g in groups if not (args.output/f'{g["date"]}-{g["group_alias"]}/result.json').exists()]
    if pending:
        detector = YOLO(str(args.model))
        config = load_config(groups[0]['config']).as_dict()
        config['features'].update(iqa_musiq=False, iqa_clipiqa=False)
        backend = TorchBackend(Config(config))
        for group in pending:
            out = args.output/f'{group["date"]}-{group["group_alias"]}'
            out.mkdir(exist_ok=True)
            start = time.perf_counter()
            result = process_group(group,detector,backend,out,args.method)
            result['seconds'] = time.perf_counter()-start
            (out/'result.json').write_text(json.dumps(result,indent=2))
            print(group['date'],group['group_alias'],result['original_counts'],result['candidate_counts'],result['proposal_count'],result['refusals'],flush=True)
    results = [json.loads((args.output/f'{g["date"]}-{g["group_alias"]}/result.json').read_text()) for g in groups]
    summary = {'method':args.method, 'groups':len(results),
               'proposed_groups':sum(bool(r['proposal_count']) for r in results),
               'proposals':sum(r['proposal_count'] for r in results),
               'stable_observed_sets':sum(r['stable_observed_set'] for r in results),
               'stable_tracks':sum(r.get('stable_track_count',0) for r in results),
               'keeper_changed_groups':0, 'refusals':dict(sum((Counter(r['refusals']) for r in results),Counter()))}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
