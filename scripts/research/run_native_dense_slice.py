"""Frozen dense visible-region correspondence across the existing eight groups.

Reuses the successful two-frame spike caches. Pair proposals are review-only;
track completeness and newly observed regions are reported separately.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageOps

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.config import Config,load_config
from src.stage1_backends import TorchBackend
from src.instance_recovery import novel_detection
from src.instance_correspondence import track_components
from src.local_quality import _iou
from probe_native_dense_tracks import dense,compare
from run_foreground_correspondence import detect_view,choose,save


def run_group(group,root,out,detector,backend):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        return json.loads((out/'result.json').read_text())
    prior=json.loads((root/f'candidate/{group["date"]}-{group["group_alias"]}/result.json').read_text())
    objects=[];descriptors=[];images=[]
    c=sqlite3.connect(f'file:{group["source_db"]}?mode=ro&immutable=1',uri=True)
    for f,frame in enumerate(group['frames']):
        row=c.execute('SELECT path,size_bytes,mtime_ns FROM files WHERE id=?',(frame['id'],)).fetchone()
        stat=Path(row[0]).stat()
        if row[0]!=frame['path'] or (stat.st_size,stat.st_mtime_ns)!=(row[1],row[2]):
            raise ValueError('Native input changed')
        with Image.open(frame['path']) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
        images.append(image)
        w,h=image.size;windows=[[0,0,w,h]]
        for y1,y2 in [(0,int(h*.6)),(int(h*.4),h)]:
            for x1,x2 in [(0,int(w*.6)),(int(w*.4),w)]:
                windows.append([x1,y1,x2,y2])
        original=json.loads((root/f'next-class-scale/{group["date"]}-{group["group_alias"]}-{frame["alias"]}.json').read_text())
        frame_out=out/frame['alias'];frame_out.mkdir(exist_ok=True)
        reuse=root/f'dense-pilot/{frame["alias"]}'
        if group['group_alias']=='G018' and reuse.exists():
            for pattern in ['window-*.json','dense-*.npz']:
                for path in reuse.glob(pattern):
                    if not (frame_out/path.name).exists():
                        shutil.copy2(path,frame_out/path.name)
        obs=[];desc=[]
        for index,seed in enumerate(original['observations']):
            if seed['class_id'] not in (0,77):
                continue
            candidates=detect_view(detector,image,windows[seed['window']],frame_out/f'window-{seed["window"]}.json')
            selected,why=choose(seed,candidates)
            if selected is None:
                continue
            selected={**selected,'source_observation_index':index,'detector_confirmed':True,
                      'frame':frame['alias'],'whole_body_completeness':'unknown',
                      'new_vs_v0_catalog':novel_detection(selected,prior['catalogs'][f])}
            cache=frame_out/f'dense-{index}.npz'
            if cache.exists():
                with np.load(cache) as loaded:
                    value={k:loaded[k] for k in loaded.files}
                why=None
            else:
                value,why=dense(backend,image,selected)
                if value is not None:
                    np.savez_compressed(cache,**value)
            selected['descriptor_refusal']=why
            obs.append(selected);desc.append(value)
        objects.append(obs);descriptors.append(desc)
        print(group['group_alias'],frame['alias'],'observations',len(obs),flush=True)
    c.close()
    pairs=[];admitted=[];edges=[]
    for s in range(len(objects)):
        for t in range(s+1,len(objects)):
            population=[]
            for i,oa in enumerate(objects[s]):
                for j,ob in enumerate(objects[t]):
                    measured=compare(descriptors[s][i],descriptors[t][j],oa,ob)
                    population.append({'source_frame':s,'target_frame':t,'source_index':i,'target_index':j,**measured})
            eligible=[p for p in population if p['eligible']]
            for pair in eligible:
                competitors=[p['score'] for p in eligible if p is not pair and
                             (p['source_index']==pair['source_index'] or p['target_index']==pair['target_index'])]
                pair['object_uniqueness_ratio']=pair['score']/max(competitors) if competitors else None
                if ((not competitors or pair['score']>=max(competitors)*1.5)
                    and _iou(objects[s][pair['source_index']]['box'],objects[t][pair['target_index']]['box'])>=.5):
                    pair['source_new_vs_v0']=objects[s][pair['source_index']]['new_vs_v0_catalog']
                    pair['target_new_vs_v0']=objects[t][pair['target_index']]['new_vs_v0_catalog']
                    admitted.append(pair)
                    edges.extend([((s,pair['source_index']),(t,pair['target_index'])),((t,pair['target_index']),(s,pair['source_index']))])
            pairs.extend(population)
    tracks=track_components([len(o) for o in objects],edges)
    proposed=[p for p in admitted if p['source_new_vs_v0'] or p['target_new_vs_v0']]
    new_nodes={(p[side+'_frame'],p[side+'_index']) for p in proposed for side in ['source','target'] if p[side+'_new_vs_v0']}
    confirmed_nodes={(p[side+'_frame'],p[side+'_index']) for p in admitted for side in ['source','target']}
    for f,(image,obs) in enumerate(zip(images,objects)):
        panel=image.copy();panel.thumbnail((1000,800));draw=ImageDraw.Draw(panel)
        sx,sy=panel.width/image.width,panel.height/image.height
        for i,o in enumerate(obs):
            color='#00ff80' if (f,i) in new_nodes else ('#80c0ff' if (f,i) in confirmed_nodes else '#ffc040')
            box=[v*(sx if k%2==0 else sy) for k,v in enumerate(o['box'])]
            draw.rectangle(box,outline=color,width=2)
            draw.text((box[0],box[1]),f'{i} cls{o["class_id"]}',fill=color,stroke_width=1,stroke_fill='black')
        panel.save(out/f'{group["frames"][f]["alias"]}.jpg',quality=90)
    result={'day':group['date'],'group_alias':group['group_alias'],'group':group,'objects':objects,
            'pairs':pairs,'accepted_pairs':admitted,'new_observation_pair_proposals':proposed,
            'tracks':tracks,'new_frame_observations':len(new_nodes),
            'complete_subject_set':'unknown','keeper_authority':False,'runtime_admitted':False,
            'refusals':dict(Counter(p['reason'] for p in pairs if not p['eligible']))}
    save(out/'result.json',result)
    print(group['group_alias'],'pairs',len(admitted),'new pairs',len(proposed),'new frames',len(new_nodes),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    a=p.parse_args()
    if not a.model.is_file():
        raise ValueError('Existing model required')
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_CONFIG_DIR=str(a.output/'ultralytics-settings'))
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    groups=json.loads((a.root/'slice.json').read_text())['groups']
    detector=YOLO(str(a.model))
    cfg=load_config(groups[0]['config']).as_dict();cfg['features'].update(iqa_musiq=False,iqa_clipiqa=False)
    backend=TorchBackend(Config(cfg))
    results=[run_group(g,a.root,a.output/f'{g["date"]}-{g["group_alias"]}',detector,backend) for g in groups]
    summary={'groups':len(results),'frames':sum(len(g['group']['frames']) for g in results),
             'new_observation_proposed_groups':sum(bool(g['new_observation_pair_proposals']) for g in results),
             'accepted_pair_proposals':sum(len(g['accepted_pairs']) for g in results),
             'new_observation_pair_proposals':sum(len(g['new_observation_pair_proposals']) for g in results),
             'new_frame_observations':sum(g['new_frame_observations'] for g in results),
             'cycle_consistent_tracklets':sum(len(g['tracks']) for g in results),
             'all_frames_observed_tracklets':sum(t['all_frames_observed'] for g in results for t in g['tracks']),
             'complete_semantic_subject_sets_confirmed':0,'runtime_admitted':False,'keeper_changed_groups':0}
    save(a.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
