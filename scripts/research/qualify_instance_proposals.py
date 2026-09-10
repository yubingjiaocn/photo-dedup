"""Qualify saved native observations without rerunning models or changing sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.instance_recovery import novel_detection, stable_tracks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Qualified snapshot exists; inspect it instead of replacing')
    paths=sorted(a.source.glob('*/result.json'))
    if len(paths)!=8:
        raise ValueError('Expected completed frozen 8-group slice')
    a.output.mkdir(parents=True)
    summaries=[]
    for path in paths:
        result=json.loads(path.read_text())
        valid=[]
        rejected=[]
        frames=result['group']['frames']
        for proposal in result['packet']['proposals']:
            target=next(i for i,f in enumerate(frames) if f['id']==proposal['target_file_id'])
            original=result['catalogs'][target][:result['original_counts'][target]]
            proposal['novelty_checked']=novel_detection(result['catalogs'][target][proposal['target_index']],original)
            if proposal['novelty_checked']:
                valid.append(proposal)
            else:
                rejected.append({**proposal,'rejection':'EXISTING_INSTANCE_OVERLAP_OR_FRAGMENT'})
        result['packet']['proposals']=valid
        result['proposal_count']=len(valid)
        result['rejected_proposals']=rejected
        edges={(x['source'],x['target'],x['source_index']):x['target_index'] for x in result['links']
               if x.get('appearance',{}).get('eligible') and x.get('motion',{}).get('eligible')
               and x.get('confirmation',{}).get('eligible')}
        tracks=stable_tracks(result['catalogs'],edges)
        result['stable_tracks']=tracks
        result['stable_track_count']=len(tracks)
        result['stable_observed_set']=bool(tracks) and all(len(c)==len(tracks) for c in result['catalogs'])
        result['packet']['stable_observed_set']=result['stable_observed_set']
        result['qualification']='cross-class overlap/fragment rejection; no model rerun'
        out=a.output/path.parent.name
        out.mkdir()
        (out/'result.json').write_text(json.dumps(result,indent=2))
        for panel in result['panels']:
            shutil.copyfile(path.parent/panel,out/panel)
        summaries.append(result)
    summary={'groups':len(summaries),'proposed_groups':sum(bool(r['proposal_count']) for r in summaries),
             'proposals':sum(r['proposal_count'] for r in summaries),
             'stable_observed_sets':sum(r['stable_observed_set'] for r in summaries),
             'stable_tracks':sum(r['stable_track_count'] for r in summaries),
             'keeper_changed_groups':0,'source':str(a.source),
             'refusals':dict(sum((Counter(r['refusals']) for r in summaries),Counter()))}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
