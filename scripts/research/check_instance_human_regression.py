"""Read fixed human fixtures; compare off/review-only without regeneration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src import phase_selection as ps  # noqa: E402


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Existing human regression result')
    base=Path('/home/ubuntu/photo-dedup-eval')
    human=base/'human-audit-v2-20260909-r3-note-migration'
    tasks={t['task_id']:t for t in json.loads((human/'manifest.json').read_text())['tasks']}
    totals={v:Counter() for v in ['off','review_only']}
    results=[]
    for line in (human/'labels-v2.jsonl').read_text().splitlines():
        label=json.loads(line)
        if label['provenance']['needs_review']:
            continue
        task=tasks[label['task_id']]
        dataset,gid=task['audit_id'].split(':G')
        database=base/f'astra-local-quality-20260910/human-regression/{dataset}/local.sqlite'
        c=sqlite3.connect(f'file:{database}?mode=ro&immutable=1',uri=True)
        c.row_factory=sqlite3.Row
        group_type=c.execute('SELECT group_type FROM groups WHERE id=?',(int(gid),)).fetchone()[0]
        members=[dict(r) for r in c.execute('SELECT f.*,fe.* FROM group_members gm JOIN files f ON f.id=gm.file_id JOIN features fe ON fe.file_id=f.id WHERE gm.group_id=? ORDER BY f.exif_timestamp,f.id',(int(gid),))]
        c.close()
        aliases=dict(zip(task['aliases'],task['member_ids']))
        phases=ps.segment_phases(members,group_type=group_type)
        selections={v:ps.select_phase_keepers(members,phases,instance_recovery_policy=v) for v in totals}
        assert selections['off']['keepers']==selections['review_only']['keepers']
        assert selections['off']['utility_scores']==selections['review_only']['utility_scores']
        assert not selections['review_only']['instance_recovery_context']['proposed_groups']
        result={'audit_id':task['audit_id'],'variants':{}}
        for variant,selection in selections.items():
            kept={members[i]['id'] for i in selection['keepers']}
            metric=Counter(groups=1,keepers=len(kept),phase_misses=0,quality_wrong=0,quality_assessed_covered=0)
            for phase in label['phases']:
                retained=kept & {aliases[x] for x in phase['members']}
                metric['phase_misses']+=not bool(retained)
                if phase['keeper_status']=='assessed' and retained:
                    metric['quality_assessed_covered']+=1
                    metric['quality_wrong']+=not bool(retained & {aliases[x] for x in phase['acceptable_keepers']})
            totals[variant].update(metric)
            result['variants'][variant]={'keepers':sorted(kept),'metrics':metric}
        results.append(result)
    output={'status':'fixed_old_human_regression','metrics':totals,'groups':results,
            'human_and_model_metrics_pooled':False,'feature_regeneration':False,'keeper_order_exact_parity':True}
    a.output.write_text(json.dumps(output,indent=2))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    main()
