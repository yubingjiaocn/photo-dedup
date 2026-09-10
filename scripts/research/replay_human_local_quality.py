"""Fixed old human labels kept separate; native feature fixtures only."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

from enrich_local_quality import enrich
from evaluate_local_quality import current, pinned
from src import db

BASE = Path('/home/ubuntu/photo-dedup-eval')
ROOT = BASE / 'astra-local-quality-20260910'
HUMAN = BASE / 'human-audit-v2-20260909-r3-note-migration'


def main():
    tasks = {t['task_id']:t for t in json.loads((HUMAN/'manifest.json').read_text())['tasks']}
    records = []
    for line in (HUMAN/'labels-v2.jsonl').read_text().splitlines():
        label = json.loads(line)
        if not label['provenance']['needs_review']:
            task=tasks[label['task_id']]
            dataset,gid=task['audit_id'].split(':G')
            records.append((dataset,int(gid),task,label))
    for dataset in sorted({x[0] for x in records}):
        folder={'disney':'disney-conservative','jx3':'jx3-identity'}[dataset]
        src=sqlite3.connect(f'file:{BASE/folder/"inventory.sqlite"}?mode=ro&immutable=1',uri=True)
        src.row_factory=sqlite3.Row
        dest=ROOT/'human-regression'/dataset/'input.sqlite'
        if dest.exists():
            raise ValueError('Existing fixture: inspect instead of overwriting')
        dest.parent.mkdir(parents=True,exist_ok=True)
        target=db.open_db(dest)
        gids=[x[1] for x in records if x[0]==dataset]
        marks=','.join('?' for _ in gids)
        ids=[r[0] for r in src.execute(f'SELECT DISTINCT file_id FROM group_members WHERE group_id IN ({marks})',gids)]
        for table,key,values in [('files','id',ids),('features','file_id',ids),('groups','id',gids),('group_members','group_id',gids)]:
            for row in src.execute(f'SELECT * FROM {table} WHERE {key} IN ({",".join("?" for _ in values)})',values):
                data=dict(row)
                names=','.join('"'+k+'"' for k in data)
                target.execute(f'INSERT INTO {table} ({names}) VALUES ({",".join("?" for _ in data)})',list(data.values()))
        target.commit()
        target.close()
        src.close()
        enrich(dest,dest.with_name('local.sqlite'),
               BASE/'astra-real-events-20260910/cache/2026-04-04/config.yaml',
               BASE/'astra-usable-algorithm-20260909/person-instance-probe/models/yolo26s-seg.pt')
    variants=json.loads((ROOT/'frozen-variants.json').read_text())
    baseline=pinned()
    totals={v:Counter() for v in variants}
    results=[]
    for dataset,gid,task,label in records:
        c=sqlite3.connect(f'file:{ROOT}/human-regression/{dataset}/local.sqlite?mode=ro&immutable=1',uri=True)
        c.row_factory=sqlite3.Row
        group_type=c.execute('SELECT group_type FROM groups WHERE id=?',(gid,)).fetchone()[0]
        members=[dict(r) for r in c.execute('SELECT f.*,fe.* FROM group_members gm JOIN files f ON f.id=gm.file_id JOIN features fe ON fe.file_id=f.id WHERE gm.group_id=? ORDER BY f.exif_timestamp,f.id',(gid,))]
        c.close()
        aliases=dict(zip(task['aliases'],task['member_ids']))
        result={'audit_id':task['audit_id'],'variants':{}}
        for name,options in variants.items():
            module=baseline if name=='baseline' else current
            selected=module.select_phase_keepers(members,module.segment_phases(members,group_type=group_type),**options)
            kept={members[i]['id'] for i in selected['keepers']}
            metric=Counter(groups=1,keepers=len(kept),phase_misses=0,quality_wrong=0,quality_assessed_covered=0)
            for phase in label['phases']:
                retained=kept & {aliases[a] for a in phase['members']}
                metric['phase_misses']+=not bool(retained)
                if phase['keeper_status']=='assessed' and retained:
                    metric['quality_assessed_covered']+=1
                    metric['quality_wrong']+=not bool(retained & {aliases[a] for a in phase['acceptable_keepers']})
            totals[name].update(metric)
            result['variants'][name]={'keepers':sorted(kept),'metrics':dict(metric)}
        results.append(result)
    output={'status':'fixed_old_human_development_regression','source':str(HUMAN),
            'human_and_model_metrics_pooled':False,'metrics':totals,'groups':results}
    (ROOT/'HUMAN-REGRESSION.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    main()
