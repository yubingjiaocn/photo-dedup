"""Actual Stage2/3 shadow A/B on copied development databases.

Model proposals are produced without labels. Evaluation labels/strata are read
only after both replay variants have finished. This script never runs Stage4.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys

import yaml

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src import stage2_cluster, stage3_report, phase_selection as ps
from real_event_evaluate import assess

BASE=Path('/home/ubuntu/photo-dedup-eval')
OLD=BASE/'astra-local-quality-20260910'
DAYS=['2026-04-04','2026-04-05','2026-07-31','2026-07-25','2026-07-26']


def connect(path):
    c=sqlite3.connect(f'file:{path}?mode=ro&immutable=1',uri=True)
    c.row_factory=sqlite3.Row
    return c


def snapshot(path):
    c=connect(path)
    groups={}
    for row in c.execute('SELECT * FROM groups'):
        members=list(c.execute('SELECT file_id,is_keep,decision FROM group_members WHERE group_id=?',(row['id'],)))
        ids=frozenset(m['file_id'] for m in members)
        decision=json.loads(row['decision_json'] or '{}')
        groups[ids]={'group_type':row['group_type'],'keep_file_id':row['keep_file_id'],
                     'keepers':sorted(m['file_id'] for m in members if m['is_keep']),
                     'decision_state':row['decision_state'],'decision':decision}
    unsafe=c.execute("SELECT COUNT(*) FROM group_members gm JOIN groups g ON gm.group_id=g.id JOIN features f ON gm.file_id=f.file_id JOIN features k ON g.keep_file_id=k.file_id WHERE gm.decision='AUTO_REMOVE' AND (g.group_type!='sha_exact' OR f.content_sha256 IS NULL OR k.content_sha256 IS NULL OR f.content_sha256!=k.content_sha256)").fetchone()[0]
    c.close()
    return groups,unsafe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    packets=defaultdict(list)
    for path in sorted(a.proposals.glob('*/result.json')):
        result=json.loads(path.read_text())
        packets[result['day']].append(result)
    outcomes=[]
    for day in DAYS:
        origin=OLD if day.startswith('2026-07-2') else BASE/'astra-real-events-20260910'
        source=BASE/f'astra-regime-validation-20260910/development/{day}/inventory.sqlite'
        mapping_path=origin/f'panels/{day}/mapping.json'
        mapping=json.loads(mapping_path.read_text())
        completed=a.output/day/'result.json'
        if completed.exists():
            outcomes.append(json.loads(completed.read_text()))
            continue
        snapshots={}
        for variant in ['off','review_only']:
            out=a.output/day/variant
            out.mkdir(parents=True,exist_ok=True)
            database=out/'inventory.sqlite'
            marker=out/'replay-complete.json'
            if not marker.exists():
                if database.exists():
                    raise ValueError(f'Partial replay found: inspect {out}; do not blindly overwrite')
                c=connect(source)
                dest=sqlite3.connect(database)
                c.backup(dest)
                c.close()
                for result in packets[day]:
                    packet=result['packet']
                    for fid in packet['group_member_ids']:
                        row=dest.execute('SELECT quality_meta FROM features WHERE file_id=?',(fid,)).fetchone()
                        meta=json.loads(row[0] or '{}')
                        meta['instance_recovery']=packet
                        dest.execute('UPDATE features SET quality_meta=? WHERE file_id=?',(json.dumps(meta),fid))
                dest.commit()
                dest.close()
                config=yaml.safe_load((origin/f'cache/{day}/config.yaml').read_text())
                config['cluster']['keeper_instance_recovery_policy']=variant
                config['paths'].update(db=str(database),output_dir=str(out),trash=str(out/'unused-trash'))
                config['execute']['dry_run']=True
                cfg=out/'config.yaml'
                cfg.write_text(yaml.safe_dump(config))
                stats=stage2_cluster.run(str(cfg))
                report=stage3_report.run(str(cfg))
                marker.write_text(json.dumps({'actual_stage2':True,'actual_stage3':True,'stats':stats,'report':report,'stage4_executed':False},indent=2))
            snapshots[variant],unsafe=snapshot(database)
            assert unsafe==0
        before,after=snapshots['off'],snapshots['review_only']
        assert before.keys()==after.keys()
        queues=[]
        for ids,left in before.items():
            right=after[ids]
            for key in ['keepers','keep_file_id','group_type']:
                assert left[key]==right[key],(day,ids,key)
            if right['group_type']!='sha_exact':
                assert right['decision_state']=='REVIEW_REQUIRED'
                lc=left['decision']['phase_selection']
                rc=right['decision']['phase_selection']
                assert lc['keepers']==rc['keepers']
                assert lc['utility_scores']==rc['utility_scores']
                assert lc['group_keeper_budget']==rc['group_keeper_budget']
                context=rc['instance_recovery_context']
                if context['proposed_groups']:
                    queues.append({'group_member_ids':sorted(ids),'keepers':right['keepers'],**context})
        # Evaluation-only inputs begin here. They do not enter stage2 or producers.
        reference=json.loads((BASE/f'astra-regime-validation-20260910/DEVELOPMENT-{day}.json').read_text())
        refs={r['group_alias']:r for r in reference['rows']}
        regimes={}
        for path in (OLD/'region-set/evaluation').glob('*-regimes.json'):
            for g in json.loads(path.read_text())['groups']:
                regimes[g['day'],g['group_alias']]=g['regime']
        totals=defaultdict(lambda:defaultdict(Counter))
        details=[]
        c=connect(a.output/day/'review_only/inventory.sqlite')
        for g in mapping:
            ids=[f['id'] for f in g['frames']]
            aliases=[f['alias'] for f in g['frames']]
            dbrows={r['id']:dict(r) for r in c.execute('SELECT f.*,fe.quality_score,fe.quality_meta,fe.face_count,fe.faces_json,fe.dinov2_embedding,fe.local_quality_embedding FROM files f JOIN features fe ON fe.file_id=f.id WHERE f.id IN ('+','.join('?' for _ in ids)+')',ids)}
            members=[dbrows[i] for i in ids]
            phases=ps.segment_phases(members,group_type=g['group_type'])
            previous=None
            metrics={}
            for variant in ['off','review_only']:
                selection=ps.select_phase_keepers(members,phases,instance_recovery_policy=variant)
                keep=[aliases[i] for i in selection['keepers']]
                assert keep==refs[g['group_alias']]['variants']['baseline']['keepers']
                assert {ids[i] for i in selection['keepers']}==set(snapshots[variant][frozenset(ids)]['keepers'])
                if previous is not None:
                    assert previous['keepers']==selection['keepers']
                    assert previous['utility_scores']==selection['utility_scores']
                previous=selection
                metric=assess(refs[g['group_alias']]['phases'],aliases,selection)
                metric['groups']=1
                metric['photos']=len(ids)
                metric['proposed_groups']=int(bool(selection.get('instance_recovery_context',{}).get('proposed_groups')))
                for stratum in [regimes[day,g['group_alias']],'overall']:
                    totals[stratum][variant].update(metric)
                metrics[variant]={'keepers':keep,'metrics':metric}
            details.append({'group_alias':g['group_alias'],'regime':regimes[day,g['group_alias']],'variants':metrics})
        c.close()
        result={'day':day,'groups':len(mapping),'metrics':totals,'rows':details,
                'proposed_groups':queues,'changed_keeper_groups':0,'unsafe_auto_remove':0,
                'keeper_order_and_primary_exact_parity':True,'actual_stage2_stage3_both_variants':True,'stage4_executed':False}
        completed.write_text(json.dumps(result,indent=2))
        outcomes.append(result)
        print(day,'groups',len(mapping),'proposed',len(queues),'unsafe',0,flush=True)
    totals=defaultdict(lambda:defaultdict(Counter))
    for day in outcomes:
        for stratum,variants in day['metrics'].items():
            for variant,metric in variants.items():
                totals[stratum][variant].update(metric)
    summary={'groups':sum(o['groups'] for o in outcomes),'metrics':totals,
             'proposed_groups':sum(len(o['proposed_groups']) for o in outcomes),
             'keeper_changed_groups':0,'unsafe_auto_remove':0,'actual_stage2_stage3_both_variants':True,
             'document_screen_support':'not evaluated; no samples',
             'no_subject_policy':'unchanged global baseline; evaluation strata are never runtime inputs',
             'stage4_executed':False,'label_status':'existing exposed model_provisional'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
