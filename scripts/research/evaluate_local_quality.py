"""Default-pinned, label-separated evaluation of native local-quality candidates."""
import argparse
import json
import sqlite3
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import phase_selection as current  # noqa: E402
from real_event_evaluate import assess  # noqa: E402

PIN = 'a61cad8'


def pinned():
    m = types.ModuleType('src._local_quality_baseline')
    m.__package__ = 'src'
    sys.modules[m.__name__] = m
    source = subprocess.check_output(['git','show',f'{PIN}:src/phase_selection.py'],cwd=REPO)
    exec(compile(source,f'{PIN}:phase_selection.py','exec'),m.__dict__)
    return m


def run(mapping_path, labels_path, database, variants_path, output, split):
    if output.exists():
        raise ValueError('Existing evaluation output; inspect or choose an explicit new development version')
    groups = json.loads(mapping_path.read_text())
    labels = {}
    for path in sorted(labels_path.glob('*.json')):
        packet = json.loads(path.read_text())
        if packet.get('status') != 'model_provisional':
            raise ValueError('New labels must be explicitly model_provisional')
        for group in packet['groups']:
            if group['group_alias'] in labels:
                raise ValueError('Duplicate group labels')
            labels[group['group_alias']] = group
    if set(labels) != {g['group_alias'] for g in groups}:
        raise ValueError('Labels must cover every complete multiframe group')
    variants = json.loads(variants_path.read_text())
    baseline = pinned()
    totals = {name: Counter() for name in variants}
    conn = sqlite3.connect(f'file:{database}?mode=ro&immutable=1',uri=True)
    conn.row_factory = sqlite3.Row
    rows = []
    for group in groups:
        ids = [f['id'] for f in group['frames']]
        aliases = [f['alias'] for f in group['frames']]
        fetched = {r['id']: dict(r) for r in conn.execute('SELECT f.*,fe.quality_score,fe.quality_meta,fe.face_count,fe.faces_json,fe.dinov2_embedding,fe.local_quality_embedding FROM files f JOIN features fe ON fe.file_id=f.id WHERE f.id IN ('+','.join('?' for _ in ids)+')',ids)}
        members = [fetched[i] for i in ids]
        old_phases = baseline.segment_phases(members,group_type=group['group_type'])
        new_phases = current.segment_phases(members,group_type=group['group_type'])
        baseline_start = time.perf_counter()
        original = baseline.select_phase_keepers(members,old_phases)
        baseline_seconds = time.perf_counter()-baseline_start
        unchanged = current.select_phase_keepers(members,new_phases)
        assert original['keepers'] == unchanged['keepers']
        assert original['utility_scores'] == unchanged['utility_scores']
        label = labels[group['group_alias']]
        row = {'group_alias':group['group_alias'],'members':len(ids),'scene_tags':label.get('scene_tags',[]),
               'phases':label['phases'],'variants':{}}
        for name, options in variants.items():
            start = time.perf_counter()
            selection = original if name=='baseline' else current.select_phase_keepers(members,new_phases,**options)
            elapsed = baseline_seconds if name=='baseline' else time.perf_counter()-start
            metric = assess(label['phases'],aliases,selection)
            metric['selection_seconds'] = elapsed
            metric['changed_groups'] = int(set(selection['keepers']) != set(original['keepers']))
            metric['local_evidence_changed_groups'] = int(bool(selection.get('local_quality_context',{}).get('changed_members')))
            pose_context = selection.get('pose_coverage') or {}
            metric['pose_trigger_groups'] = int(pose_context.get('added_keeper') is not None)
            metric['pose_abstain_groups'] = int(bool(pose_context) and pose_context.get('added_keeper') is None)
            keep = {aliases[i] for i in selection['keepers']}
            metric['usable_required_phases'] = 0
            metric['adjudicable_required_phases'] = 0
            for phase in label['phases']:
                if phase['required'] and phase['quality_status']=='adjudicable':
                    metric['adjudicable_required_phases'] += 1
                    metric['usable_required_phases'] += bool(keep & set(phase['acceptable_keeper_aliases']))
            totals[name].update(metric)
            row['variants'][name] = {'keepers':[aliases[i] for i in selection['keepers']],
                'metrics':metric,'review':selection['review_required'],'budget':selection['group_keeper_budget'],
                'local_quality_context':selection.get('local_quality_context'),
                'pose_context':selection.get('pose_coverage')}
        rows.append(row)
    conn.close()
    count = sum(len(g['frames']) for g in groups)
    for metric in totals.values():
        metric['retention'] = metric['keepers']/count if count else 0
    result = {'baseline_pin':PIN,'split':split,'label_status':'model_provisional','groups':len(groups),
              'photos':count,'variants':variants,'metrics':totals,'rows':rows,
              'default_exact_keeper_and_score_parity_groups':len(groups),'runtime_label_input':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['mapping','labels','db','variants','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['development','holdout'],required=True)
    a=p.parse_args()
    run(a.mapping,a.labels,a.db,a.variants,a.output,a.split)
