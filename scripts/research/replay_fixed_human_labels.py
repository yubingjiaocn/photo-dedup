"""Separate fixed old human regression; never pooled with model labels."""
import json
import sqlite3
from collections import Counter

from real_event_evaluate import ROOT, current, old_development, pinned_module

BASE = ROOT.parent
human = BASE / 'human-audit-v2-20260909-r3-note-migration'
tasks = {t['task_id']: t for t in json.loads((human / 'manifest.json').read_text())['tasks']}
variants = json.loads((ROOT / 'frozen-runtime-variants.json').read_text())
baseline = pinned_module()
groups, _ = old_development()
pose_rows = json.loads((BASE / 'astra-usable-algorithm-20260909/pose-probe/pose-data.json').read_text())['frames']
poses = {(p['group_alias'], p['alias']): p for p in pose_rows}
by_id = {}
for group in groups:
    for frame in group['frames']:
        by_id[(group['dataset'], frame.get('id', frame.get('file_id')))] = poses.get((group['group_alias'], frame['alias']))
results, totals = [], {v: Counter() for v in variants}
for line in (human / 'labels-v2.jsonl').read_text().splitlines():
    label = json.loads(line)
    if label['provenance']['needs_review']:
        continue
    task = tasks[label['task_id']]
    dataset, gid = task['audit_id'].split(':G')
    folder = {'disney': 'disney-conservative', 'jx3': 'jx3-identity'}[dataset]
    conn = sqlite3.connect(f'file:{BASE / folder / "inventory.sqlite"}?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    group_type = conn.execute('SELECT group_type FROM groups WHERE id=?', (int(gid),)).fetchone()[0]
    members = [dict(r) for r in conn.execute('SELECT f.*,fe.quality_score,fe.quality_meta,fe.face_count,fe.faces_json,fe.dinov2_embedding FROM group_members gm JOIN files f ON f.id=gm.file_id JOIN features fe ON fe.file_id=f.id WHERE gm.group_id=? ORDER BY f.exif_timestamp,f.id', (int(gid),))]
    conn.close()
    for member in members:
        pose = by_id.get((dataset, member['id']))
        if pose:
            meta = json.loads(member['quality_meta'] or '{}')
            meta['pose_evidence'] = {'producer': 'yolo26s-pose.pt', 'status': 'ok', 'shape': pose['shape'], 'people': pose['people']}
            member['quality_meta'] = json.dumps(meta)
    aliases = dict(zip(task['aliases'], task['member_ids']))
    result = {'audit_id': task['audit_id'], 'variants': {}}
    for name, options in variants.items():
        module = baseline if name == 'baseline' else current
        selection = module.select_phase_keepers(members, module.segment_phases(members, group_type=group_type), **options)
        retained = {members[i]['id'] for i in selection['keepers']}
        metric = Counter(groups=1, keepers=len(retained), phase_misses=0, quality_wrong=0, quality_assessed_covered=0)
        for phase in label['phases']:
            kept = retained & {aliases[a] for a in phase['members']}
            metric['phase_misses'] += not bool(kept)
            if phase['keeper_status'] == 'assessed' and kept:
                metric['quality_assessed_covered'] += 1
                metric['quality_wrong'] += not bool(kept & {aliases[a] for a in phase['acceptable_keepers']})
        totals[name].update(metric)
        result['variants'][name] = {'keepers': sorted(retained), 'metrics': dict(metric)}
    results.append(result)
output = {'source': str(human), 'status': 'fixed_old_human_labels_development_exposed',
          'human_and_model_metrics_pooled': False, 'metrics': totals, 'groups': results}
(ROOT / 'fixed-human-regression.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(totals, indent=2))
