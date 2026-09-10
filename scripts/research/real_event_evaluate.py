"""Frozen real-event A/B; labels are evaluation-only, never runtime input."""
from __future__ import annotations

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
ROOT = Path('/home/ubuntu/photo-dedup-eval/astra-real-events-20260910')
PIN = '09ea67adfe03cb3d27278d2de39c243c39e1338b'
sys.path.insert(0, str(REPO))
from src import phase_selection as current  # noqa: E402
from src.pose_evidence import protect_pose_variants  # noqa: E402


def pinned_module():
    source = subprocess.check_output(['git', 'show', f'{PIN}:src/phase_selection.py'], cwd=REPO)
    module = types.ModuleType('src._real_event_baseline')
    sys.modules[module.__name__] = module
    module.__package__ = 'src'
    exec(compile(source, f'{PIN}:phase_selection.py', 'exec'), module.__dict__)
    return module


def members_for(group):
    db = group.get('db') or group['source_db']
    ids = [f.get('id', f.get('file_id')) for f in group['frames']]
    conn = sqlite3.connect(f'file:{db}?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    rows = {r['id']: dict(r) for r in conn.execute(
        'SELECT f.*,fe.quality_score,fe.quality_meta,fe.face_count,fe.faces_json,fe.dinov2_embedding FROM files f JOIN features fe ON f.id=fe.file_id WHERE f.id IN (' + ','.join('?' for _ in ids) + ')', ids)}
    conn.close()
    return [rows[i] for i in ids]


def assess(phases, aliases, selection):
    keep = {aliases[i] for i in selection['keepers']}
    flat = [a for p in phases for a in p['member_aliases']]
    if len(flat) != len(set(flat)) or set(flat) != set(aliases):
        raise ValueError('Label phases do not partition group')
    result = {'keepers': len(keep), 'review_groups': int(selection['review_required']),
              'phase_misses': 0, 'high_confidence_phase_misses': 0, 'quality_wrong': 0,
              'quality_assessed_covered': 0, 'selected_unacceptable': 0, 'required_phases': 0}
    for phase in phases:
        retained = keep.intersection(phase['member_aliases'])
        if phase['required']:
            result['required_phases'] += 1
            result['phase_misses'] += not bool(retained)
            result['high_confidence_phase_misses'] += not retained and phase['confidence'] == 'high'
        if phase['quality_status'] == 'adjudicable':
            acceptable = set(phase['acceptable_keeper_aliases'])
            if not acceptable or not acceptable <= set(phase['member_aliases']):
                raise ValueError('Invalid acceptable keeper label')
            result['quality_assessed_covered'] += bool(retained)
            result['quality_wrong'] += bool(retained) and not bool(retained & acceptable)
            result['selected_unacceptable'] += len(retained - acceptable)
    return result


def run(groups, labels, variants, output, old_poses=None):
    baseline = pinned_module()
    totals = {name: Counter() for name in variants}
    rows = []
    for group in groups:
        alias = group['group_alias']
        members = members_for(group)
        aliases = [f['alias'] for f in group['frames']]
        if old_poses is not None:
            for member, frame_alias in zip(members, aliases):
                pose = old_poses[(alias, frame_alias)]
                meta = json.loads(member['quality_meta'] or '{}')
                meta['pose_evidence'] = {'producer': 'yolo26s-pose.pt', 'status': 'ok',
                                         'shape': pose['shape'], 'people': pose['people']}
                member['quality_meta'] = json.dumps(meta)
        label = labels[alias]
        bp = baseline.segment_phases(members, group_type=group['group_type'])
        cp = current.segment_phases(members, group_type=group['group_type'])
        row = {'group_alias': alias, 'members': len(members), 'group_type': group['group_type'],
               'scene_tags': label.get('scene_tags', []), 'phases': label['phases'], 'variants': {}}
        for name, config in variants.items():
            module, phases = (baseline, bp) if name == 'baseline' else (current, cp)
            start = time.perf_counter()
            selection_config = dict(config)
            pose_threshold = selection_config.pop('_pose_displacement', None)
            participant_policy = selection_config.pop('_pose_participant_policy', 'consensus')
            selected = module.select_phase_keepers(members, phases, **selection_config)
            if pose_threshold is not None:
                retained, evidence = protect_pose_variants(members, selected['keepers'], selected['utility_scores'],
                                                          displacement_threshold=pose_threshold,
                                                          participant_policy=participant_policy)
                selected['keepers'] = retained
                selected['group_keeper_budget'] = max(selected['group_keeper_budget'], len(retained))
                selected['review_required'] = selected['review_required'] or evidence['added_keeper'] is not None
            elapsed = time.perf_counter() - start
            metrics = assess(label['phases'], aliases, selected)
            totals[name].update(metrics)
            totals[name]['selection_seconds'] += elapsed
            row['variants'][name] = {'keepers': [aliases[i] for i in selected['keepers']],
                                    'review': selected['review_required'], 'metrics': metrics,
                                    'runtime_phase_count': len(phases), 'budget': selected['group_keeper_budget']}
        rows.append(row)
    size = sum(len(g['frames']) for g in groups)
    summary = {name: {**metrics, 'retention': metrics['keepers'] / size,
                      'changed_groups': sum(r['variants'][name]['keepers'] != r['variants']['baseline']['keepers'] for r in rows)}
               for name, metrics in totals.items()}
    result = {'baseline_commit': PIN, 'model_label_status': 'model_provisional',
              'groups': len(rows), 'photos': size, 'variants': variants, 'metrics': summary, 'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))


def old_development():
    old = ROOT.parent / 'astra-usable-algorithm-20260909'
    mappings = []
    labels = {}
    for panel in ['development-panel', 'development-expanded', 'newday-panel']:
        mappings.extend(json.loads((old / panel / 'sealed/mapping.json').read_text())['groups'])
    for file in ['guarded-development-ab.json', 'guarded-expanded-development-ab.json', 'guarded-newday-development-ab.json']:
        for row in json.loads((old / file).read_text())['groups']:
            labels[row['group_alias']] = {'phases': row['model_phases'], 'scene_tags': row['scene_tags']}
    for group in mappings:
        group['db'] = group.get('source_db') or str(ROOT.parent / {'jx3': 'jx3-identity', 'disney': 'disney-conservative'}[group['dataset']] / 'inventory.sqlite')
        conn = sqlite3.connect(f'file:{group["db"]}?mode=ro&immutable=1', uri=True)
        group['group_type'] = conn.execute('SELECT group_type FROM groups WHERE id=?', (group['group_id'],)).fetchone()[0] if isinstance(group['group_id'], int) else 'singleton'
        conn.close()
    return [g for g in mappings if len(g['frames']) > 1], labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-development', action='store_true')
    parser.add_argument('--day')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--old-poses', action='store_true')
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.old_development:
        groups, labels = old_development()
    else:
        groups = json.loads((ROOT / 'panels' / args.day / 'mapping.json').read_text())
        labels = {}
        for p in sorted(args.labels.glob('*.json')):
            packet = json.loads(p.read_text())
            if packet.get('status') != 'model_provisional':
                raise ValueError('Labels must be explicitly model_provisional')
            for row in packet['groups']:
                if row['group_alias'] in labels:
                    raise ValueError('Duplicate group labels')
                labels[row['group_alias']] = row
        if set(labels) != {g['group_alias'] for g in groups}:
            raise ValueError('All complete groups need blind labels')
    if args.db:
        for group in groups:
            group['db'] = str(args.db)
    poses = None
    if args.old_poses:
        if not args.old_development:
            raise ValueError('Old pose data is development-only')
        pose_path = ROOT.parent / 'astra-usable-algorithm-20260909/pose-probe/pose-data.json'
        poses = {(p['group_alias'], p['alias']): p for p in json.loads(pose_path.read_text())['frames']}
    run(groups, labels, json.loads(args.variants.read_text()), args.output, poses)
