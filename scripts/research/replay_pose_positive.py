"""Actual Stage2 regression fixture from fixed-human positive cases, not holdout."""
import json
import sqlite3
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/ubuntu/photo-dedup-eval/astra-real-events-20260910')
sys.path.insert(0, str(REPO))
from src import db, stage2_cluster  # noqa: E402

human = json.loads((ROOT / 'fixed-human-regression.json').read_text())
poses = json.loads((ROOT.parent / 'astra-usable-algorithm-20260909/pose-probe/pose-data.json').read_text())['frames']
by_sha = {p['original_sha256']: p for p in poses}
proof = []
for case in human['groups']:
    if case['variants']['pose_040']['metrics']['phase_misses'] >= case['variants']['baseline']['metrics']['phase_misses']:
        continue
    dataset, gid = case['audit_id'].split(':G')
    folder = {'disney': 'disney-conservative', 'jx3': 'jx3-identity'}[dataset]
    source = sqlite3.connect(f'file:{ROOT.parent / folder / "inventory.sqlite"}?mode=ro&immutable=1', uri=True)
    source.row_factory = sqlite3.Row
    ids = [r[0] for r in source.execute('SELECT file_id FROM group_members WHERE group_id=?', (int(gid),))]
    out = ROOT / 'stage2-positive-regression' / f'{dataset}-{gid}'
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'inventory.sqlite'
    if destination.exists():
        raise ValueError('Existing regression output; do not overwrite')
    target = db.open_db(destination)
    paths = []
    for table, key in [('files', 'id'), ('features', 'file_id')]:
        rows = source.execute(f'SELECT * FROM {table} WHERE {key} IN ({",".join("?" for _ in ids)})', ids).fetchall()
        for row in rows:
            values = dict(row)
            if table == 'files':
                paths.append(Path(values['path']))
            else:
                p = by_sha[values['content_sha256']]
                meta = json.loads(values['quality_meta'] or '{}')
                meta['pose_evidence'] = {'producer': 'yolo26s-pose.pt', 'status': 'ok', 'shape': p['shape'], 'people': p['people']}
                values['quality_meta'] = json.dumps(meta)
            columns = ','.join('"'+c+'"' for c in values)
            target.execute(f'INSERT INTO {table} ({columns}) VALUES ({",".join("?" for _ in values)})', list(values.values()))
    target.commit()
    target.close()
    source.close()
    assert len({p.parent for p in paths}) == 1
    config = {'paths': {'root': str(paths[0].parent), 'db': str(destination), 'output_dir': str(out)},
              'cluster': {'keeper_pose_policy': 'consensus', 'keeper_pose_displacement_threshold': 0.4},
              'execute': {'dry_run': True}}
    cfg = out / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    stats = stage2_cluster.run(str(cfg))
    c = sqlite3.connect(f'file:{destination}?mode=ro', uri=True)
    keep = {r[0] for r in c.execute('SELECT file_id FROM group_members WHERE is_keep=1')}
    assert keep == set(case['variants']['pose_040']['keepers'])
    assert c.execute("SELECT COUNT(*) FROM group_members WHERE reason='POSE_VARIANT_KEEPER'").fetchone()[0] == 1
    assert c.execute("SELECT COUNT(*) FROM groups WHERE decision_state!='REVIEW_REQUIRED'").fetchone()[0] == 0
    assert stats['auto_remove'] == 0
    c.close()
    proof.append({'audit_id': case['audit_id'], 'native_cached_features': True, 'actual_stage2': True,
                  'keepers': sorted(keep), 'pose_variant_reason_persisted': True,
                  'review_required': True, 'auto_remove': 0, 'photo_modified': False})
(ROOT / 'stage2-positive-regression.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
