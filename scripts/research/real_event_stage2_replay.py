"""Run actual Stage2/Stage3 on a NEW DB copy; compare only AFTER selection."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/ubuntu/photo-dedup-eval/astra-real-events-20260910')
sys.path.insert(0, str(REPO))
from src import db, stage2_cluster, stage3_report  # noqa: E402

KEYS = {'score_policy': 'keeper_score_policy', 'score_change_margin': 'keeper_score_change_margin',
        'diversity_policy': 'keeper_diversity_policy', 'diversity_quality_slack': 'keeper_diversity_quality_slack',
        'pose_policy': 'keeper_pose_policy', 'pose_displacement_threshold': 'keeper_pose_displacement_threshold'}


def replay(day, source_db, variant, ab_file, report=False, verify_only=False):
    out = ROOT / 'stage2-replay' / day / variant
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'inventory.sqlite'
    config_file = out / 'config.yaml'
    params = json.loads((ROOT / 'frozen-runtime-variants.json').read_text())[variant]
    if verify_only:
        if not destination.is_file():
            raise ValueError('No completed runtime DB to verify')
        config = yaml.safe_load(config_file.read_text())
        assert all(config['cluster'][KEYS[k]] == v for k, v in params.items())
        stats, rendered = None, None
    else:
        if destination.exists():
            raise ValueError('Replay DB exists; use --verify-only after inspecting completed stages')
        source = sqlite3.connect(f'file:{source_db}?mode=ro&immutable=1', uri=True)
        target = sqlite3.connect(destination)
        source.backup(target)
        source.close()
        target.close()
        config = yaml.safe_load((ROOT / 'cache' / day / 'config.yaml').read_text())
        config['cluster'].update({KEYS[key]: value for key, value in params.items()})
        config['paths'].update(db=str(destination), output_dir=str(out), trash=str(out / 'unused-trash'))
        config['execute']['dry_run'] = True
        config_file.write_text(yaml.safe_dump(config))
        stats = stage2_cluster.run(str(config_file))
        rendered = stage3_report.run(str(config_file)) if report else None
    # Evaluation artifacts cannot influence runtime: opened only after both stages.
    reference = json.loads(ab_file.read_text())
    mapping = json.loads((ROOT / 'panels' / day / 'mapping.json').read_text())
    expected = {g['group_alias']: g for g in reference['rows']}
    conn = sqlite3.connect(f'file:{destination}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    if stats is None:
        stats = json.loads(db.get_meta(conn, 'stage2_stats', '{}'))
    if report:
        assert (out / 'review.html').is_file()
    actual = {}
    for group in conn.execute('SELECT id,group_type,decision_state FROM groups'):
        members = list(conn.execute('SELECT file_id,is_keep,decision FROM group_members WHERE group_id=?', (group['id'],)))
        actual[frozenset(m['file_id'] for m in members)] = (group, members)
    assert sum(g['group_type'] != 'sha_exact' for g, _ in actual.values()) == len(mapping)
    for group in mapping:
        alias_by_id = {f['id']: f['alias'] for f in group['frames']}
        stored, members = actual[frozenset(alias_by_id)]
        assert stored['group_type'] == group['group_type']
        assert stored['decision_state'] == 'REVIEW_REQUIRED'
        keepers = {alias_by_id[m['file_id']] for m in members if m['is_keep']}
        assert keepers == set(expected[group['group_alias']]['variants'][variant]['keepers']), group['group_alias']
    unsafe = conn.execute("SELECT COUNT(*) FROM group_members gm JOIN groups g ON gm.group_id=g.id JOIN features f ON gm.file_id=f.file_id JOIN features k ON g.keep_file_id=k.file_id WHERE gm.decision='AUTO_REMOVE' AND (g.group_type!='sha_exact' OR f.content_sha256 IS NULL OR f.content_sha256!=k.content_sha256)").fetchone()[0]
    assert unsafe == 0
    conn.close()
    result = {'day': day, 'variant': variant, 'verified_multiframe_groups': len(mapping),
              'keeper_sets_match_evaluator': True, 'physical_group_membership_unchanged': True,
              'visual_groups_review_required': True, 'unsafe_auto_remove': unsafe,
              'runtime_label_input': False, 'stage4_executed': False, 'config': str(config_file),
              'stats': stats, 'report': rendered}
    (out / 'replay-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--day', required=True)
    p.add_argument('--source-db', type=Path, required=True)
    p.add_argument('--variant', choices=['baseline', 'pose_040', 'coverage_balanced'], required=True)
    p.add_argument('--ab', type=Path, required=True)
    p.add_argument('--report', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    replay(a.day, a.source_db, a.variant, a.ab, a.report, a.verify_only)
