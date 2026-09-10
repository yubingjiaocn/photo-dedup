"""Freeze a development-only slice from already exposed upstream refusals."""
import json
from pathlib import Path

BASE = Path('/home/ubuntu/photo-dedup-eval')
OUT = BASE / 'astra-instance-recovery-20260910'
OLD = BASE / 'astra-local-quality-20260910'


def main():
    output = OUT / 'slice.json'
    if output.exists():
        raise ValueError('Slice already frozen; inspect instead of replacing')
    diagnosis = json.loads((BASE / 'astra-regime-validation-20260910/ACTOR-DIAGNOSIS.json').read_text())
    selected = [g for g in diagnosis['missed_groups'] if 'SET_COVERAGE_UNSTABLE' in g['refusals']][:8]
    groups = []
    for row in selected:
        day = row['day']
        origin = OLD if day.startswith('2026-07-2') else BASE / 'astra-real-events-20260910'
        mapping = origin / f'panels/{day}/mapping.json'
        g = next(g for g in json.loads(mapping.read_text()) if g['group_alias'] == row['group_alias'])
        groups.append({**g, 'split': 'exposed_development', 'source_db': str(BASE / f'astra-regime-validation-20260910/development/{day}/inventory.sqlite'), 'config': str(origin / f'cache/{day}/config.yaml'), 'source_mapping': str(mapping)})
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'selection': 'first eight exposed missed-phase groups with SET_COVERAGE_UNSTABLE in frozen diagnosis order; not representative holdout', 'groups': groups}, indent=2))
    print([(g['date'], g['group_alias'], len(g['frames'])) for g in groups])


if __name__ == '__main__':
    main()
