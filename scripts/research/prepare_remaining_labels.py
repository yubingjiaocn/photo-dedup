"""Reuse frozen development labels by complete ordered source identity only."""
import json
from pathlib import Path

ROOT = Path('/home/ubuntu/photo-dedup-eval/astra-real-events-20260910')
old_mapping = json.loads((ROOT / 'panels/development-mid/mapping.json').read_text())
new_mapping = json.loads((ROOT / 'panels/2026-07-31/mapping.json').read_text())
labels = {}
for path in (ROOT / 'labels/development-mid').glob('*.json'):
    for group in json.loads(path.read_text())['groups']:
        labels[group['group_alias']] = group


def identity(group):
    return tuple(Path(frame['path']).name for frame in group['frames'])


old = {identity(group): labels[group['group_alias']] for group in old_mapping}
reused, unseen = [], []
for group in new_mapping:
    key = identity(group)
    if key in old:
        record = dict(old[key])
        record['group_alias'] = group['group_alias']
        reused.append(record)
    else:
        unseen.append(group['group_alias'])
out = ROOT / 'labels/development-full'
out.mkdir(parents=True, exist_ok=True)
path = out / 'labels-reused.json'
if path.exists():
    raise ValueError('Frozen reuse output already exists')
path.write_text(json.dumps({'status': 'model_provisional', 'actual_model': 'litellm-openai/gpt-6-astra',
                            'provenance': 'frozen development-mid labels; ordered complete source-identity join; not independent/new blind evidence',
                            'groups': reused}, ensure_ascii=False, indent=2))
packets = []
for path in sorted((ROOT / 'panels/2026-07-31').glob('blind-*.json')):
    packets.extend(g for g in json.loads(path.read_text())['groups'] if g['group_alias'] in unseen)
packet_dir = ROOT / 'new-development-label-packets'
packet_dir.mkdir(exist_ok=True)
for start in range(0, len(packets), 10):
    path = packet_dir / f'blind-{start // 10 + 1:02d}.json'
    if path.exists():
        raise ValueError('Blind packet already exists')
    path.write_text(json.dumps({'groups': packets[start:start + 10]}, indent=2))
summary = {'reused_groups': len(reused), 'new_groups': len(packets),
           'new_photos': sum(len(g['frames']) for g in packets), 'aliases': unseen}
(packet_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
