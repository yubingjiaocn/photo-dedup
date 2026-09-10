"""Bounded evidence inspection, never reruns inference or changes observations."""
import json
from pathlib import Path
root=Path('/home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910/foreground-pilot')
for path in root.glob('*/result.json'):
 r=json.loads(path.read_text())
 print(r['group_alias'],'frames',[(x['frame']['alias'],len(x['observations']),sum(o['view_confirmed'] for o in x['observations'])) for x in r['records']])
 for f,rec in enumerate(r['records']):
  print(' F',f,[(i,o['class_id'],o['view_confirmed'],o['new_vs_v0_catalog']) for i,o in enumerate(rec['observations'])])
 for link in sorted(r['links'],key=lambda l:l.get('similarity',0),reverse=True)[:12]:
  print({k:v for k,v in link.items() if k!='support'}, {k:v for k,v in link.get('support',{}).items() if 'inliers_native' not in k})
