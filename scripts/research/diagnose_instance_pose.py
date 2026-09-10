import json
import sqlite3
from pathlib import Path
root=Path('/home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910')
rows=[]
for group in json.loads((root/'slice.json').read_text())['groups']:
 c=sqlite3.connect(f'file:{group["source_db"]}?mode=ro&immutable=1',uri=True)
 for f in group['frames']:
  meta=json.loads(c.execute('SELECT quality_meta FROM features WHERE file_id=?',(f['id'],)).fetchone()[0])
  people=meta.get('pose_evidence',{}).get('people',[])
  rows.append({'day':group['date'],'group':group['group_alias'],'frame':f['alias'],'people':[{'box':p['box'],'confidence':p['box_confidence'],'reliable_joints':sum(x[2]>=.5 for x in p['keypoints'])} for p in people]})
 c.close()
(root/'pose-input-diagnosis.json').write_text(json.dumps(rows,indent=2))
for r in rows:
 print(r['day'],r['group'],r['frame'],[(round(p['confidence'],2),p['reliable_joints']) for p in r['people']])
