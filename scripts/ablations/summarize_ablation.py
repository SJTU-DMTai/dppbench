#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out', required=True); a=p.parse_args()
base=Path(a.out)
rows=[]
for f in sorted(base.glob('*.json')):
    d=json.loads(f.read_text(encoding='utf-8'))
    m=d.get('metrics') or {}
    rows.append({'case':d.get('case'), 'success':d.get('success'), 'validation_auc':d.get('fitness'), 'duration_seconds':d.get('duration_seconds'), 'error':d.get('error'), 'metrics':json.dumps(m,ensure_ascii=False)})
with (base/'summary.csv').open('w', newline='', encoding='utf-8') as out:
    w=csv.DictWriter(out,fieldnames=['case','success','validation_auc','duration_seconds','error','metrics']);w.writeheader();w.writerows(rows)
print('case,success,validation_auc,error')
for r in rows:print(f"{r['case']},{r['success']},{r['validation_auc']},{r['error']}")
print('Saved:',base/'summary.csv')
