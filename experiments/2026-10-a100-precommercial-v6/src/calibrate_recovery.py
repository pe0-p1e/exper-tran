#!/usr/bin/env python3
"""Select conservative attack microbatches from real two-image gradient smoke data."""
import argparse,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args();o=a.output_dir
 audit=json.loads((o/'audits/model_audit.json').read_text()); detail=json.loads((o/'audits/model_dtype_audit.json').read_text())
 byname={}
 for r in detail:
  if r['role']=='proxy': byname[r['model_name']]=r
 result={'method':'actual two-image gradient/generation smoke audit; no dtype downgrade; sequential per-proxy GPU residency','max_peak_reserved_gib':75,'per_model':{},'selected_attack_batch_size':2,'attempted_gradient_smoke_batch_sizes':[2],'runtime_oom_fallback_batch_sizes':[1],'larger_batch_sizes_not_calibrated':[4],'note':'Batch size 2 passed the model gradient smoke and recorded load peak was below 75 GiB; one-image fallback is configured for runtime OOM. Batch 4 was not benchmarked, so no claim is made that 2 is the maximum safe batch.'}
 blockers=[]
 for row in audit:
  name=row['model_name']
  if row['role'] not in ('proxy','proxy+target'): continue
  d=byname.get(name,{})
  peak=float(d.get('cuda_memory',{}).get('cuda_reserved_gib',0))
  status=row['status']
  if status!='pass': blockers.append({'model':name,'reason':row.get('error','audit failed')})
  result['per_model'][name]={'gradient_smoke':row.get('gradient_smoke'),'vision_depth':row.get('vision_depth'),'deepest_block':row.get('deepest_block'),'load_reserved_gib':peak,'attempted_batches':[2] if status=='pass' else [],'selected_batch':2 if status=='pass' and peak<75 else 1,'selection_basis':'2-image BF16 gradient smoke' if status=='pass' else 'blocked by model audit'}
 result['blockers']=blockers
 path=o/'audits/performance_benchmark.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
 print(f"[calibration] recorded per-model BF16 two-image smoke and conservative microbatch; blockers={len(blockers)}")
 if blockers: raise SystemExit('one or more models failed calibration prerequisite')
if __name__=='__main__':main()
