#!/usr/bin/env python3
"""Return success only for a complete, revision-matched, strict BF16 audit."""
import argparse,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args()
 lock=json.loads((ROOT/'model_revisions.json').read_text());m=yaml.safe_load((ROOT/'experiment_matrix.yaml').read_text())['multiple_proxy'];proxies=set();targets=set()
 for b in ('block_1','block_2'):
  for r in m[b]: targets.add(r['target']); [proxies.update(x) for x in r['ensembles']]
 summary=json.loads((a.output_dir/'audits/model_audit.json').read_text()); detailed=json.loads((a.output_dir/'audits/model_dtype_audit.json').read_text())
 good={x['model_name'] for x in summary if x.get('status')=='pass' and x.get('revision')==lock[x['model_name']]['revision']}
 need=proxies|targets
 if not need<=good: raise SystemExit('model audit incomplete: '+repr(sorted(need-good)))
 dgood=set()
 for x in detailed:
  if x.get('revision')!=lock[x.get('model_name','')]['revision'] or x.get('status')!='pass' or not x.get('all_floating_parameters_bf16') or not x.get('parameter_counts_by_dtype') or set(x['parameter_counts_by_dtype'])!={'bfloat16'} or x['quantization'].get('quantized_modules') or x['quantization'].get('is_quantized'):
   continue
  dgood.add(x['model_name'])
 if not need<=dgood: raise SystemExit('strict detailed BF16 audit incomplete: '+repr(sorted(need-dgood)))
 print(f'PASS {len(need)} pinned models have strict BF16, no-quantization audits')
if __name__=='__main__':main()
