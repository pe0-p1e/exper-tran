"""End-of-budget report, validation, compact archive and authorized Git upload."""
import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from PIL import Image
from v7_common import OUT,EXP,REPO,TARGETS,TIDS,sha,atomic_json


def savecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
 p=argparse.ArgumentParser();p.add_argument('--exit-code',type=int,default=0);a=p.parse_args()
 r=OUT/'ten_hour_research';r.mkdir(parents=True,exist_ok=True)
 state=json.loads((r/'state.json').read_text()) if (r/'state.json').exists() else dict(results=[],failures=[],skipped=[])
 budget=json.loads((r/'budget.json').read_text()) if (r/'budget.json').exists() else {}
 shutil.copyfile(EXP/'twenty_hour_plan.yaml',r/'experiment_plan.yaml')
 atomic_json(r/'model_revisions.json',json.loads((OUT/'audits/model_revisions.json').read_text()))
 lock=json.loads((r/'selected_method_lock.json').read_text()) if (r/'selected_method_lock.json').exists() else {}
 diagnostics=[]
 for item in state['results']:
  for path in Path(item['root']).glob('T*/batch_*/state.json'):
   batch=json.loads(path.read_text())
   for restart,trace in enumerate(batch.get('trace',[])):
    if not trace:continue
    cosines=np.array([x['pairwise_cosines'] for x in trace],dtype=float)
    cancellation=np.array([x['cancellation_ratio'] for x in trace],dtype=float)
    diagnostics.append(dict(condition=item['config']['id'],family=item['config'].get('family','unknown'),transition=batch['transition_id'],batch=path.parent.name,restart=restart,images=len(batch['image_ids']),mean_pairwise_gradient_cosine=float(cosines.mean()) if cosines.size else None,negative_pairwise_cosine_fraction=float((cosines<0).mean()) if cosines.size else None,mean_equal_mean_cancellation_ratio=float(cancellation.mean()),initial_proxy_losses=json.dumps(trace[0]['losses']),last_preupdate_proxy_losses=json.dumps(trace[-1]['losses']),initial_proxy_margins=json.dumps(trace[0]['margins']),last_preupdate_proxy_margins=json.dumps(trace[-1]['margins']),runtime_seconds=batch['runtime_seconds'],peak_reserved_gib=batch['peak_reserved_gib']))
 savecsv(r/'gradient_mechanism_diagnostics.csv',diagnostics)
 rows=[];detailed=[];masks={};linf_violations=[];cohort_errors=[]
 for condition in lock.get('allowed_test_conditions',[]):
  for target in TARGETS:
   for tid in TIDS:
    data=[]
    for file in (OUT/'evaluations'/condition/target.replace('/','_')/tid).glob('batch_*/metrics.json'):
     d=json.loads(file.read_text());data+=d['images']
     for image in d['images']:detailed.append(dict(condition=condition,target=target,transition=tid,**image))
    n=len(data);h=sum(x['target_hit'] for x in data);cc=sum(x['clean_correct'] for x in data);hc=sum(x['clean_correct'] and x['target_hit'] for x in data)
    ci=binomtest(h,n).proportion_ci() if n else None
    masks[condition,target,tid]={x['image_id']:bool(x['target_hit']) for x in data}
    rows.append(dict(condition=condition,target=target,transition=tid,hits=h,denominator=n,ASR=h/n if n else None,CI_low=ci.low if ci else None,CI_high=ci.high if ci else None,clean_correct=cc,clean_correct_hits=hc,clean_correct_ASR=hc/cc if cc else None,legacy_conditional_TASR='N/A',parse_success=sum(x['adv_parser']=='ok' for x in data),status='complete' if n==30 else 'missing_or_partial'))
  for tid in TIDS:
   ids=[]
   for f in sorted((OUT/'attacks/test'/condition/tid).glob('batch_*/state.json')):
    d=json.loads(f.read_text());ids+=d['image_ids']
    for c,adv in zip(d['clean_paths'],d['adv_paths']):
     delta=np.asarray(Image.open(OUT/adv)).astype('int16')-np.asarray(Image.open(OUT/c)).astype('int16')
     if abs(delta).max()>16:linf_violations.append(adv)
    for rel,digest in d['png_hashes'].items():
     if sha(OUT/rel)!=digest:cohort_errors.append('Changed PNG '+rel)
    if d['target_access']:cohort_errors.append('Target used in attack '+str(f))
   original=[json.loads(x)['image_id'] for x in (OUT/'manifests/test'/f'{tid}.jsonl').read_text().splitlines() if x.strip()]
   if ids!=original:cohort_errors.append(condition+'/'+tid+' incomplete or mismatched cohort')
 savecsv(r/'final_target_results.csv',rows);savecsv(r/'final_target_per_image.csv',detailed)
 paired=[];rng=np.random.default_rng(42);names=lock.get('allowed_test_conditions',[])
 if names:
  for baseline in names[1:]:
   for target in TARGETS:
    groups=[]
    for tid in TIDS:
     chosen=masks.get((names[0],target,tid),{});base=masks.get((baseline,target,tid),{})
     if set(chosen)!=set(base) or len(base)!=30:continue
     groups.append(np.array([int(chosen[i])-int(base[i]) for i in sorted(base)]))
    if len(groups)==5:
     boot=np.stack([g[rng.integers(0,30,(2000,30))].mean(1) for g in groups]).mean(0)
     paired.append(dict(selected=names[0],baseline=baseline,target=target,paired_N=150,ASR_difference=float(np.concatenate(groups).mean()),CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975))))
 savecsv(r/'paired_final_comparisons.csv',paired)
 all_complete=bool(rows) and all(x['status']=='complete' for x in rows) and not linf_violations and not cohort_errors
 goal=bool(names) and len([x for x in rows if x['condition']==names[0]])==15 and all(x['denominator']==30 and x['hits']>=15 for x in rows if x['condition']==names[0])
 atomic_json(r/'validation.json',dict(all_complete=all_complete,linf_violations=linf_violations,cohort_errors=cohort_errors,all_15_selected_cells_at_least_50pct=goal))
 # Optional post-hoc measurements use only frozen attacks and do not feed selection.
 if all_complete and budget.get('deadline',0)-time.time()>600:
  try:
   subprocess.run([str(REPO/'.venv-a100/bin/python'),str(EXP/'src/analyze_v7.py'),'--method-lock',str(r/'selected_method_lock.json'),'--analysis-dir',str(r/'analysis'),'--results-csv',str(r/'final_target_results.csv')],check=True,timeout=max(1,budget['deadline']-time.time()-60),stdout=(r/'analysis.log').open('a'),stderr=subprocess.STDOUT)
  except Exception as exc:atomic_json(r/'analysis_status.json',dict(status='incomplete',error=str(exc)))
 audits=json.loads((OUT/'audits/model_dtype_audit.json').read_text());latest={}
 for event in audits:latest[event['model_name']]=event
 atomic_json(r/'bf16_audit.json',dict(all_loaded_float_parameters_bf16=all(x['all_floating_parameters_bf16'] for x in audits),load_events=len(audits),latest_full_audit_by_model=latest))
 atomic_json(r/'end_of_budget.json',dict(exit_code=a.exit_code,finished=time.time(),research_conditions=len(state['results']),failed_attempts=len(state['failures']),budget_skipped=state['skipped'],budget=budget,actual_wall_seconds=time.time()-budget.get('started',time.time()),all_final_cells_complete=all_complete))
 with (r/'final_report.md').open('a') as f:
  f.write(f'\n## Final execution validation\n\nProcess exit: {a.exit_code}; research conditions: {len(state["results"])}; failed attempts: {len(state["failures"])}. All final cells complete: {all_complete}. Selected method meets >=15/30 in all 15 cells: {goal}.\n')
  f.write('\n| Condition | Target | Direction | Hits/N | Clean-correct hits/N |\n|---|---|---|---|---|\n')
  for x in rows:f.write(f"| {x['condition']} | {x['target']} | {x['transition']} | {x['hits']}/{x['denominator']} | {x['clean_correct_hits']}/{x['clean_correct']} |\n")
  f.write('\nLegacy conditional TASR is N/A because no all-proxy native-output hit event was measured. Missing measurements are not zero. Nested development confirmation is not an untouched generalization set. The final test cohort is independent and unchanged. Actual wall runtime is in end_of_budget.json; the 20-hour budget is an upper limit, not measured GPU compute. Gradient diagnostics describe pairwise cosine negativity, not pixelwise sign disagreement; cancellation ratio is measured for the unfused equal mean even when a different fusion is used. Trace endpoints are before their respective update.\n')
 # Export a few real success/failure examples, with raw answer metadata.
 gallery=[]
 for wanted in [True,False]:
  row=next((x for x in detailed if x['target_hit']==wanted),None)
  if row:
   for f in (OUT/'attacks/test'/row['condition']/row['transition']).glob('batch_*/state.json'):
    d=json.loads(f.read_text())
    if row['image_id'] not in d['image_ids']:continue
    i=d['image_ids'].index(row['image_id']);dest=r/'examples';dest.mkdir(exist_ok=True)
    for kind,key in [('clean','clean_paths'),('adv','adv_paths')]:shutil.copyfile(OUT/d[key][i],dest/f'{"success" if wanted else "failure"}_{kind}.png')
    gallery.append(row);break
 atomic_json(r/'examples_metadata.json',gallery)
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
 if branch=='experiment/a100-precommercial-v7-targeted-50':
  files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.md','.csv','.json','.yaml')]+list((r/'examples').glob('*.png'))
  rel=[str(p.relative_to(REPO)) for p in files]
  try:
   subprocess.run(['git','add','-f','--',*rel],cwd=REPO,check=True)
   changed=subprocess.run(['git','diff','--cached','--quiet','--',*rel],cwd=REPO).returncode!=0
   if changed:subprocess.run(['git','commit','--only','-m','Archive budgeted V7 experiment results and validation','--',*rel],cwd=REPO,check=True)
   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
   subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:'+branch],cwd=REPO,check=True,timeout=120)
   atomic_json(r/'upload_status.json',dict(status='pushed',commit=commit))
  except Exception as exc:atomic_json(r/'upload_status.json',dict(status='failed',error=str(exc)))
 print('Saved final report:',r/'final_report.md')

if __name__=='__main__':main()
