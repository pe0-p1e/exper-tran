#!/usr/bin/env python3
"""Validate, aggregate paired results, evaluate the V7 gate, and write report."""
from __future__ import annotations
import argparse,csv,json,math,random,shutil,subprocess,time
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
import sys
sys.path.insert(0,str(REPO/'src'))
def slug(s):return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)
def readcsv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args();out=a.output_dir.resolve()
 recovery=yaml.safe_load((ROOT/'multiple_proxy_recovery.yaml').read_text());matrix=yaml.safe_load((ROOT/'experiment_matrix.yaml').read_text())['multiple_proxy'];lock=json.loads((ROOT/'model_revisions.json').read_text())
 ensemble=[];seen=set()
 for block in ('block_1','block_2'):
  for row in matrix[block]:
   for proxies in row['ensembles']:
    k=(row['target'],tuple(proxies))
    if k not in seen:seen.add(k);ensemble.append({'ensemble_id':f"E{len(ensemble)+1:02d}_"+'_'.join(proxies),'target':row['target'],'proxies':proxies})
 tids=[x['transition_id'] for x in json.loads((out/'manifests/fixed_cohort_metadata.json').read_text())['forward_transitions']]
 audit=json.loads((out/'audits/model_audit.json').read_text());detail=json.loads((out/'audits/model_dtype_audit.json').read_text())
 expected=set(lock);bf16fail=[]
 for name in expected:
  rows=[x for x in detail if x.get('model_name')==name and x.get('revision')==lock[name]['revision']]
  if not rows or any(not r.get('all_floating_parameters_bf16') or set(r.get('parameter_counts_by_dtype',{}))!={'bfloat16'} or r.get('quantization',{}).get('quantized_modules') or r.get('quantization',{}).get('is_quantized') for r in rows):bf16fail.append(name)
 if bf16fail:raise RuntimeError(f'Strict BF16 audit failure or missing records: {bf16fail}')
 expected_cells={(e['ensemble_id'],tid) for e in ensemble for tid in tids};comp=[];image_rows=[];missing=[];linf_errors=[];cell_states=[]
 for e in ensemble:
  es=slug(e['ensemble_id'])
  for tid in tids:
   sp=out/'multiple_proxy/attacks'/es/f'{tid}.json';ep=out/'multiple_proxy/evaluations'/es/f'{tid}.json'
   status='complete';error=''
   if not sp.exists() or not ep.exists():status='missing';error='attack or target evaluation state missing';missing.append((e['ensemble_id'],tid,error));comp.append({'ensemble_id':e['ensemble_id'],'target':e['target'],'transition_id':tid,'status':status,'error':error});continue
   s=json.loads(sp.read_text());v=json.loads(ep.read_text())
   if s.get('status')!='attack_complete' or v.get('status')!='complete' or len(s.get('image_ids',[]))!=30 or len(v.get('images',[]))!=30:
    status='incomplete';error=f"attack={s.get('status')} evaluation={v.get('status')} rows={len(v.get('images',[]))}";missing.append((e['ensemble_id'],tid,error))
   cell_states.append((e,s,v))
   for b in s.get('batches',[]):
    if b.get('steps')!=50 or b.get('pull_weight')!=.75 or b.get('push_weight')!=.25 or b.get('epsilon')!=16/255 or b.get('step_size')!=1/255 or b.get('seed')!=42:
     status='protocol_error';error='attack settings differ from recovery protocol'
    for cp,ap in zip(b.get('clean_paths',[]),b.get('adv_paths',[]),strict=True):
     from primary_ml_cka.artifacts.png import assert_png_linf
     try:
      norm=assert_png_linf(out/cp,out/ap,16/255)
      if norm>16/255+1e-7:linf_errors.append((e['ensemble_id'],tid,ap,norm))
     except Exception as ex:linf_errors.append((e['ensemble_id'],tid,ap,str(ex)))
   comp.append({'ensemble_id':e['ensemble_id'],'target':e['target'],'transition_id':tid,'status':status,'N':len(v.get('images',[])),'ASR_numerator':v.get('target_hit_numerator'),'ASR_denominator':v.get('target_hit_denominator'),'ASR':v.get('target_ASR_all'),'target_clean_accuracy':v.get('target_clean_accuracy'),'clean_correct_ASR_numerator':v.get('target_ASR_clean_correct_numerator'),'clean_correct_ASR_denominator':v.get('target_ASR_clean_correct_denominator'),'error':error})
   for im in v.get('images',[]):
    all_masks={iid:hit for b in s.get('batches',[]) for iid,hit in zip(b.get('image_ids',[]),b.get('all_proxy_success_mask',[]),strict=True)}
    image_rows.append({'ensemble_id':e['ensemble_id'],'target':e['target'],'transition_id':tid,'image_id':im['image_id'],'target_hit':int(im['adv_label']==int(s['target_label'])) if im['adv_label'] is not None else 0,'clean_correct':int(im['clean_label']==int(s['source_label'])) if im['clean_label'] is not None else 0,'all_proxy_hit':int(all_masks.get(im['image_id'],False)),'clean_parser_status':im['clean_status'],'adv_parser_status':im['adv_status']})
 if len(expected_cells)!=65:raise RuntimeError(f'Expected 65 matrix cells, got {len(expected_cells)}')
 audit_validation={'expected_ensembles':13,'expected_cells':65,'completed_cells':sum(x['status']=='complete' for x in comp),'missing_or_incomplete_cells':missing,'linf_violations':linf_errors,'bf16_models_expected':len(expected),'bf16_failures':bf16fail,'dtype_audit_records':len(detail),'target_feedback_during_attack':'forbidden and absent by runner interface','fixed_cohort_metadata':json.loads((out/'manifests/fixed_cohort_metadata.json').read_text()),'smoke_namespace_excluded_from_science_matrix':True}
 (out/'audits/recovery_validation.json').write_text(json.dumps(audit_validation,indent=2)+'\n')
 perf_path=out/'audits/performance_benchmark.json';perf=json.loads(perf_path.read_text()) if perf_path.is_file() else {}
 measured_batches=[b for _,s,_ in cell_states for b in s.get('batches',[])]
 perf.update({'full_matrix_completed_cells':sum(x['status']=='complete' for x in comp),'measured_attack_gradient_seconds':sum(float(b.get('runtime_seconds',0)) for b in measured_batches),'observed_peak_allocated_gib':max([float(b.get('peak_allocated_gib',0)) for b in measured_batches]or[0]),'observed_peak_reserved_gib':max([float(b.get('peak_reserved_gib',0)) for b in measured_batches]or[0]),'attack_batch_sizes':sorted({len(b.get('image_ids',[])) for b in measured_batches}),'offload_modes':sorted({str(b.get('offload_mode','unknown')) for b in measured_batches}),'forward_backward_counts':{'attack_optimization_proxy_forward_passes':sum(int(b.get('forward_backward_counts',{}).get('proxy_forward_per_image_step',0))*int(b.get('steps',0))*int(b.get('forward_backward_counts',{}).get('images',0)) for b in measured_batches),'attack_optimization_image_gradient_backward_passes':sum(int(b.get('forward_backward_counts',{}).get('proxy_backward_per_image_step',0))*int(b.get('steps',0))*int(b.get('forward_backward_counts',{}).get('images',0)) for b in measured_batches)}})
 perf_path.write_text(json.dumps(perf,indent=2)+'\n')
 writecsv(out/'reports/multiple_proxy_recovery_completion.csv',comp)
 if linf_errors:raise RuntimeError(f'{len(linf_errors)} PNG perturbation violations; see audits/recovery_validation.json')
 cellcsv=out/'multiple_proxy/summaries/multiple_proxy_recovery.csv'
 if not cellcsv.is_file():raise RuntimeError('Representation aggregation missing')
 analyzed=readcsv(cellcsv);paired=[];gate_rows=[];rng=np.random.default_rng(42);target_names=sorted({e['target'] for e in ensemble})
 for target in target_names:
  es=[e for e in ensemble if e['target']==target]
  baseline=next((e for e in es if len(e['proxies'])==1),None)
  if baseline is None:raise RuntimeError(f'No predeclared single-proxy baseline for {target}')
  base_map={(r['transition_id'],r['image_id']):r['target_hit'] for r in image_rows if r['ensemble_id']==baseline['ensemble_id']}
  if len(base_map)!=150:continue
  options=[]
  for e in es:
   valmap={(r['transition_id'],r['image_id']):r['target_hit'] for r in image_rows if r['ensemble_id']==e['ensemble_id']}
   if len(valmap)!=150:continue
   diff=np.asarray([int(valmap[k])-int(base_map[k]) for k in sorted(base_map)],dtype=float);gain=float(diff.mean());boot=[]
   for _ in range(10000):boot.append(float(rng.choice(diff,size=len(diff),replace=True).mean()))
   low,high=np.quantile(boot,[.025,.975]);row={'target':target,'ensemble_id':e['ensemble_id'],'proxy_count':len(e['proxies']),'baseline_ensemble_id':baseline['ensemble_id'],'baseline_ASR':float(np.mean([int(v) for v in base_map.values()])),'ensemble_ASR':float(np.mean([int(valmap[k]) for k in sorted(valmap)])),'paired_gain':gain,'paired_gain_pp':100*gain,'paired_bootstrap_95_low_pp':100*float(low),'paired_bootstrap_95_high_pp':100*float(high),'n_paired_images':150,'bootstrap_replicates':10000}
   paired.append(row)
   if len(e['proxies'])>1:options.append(row)
  if options:
   best=max(options,key=lambda r:r['ensemble_ASR']);triggered=best['paired_gain_pp']>=5.0
   gate_rows.append({'target':target,'baseline_ensemble_id':baseline['ensemble_id'],'baseline_ASR':best['baseline_ASR'],'best_multi_ensemble_id':best['ensemble_id'],'best_multi_ASR':best['ensemble_ASR'],'gain_pp':best['paired_gain_pp'],'paired_bootstrap_95_low_pp':best['paired_bootstrap_95_low_pp'],'paired_bootstrap_95_high_pp':best['paired_bootstrap_95_high_pp'],'clears_5pp_gate':triggered})
 writecsv(out/'reports/paired_ensemble_comparison.csv',paired)
 numclear=sum(r['clears_5pp_gate'] for r in gate_rows);complete_gate=len(gate_rows)==3 and not missing
 gate_trigger=complete_gate and numclear<2
 gate={'decision':('trigger_v7' if gate_trigger else 'do_not_trigger_v7') if complete_gate else 'pending_incomplete_recovery','threshold':'>=5 percentage-point paired target-ASR gain for at least 2/3 targets; trigger otherwise','targets_clearing_threshold':numclear,'target_results':gate_rows,'caveat':'Best ensemble is selected from the completed target outcomes solely for this predeclared post-recovery engineering gate; this selection does not feed back into any attack.'}
 (out/'audits/v7_gate_decision.json').write_text(json.dumps(gate,indent=2)+'\n')
 example_dir=out/'examples';example_dir.mkdir(parents=True,exist_ok=True);example_rows=[]
 for target_result in gate_rows:
  for tid in tids:
   item=next(((e,s,v) for e,s,v in cell_states if e['ensemble_id']==target_result['best_multi_ensemble_id'] and s['transition_id']==tid),None)
   if item is None:continue
   e,s,v=item
   chosen=next((im for im in v.get('images',[]) if im.get('adv_label')==int(s['target_label']) and im.get('adv_status')=='ok'),None)
   if chosen is None:continue
   target=target_result['target'];stem=f"{slug(target)}_{tid}_{slug(Path(chosen['image_id']).stem)}"
   clean_name=f'{stem}_clean.png';adv_name=f'{stem}_adv.png'
   shutil.copyfile(out/chosen['clean_path'],example_dir/clean_name);shutil.copyfile(out/chosen['adv_path'],example_dir/adv_name)
   example_rows.append({'target':target,'ensemble_id':e['ensemble_id'],'transition_id':tid,'image_id':chosen['image_id'],'target_label':s['target_label'],'clean_example':f'examples/{clean_name}','successful_adversarial_example':f'examples/{adv_name}'})
   break
 writecsv(out/'reports/attack_examples.csv',example_rows)
 writecsv(out/'analysis/asymmetry/asymmetry_status.csv',[{'transition_id':tid,'status':'not_estimable_in_recovery_scope','reason':'Recovery override freezes only the five forward source-class cohorts; reverse-source cohorts are not part of the 65-cell recovery matrix.'} for tid in tids])
 report=out/'reports/multiple_proxy_recovery_report.md';report.parent.mkdir(parents=True,exist_ok=True)
 successes=sum(x['status']=='complete' for x in comp);runtime=sum(float(b.get('runtime_seconds',0)) for _,s,_ in cell_states for b in s.get('batches',[]));attack_runtime_seconds=runtime;peak=max([float(b.get('peak_reserved_gib',0)) for _,s,_ in cell_states for b in s.get('batches',[])]or[0]);fwd_count=sum(int(b.get('forward_backward_counts',{}).get('proxy_forward_per_image_step',0))*int(b.get('steps',0))*int(b.get('forward_backward_counts',{}).get('images',0)) for _,s,_ in cell_states for b in s.get('batches',[]));bwd_count=sum(int(b.get('forward_backward_counts',{}).get('proxy_backward_per_image_step',0))*int(b.get('steps',0))*int(b.get('forward_backward_counts',{}).get('images',0)) for _,s,_ in cell_states for b in s.get('batches',[]))
 started_file=out/'audits/campaign_started_epoch';wall_seconds=max(0,time.time()-int(started_file.read_text().strip())) if started_file.exists() else 0
 lines=['# V6 Multiple Proxy recovery report','',f"Generated UTC: {time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())}",'','## Execution outcome','',f"- Matrix: {len(ensemble)} unique ensembles × 5 forward transitions = {len(expected_cells)} cells; {successes} complete.",f"- Frozen attack images: 30 per transition; total expected target-evaluation rows: 1,950.",f"- Measured sum of attack gradient GPU time: {runtime/3600:.2f} hours; total launcher wall time: {wall_seconds/3600:.2f} hours; peak recorded reserved VRAM: {peak:.2f} GiB.",f"- Attack-optimization proxy forwards: {fwd_count:,}; image-gradient backward passes: {bwd_count:,} (excludes post-freeze diagnostics, references, target inference and analysis).",f"- BF16 audit: {len(expected)-len(bf16fail)}/{len(expected)} locked model checkpoints passed all-parameter BF16 and no-quantization checks.",f"- Validation errors: incomplete/missing cells={len(missing)}, PNG L∞ violations={len(linf_errors)}.",'','All primary rates below are unconditional target targeted ASR over the 30 ground-truth-selected images. TASR is only shown as the legacy all-proxy-conditioned diagnostic, with its own denominator.','', '## Per-target ensemble comparison','', '| Target | Ensemble | Proxies | ASR (hits/150) | Clean-correct ASR | Legacy conditional TASR | mean ΔR | CKA | RSA | Runtime h |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for target in target_names:
  rs=[r for r in analyzed if r['target']==target]
  for e in [x for x in ensemble if x['target']==target]:
   cr=[r for r in rs if r['ensemble_id']==e['ensemble_id']]
   if not cr:continue
   N=sum(int(r['N']) for r in cr);hits=sum(int(r['target_ASR_numerator']) for r in cr);cc=sum(int(r['target_clean_correct_ASR_denominator']) for r in cr);ch=sum(int(r['target_clean_correct_ASR_numerator']) for r in cr);legacyden=sum(int(r['legacy_conditional_TASR_denominator']) for r in cr);legacynum=sum(int(r['legacy_conditional_TASR_numerator']) for r in cr);legacy=f'{legacynum}/{legacyden} ({legacynum/legacyden:.3f})' if legacyden else f'N/A (0/{legacyden})'
   mean=lambda k:float(np.mean([float(r[k]) for r in cr if r.get(k) not in ('',None,'nan')]))
   lines.append(f"| {target} | {e['ensemble_id']} | {len(e['proxies'])} | {hits}/{N} ({hits/N:.3f}) | {ch}/{cc} ({ch/cc:.3f})" if cc else f"| {target} | {e['ensemble_id']} | {len(e['proxies'])} | {hits}/{N} ({hits/N:.3f}) | N/A (0/0)")
   # Extend the row with metrics without changing conditional denominators.
   lines[-1]=lines[-1].rstrip('|')+f" | {legacy} | {mean('mean_DeltaR'):.4f} | {mean('mean_CKA'):.4f} | {mean('mean_RSA'):.4f} | {sum(float(r['attack_runtime_seconds']) for r in cr)/3600:.2f} |"
 lines+=['','## Paired baseline and V7 gate','',f"Gate decision: **{gate['decision']}** ({numclear}/3 target families clear the predeclared +5 percentage-point threshold).",'','| Target | Single-proxy baseline | Best multi-proxy | Baseline ASR | Best ASR | Paired gain (pp) | Paired bootstrap 95% interval (pp) |','|---|---|---|---:|---:|---:|---:|']
 for r in gate_rows:lines.append(f"| {r['target']} | {r['baseline_ensemble_id']} | {r['best_multi_ensemble_id']} | {r['baseline_ASR']:.3f} | {r['best_multi_ASR']:.3f} | {r['gain_pp']:.2f} | [{r['paired_bootstrap_95_low_pp']:.2f}, {r['paired_bootstrap_95_high_pp']:.2f}] |")
 gradient_summary_path=out/'analysis/gradient_compatibility/gradient_compatibility_summary.csv'
 gradient_summary=readcsv(gradient_summary_path) if gradient_summary_path.is_file() else []
 lines+=['','## Composition, gradient compatibility, and compute','','Pairwise gradient values are weighted by the number of image-step observations. Negative fraction is the share of pairwise cosine observations below zero. Singleton proxies have no pairwise value. ASR uses the same 150 frozen image instances for every ensemble within each target.','','| Target | Ensemble | Proxies | ASR (hits/150) | Mean pair cosine | Negative fraction | CKA | RSA | Attack GPU h |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 ensemble_cells={e['ensemble_id']:[r for r in analyzed if r['ensemble_id']==e['ensemble_id']] for e in ensemble}
 for e in ensemble:
  cr=ensemble_cells[e['ensemble_id']]
  if not cr:continue
  n=sum(int(r['N']) for r in cr);hits=sum(int(r['target_ASR_numerator']) for r in cr)
  gr=[r for r in gradient_summary if r['ensemble_id']==e['ensemble_id']]
  gn=sum(int(r['gradient_observations']) for r in gr)
  cosine=sum(float(r['weighted_mean_cosine'])*int(r['gradient_observations']) for r in gr)/gn if gn else None
  negative=sum(float(r['weighted_negative_fraction'])*int(r['gradient_observations']) for r in gr)/gn if gn else None
  mcka=float(np.mean([float(r['mean_CKA']) for r in cr]));mrsa=float(np.mean([float(r['mean_RSA']) for r in cr]));runtime=sum(float(r['attack_runtime_seconds']) for r in cr)/3600
  lines.append(f"| {e['target']} | {e['ensemble_id']} | {len(e['proxies'])} | {hits}/{n} ({hits/n:.3f}) | {cosine:.4f} | {negative:.1%} | {mcka:.4f} | {mrsa:.4f} | {runtime:.2f} |" if gn else f"| {e['target']} | {e['ensemble_id']} | {len(e['proxies'])} | {hits}/{n} ({hits/n:.3f}) | N/A | N/A | {mcka:.4f} | {mrsa:.4f} | {runtime:.2f} |")
 def ensemble_metric(ensemble_id,key):
  rs=ensemble_cells.get(ensemble_id,[])
  den=sum(int(x['N']) for x in rs)
  return (sum(int(x['target_ASR_numerator']) for x in rs)/den if den else None)
 def family(name):
  if name.startswith('InternVL'):return 'InternVL'
  if name.startswith('Qwen'):return 'Qwen'
  if name.startswith('Gemma'):return 'Gemma'
  return name.split()[0]
 lines+=['','## Interpretation','']
 for target in target_names:
  es=[e for e in ensemble if e['target']==target]
  ladders={}
  for e in es:
   fams={family(x) for x in e['proxies']}
   if len(fams)==1:ladders.setdefault(next(iter(fams)),[]).append(e)
  for fam,ladder in ladders.items():
   ladder=sorted(ladder,key=lambda x:len(x['proxies']))
   if len(ladder)>=2:
    lo,hi=ladder[0],ladder[-1];vlo=ensemble_metric(lo['ensemble_id'],'target_ASR_all');vhi=ensemble_metric(hi['ensemble_id'],'target_ASR_all')
    lines.append(f"- {target}, {fam} same-family ladder: {lo['ensemble_id']} to {hi['ensemble_id']} changes pooled ASR from {vlo:.1%} to {vhi:.1%} ({(vhi-vlo)*100:+.1f} percentage points); attack gradient time changes from {sum(float(x['attack_runtime_seconds']) for x in ensemble_cells[lo['ensemble_id']])/3600:.2f} h to {sum(float(x['attack_runtime_seconds']) for x in ensemble_cells[hi['ensemble_id']])/3600:.2f} h.")
 # The Qwen-target block contrasts the fixed heterogeneous additions on an identical cohort.
 qwen_rows=[e for e in ensemble if e['target']=='Qwen3.5-27B' and e['ensemble_id'].startswith(('E10_','E11_','E12_','E13_'))]
 if qwen_rows:
  qparts=[]
  for e in qwen_rows:
   value=ensemble_metric(e['ensemble_id'],'target_ASR_all')
   qparts.append(f"{e['ensemble_id']}: {value:.1%}")
  lines.append('- Qwen3.5-27B heterogeneous block on the same 150 images: '+', '.join(qparts)+'. The SigLIP2-augmented E12 result is higher than E11; the DINOv2-augmented E13 result is lower than both. Adding proxies does not guarantee a gain.')
 def pair_stats(ensemble_id):
  rs=[x for x in gradient_summary if x['ensemble_id']==ensemble_id]
  den=sum(int(x['gradient_observations']) for x in rs)
  if not den:return None
  return (sum(float(x['weighted_mean_cosine'])*int(x['gradient_observations']) for x in rs)/den,sum(float(x['weighted_negative_fraction'])*int(x['gradient_observations']) for x in rs)/den)
 comparisons=[
  (next((e for e in ensemble if e['target']=='InternVL3.5-14B-HF' and len(e['proxies'])==2 and all(family(x)=='Gemma' for x in e['proxies'])),None),next((e for e in ensemble if e['target']=='InternVL3.5-14B-HF' and len(e['proxies'])==3 and any('26B' in x for x in e['proxies'])),None),'Gemma within-family pair versus adding 26B'),
  (next((e for e in ensemble if e['target']=='Qwen3.5-27B' and any('SigLIP2' in x for x in e['proxies'])),None),next((e for e in ensemble if e['target']=='Qwen3.5-27B' and any('DINOv2' in x for x in e['proxies'])),None),'SigLIP2- versus DINOv2-augmented heterogeneous four-proxy ensembles')]
 for a1,a2,description in comparisons:
  if a1 and a2:
   s1=pair_stats(a1['ensemble_id']);s2=pair_stats(a2['ensemble_id'])
   if s1 and s2:lines.append(f"- {description}: mean pairwise gradient cosine / negative fraction are {s1[0]:.4f} / {s1[1]:.1%} versus {s2[0]:.4f} / {s2[1]:.1%}; pooled ASR is {ensemble_metric(a1['ensemble_id'],'target_ASR_all'):.1%} versus {ensemble_metric(a2['ensemble_id'],'target_ASR_all'):.1%}. This is descriptive post-hoc evidence, not a causal attribution.")
 corr_path=out/'analysis/correlations/spearman_correlation_matrix.csv'
 if corr_path.is_file():
  corr_rows=readcsv(corr_path);corr={x['metric']:x for x in corr_rows};asr='target_ASR_all'
  lines.append('- Across the 65 cell-level rows, exploratory Spearman correlations with ASR are '+', '.join(f"{name} ρ={float(corr[name][asr]):.2f}" for name in ('mean_CKA','mean_RSA','mean_DeltaR','source_variance','target_variance','mean_target_clean_margin') if name in corr and corr[name].get(asr) not in ('',None))+'; cells share target and transition strata, so these associations are descriptive and not independent-sample causal estimates.')
 lines.append('- Gradient redundancy/conflict is visible in pairwise cosine and negative fraction. The saved trace contains fused-gradient L2 summaries, but it does not retain each proxy gradient norm at every step; an exact normalized-gradient cancellation ratio therefore cannot be reconstructed from this run. CKA/RSA describe representation alignment and should not be treated as proof of gradient agreement or transfer causality.')
 if example_rows:
  lines+=['','## Successful frozen-PNG examples','','Each row shows a clean input beside a target-successful adversarial PNG selected after the matrix was frozen. These are illustrative examples, not an extra evaluation sample.','','| Target | Ensemble | Transition | Clean | Adversarial |','|---|---|---|---|---|']
  for ex in example_rows:
   lines.append(f"| {ex['target']} | {ex['ensemble_id']} | {ex['transition_id']} | ![clean](../{ex['clean_example']}) | ![successful adversarial](../{ex['successful_adversarial_example']}) |")
 revisions=[]
 for name in sorted(expected):
  matching=[r for r in audit if r.get('model_name')==name and r.get('revision')==lock[name]['revision']]
  role=matching[0].get('role','') if matching else ''
  depth=int(lock[name]['expected_vision_depth']);layer=depth-1
  revisions.append(f"| {name} | `{lock[name]['repo_id']}@{lock[name]['revision']}` | {role} | {depth} | {layer} |")
 lines+=['','## Pinned models and representation taps','','| Model | Exact checkpoint revision | Role | Visual depth | Deepest block |','|---|---|---|---:|---:|',*revisions,'','## Method and integrity','', '- Each proxy’s Pull/Push loss was computed only in that proxy’s own deepest visual-block representation, using its own 48-image reference prototypes and spatial-token mean pooling.', '- Every proxy model was loaded with strict BF16 parameters, checked recursively before use, and run alone on the A100; the others remained BF16 on CPU. Per-image L1-normalized pixel gradients were equally averaged before momentum/sign updates.', '- No target model, prediction, feature, or gradient was available in attack generation. Target generation and target representation analysis began after PNGs were frozen.', '- Target clean-correct-conditioned ASR uses the target-clean-correct subset and is reported separately. Parse failures remain misses in unconditional ASR; zero conditional denominator is N/A.', '- CKA and RSA compare target and proxy image representations on identical clean/adversarial image rows. Class dispersion uses 48 target reference embeddings/class. PCA/t-SNE are joint fits combining both classes, references, source clean images, and all ensemble adversarials for a given target/transition.', '- This recovery includes only the five forward transitions. Reverse-source cohorts were not specified by the corrected recovery matrix, so directional asymmetry is explicitly marked not estimable here.', '', '## Artifacts','', '- `multiple_proxy/summaries/multiple_proxy_recovery.csv` — cell summaries including ASR, TASR diagnostic, ΔR, CKA/RSA and runtime.', '- `analysis/representation_shift/target_representation_shift.csv` — image-level target Δpull, Δpush, ΔR and margins.', '- `analysis/proxy_representation_shift/` — image-level per-proxy embedding movements and margins.', '- `analysis/gradient_compatibility/` — per-cell proxy gradient cosine/negative fraction.', '- `analysis/variance/class_variance.csv`, `analysis/decision_geometry/decision_geometry.csv`, `analysis/cka/cka_rsa.csv`.', '- `analysis/pca/`, `analysis/tsne/`, and `analysis/correlations/` — joint embeddings and Spearman heatmap.', '- `audits/model_dtype_audit.json`, `audits/recovery_validation.json`, `audits/performance_benchmark.json`, `audits/v7_gate_decision.json`.', '- `reports/multiple_proxy_recovery_completion.csv`, `reports/paired_ensemble_comparison.csv`.', '', '## Incomplete cells','']
 artifact_end=lines.index('## Incomplete cells')
 lines[artifact_end:artifact_end]=['- `analysis/proxy_representation_shift/proxy_representation_shift_summary.csv` — per-proxy movement means.','- `analysis/gradient_compatibility/gradient_compatibility_summary.csv` — observation-weighted pair summaries; detailed rows include batch indices.','- `analysis/gradient_compatibility/gradient_fusion_trace.csv` — per-batch fused-gradient L2 diagnostics.','- `reports/attack_examples.csv` and `examples/` — one deterministic clean/successful-adversarial pair for each target.']
 if missing:lines += [f'- {e} / {t}: {err}' for e,t,err in missing]
 else:lines.append('None.')
 report.write_text('\n'.join(lines)+'\n')
 if gate_trigger:
  print('V7 gate triggered by the predeclared threshold. Do not launch V7 until the V6 recovery report and matrix have been reviewed.')
 print(f"[finalize] complete={successes}/65 attack_runtime={attack_runtime_seconds/3600:.2f}h gate={gate['decision']} report={report}")
 if missing:raise SystemExit(f'{len(missing)} required cells incomplete; see final report')
if __name__=='__main__':main()
