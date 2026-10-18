#!/usr/bin/env python3
"""Validate complete V7 artifacts and report exact performance with uncertainty."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import binomtest
from v7_common import OUT, TARGETS, TIDS, records, sha, atomic_json


def writecsv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    lock=json.loads((OUT/'audits/method_lock.json').read_text())
    audits=json.loads((OUT/'audits/model_dtype_audit.json').read_text())
    bf16=all(r['all_floating_parameters_bf16'] and r['status']=='pass' for r in audits)
    atomic_json(OUT/'audits/bf16.json',dict(pass_=bf16,load_events=len(audits),models=sorted(set(r['model_name'] for r in audits)),source='model_dtype_audit.json'))
    validations=[];runtime=0.;budget=[]
    for condition in lock['allowed_test_conditions']:
        for tid in TIDS:
            expected=[r['image_id'] for r in records(OUT/'manifests/test'/f'{tid}.jsonl')];ids=[]
            for p in sorted((OUT/'attacks/test'/condition/tid).glob('batch_*/state.json')):
                d=json.loads(p.read_text());ids+=d['image_ids'];runtime+=d['runtime_seconds']
                for rel,h in d['png_hashes'].items():
                    if sha(OUT/rel)!=h:raise RuntimeError('Changed frozen PNG')
                for c,a in zip(d['clean_paths'],d['adv_paths']):
                    x=np.asarray(Image.open(OUT/c)).astype('int16');y=np.asarray(Image.open(OUT/a)).astype('int16')
                    if np.max(abs(x-y))>16:raise RuntimeError('L-infinity violation')
                if d['target_access']:raise RuntimeError('Target isolation failure')
                budget.append(dict(condition=condition,transition=tid,N=len(d['image_ids']),steps=d['steps'],restarts=d['restarts'],seconds=d['runtime_seconds'],peak_gib=d['peak_reserved_gib'],calls=json.dumps(d['forward_counts'])))
            validations.append(dict(condition=condition,transition=tid,images=len(ids),cohort_match=ids==expected))
    writecsv(OUT/'reports/computation_budget.csv',budget)
    atomic_json(OUT/'audits/linf.json',dict(pass_=True,unit='uint8 channel difference',max_allowed=16))
    atomic_json(OUT/'audits/target_isolation.json',dict(pass_=True,method_lock_sha256=sha(OUT/'audits/method_lock.json'),target_feedback=False))
    atomic_json(OUT/'audits/runtime.json',dict(attack_seconds=runtime,scope='test generation only; development/calibration excluded'))
    rows=[];masks={}
    for cond in lock['allowed_test_conditions']:
        for target in TARGETS:
            for tid in TIDS:
                metrics=[]
                for p in sorted((OUT/'evaluations'/cond/target.replace('/','_')/tid).glob('batch_*/metrics.json')):metrics+=json.loads(p.read_text())['images']
                n=len(metrics);h=sum(r['target_hit'] for r in metrics);cc=sum(r['clean_correct'] for r in metrics);ch=sum(r['clean_correct'] and r['target_hit'] for r in metrics)
                ci=binomtest(h,n).proportion_ci() if n else None
                masks[cond,target,tid]={r['image_id']:r['target_hit'] for r in metrics}
                rows.append(dict(condition=cond,target=target,transition=tid,status='complete' if n==30 else 'missing',hits=h,denominator=n,ASR=h/n if n else None,CI_low=ci.low if ci else None,CI_high=ci.high if ci else None,clean_correct=cc,conditional_hits=ch,conditional_denominator=cc,conditional_ASR=ch/cc if cc else None,passes_goal=n==30 and h>=15))
    writecsv(OUT/'reports/completion_matrix.csv',rows);writecsv(OUT/'reports/target_direction_results.csv',rows);writecsv(OUT/'reports/controlled_ablation.csv',rows)
    # Paired differences, stratified by fixed transition, with explicitly matched IDs.
    rng=np.random.default_rng(42);paired=[]
    baseline=next((c for c in lock['allowed_test_conditions'] if c.startswith('A0__equal__') and '+' in c),None)
    if baseline:
        for cond in lock['allowed_test_conditions']:
            if cond==baseline:continue
            for target in TARGETS:
                diffs=[]
                for tid in TIDS:
                    a=masks.get((cond,target,tid),{});b=masks.get((baseline,target,tid),{})
                    if set(a)!=set(b) or len(a)!=30:continue
                    ids=sorted(a);diffs.append(np.array([int(a[i])-int(b[i]) for i in ids]))
                if len(diffs)!=5:continue
                boot=np.stack([d[rng.integers(0,30,(2000,30))].mean(1) for d in diffs]).mean(0)
                paired.append(dict(condition=cond,baseline=baseline,target=target,paired_N=150,absolute_ASR_difference=float(np.concatenate(diffs).mean()),bootstrap_low=float(np.quantile(boot,.025)),bootstrap_high=float(np.quantile(boot,.975))))
    writecsv(OUT/'reports/paired_baselines.csv',paired)
    evaluation_complete=bf16 and all(v['cohort_match'] for v in validations) and all(r['status']=='complete' for r in rows)
    analysis_path=OUT/'audits/posthoc_analysis.json'
    analysis=json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    complete=evaluation_complete and analysis.get('method_lock_sha256')==sha(OUT/'audits/method_lock.json') and analysis.get('status')=='complete'
    atomic_json(OUT/'audits/final_validation.json',dict(pass_=complete,cells=len(rows),complete_cells=sum(r['status']=='complete' for r in rows),cohorts=validations))
    lines=['# V7 experiment report','',f'Execution status: {"COMPLETE" if complete else "EVALUATIONS COMPLETE; ANALYSIS PENDING" if evaluation_complete else "INCOMPLETE"}. Exact model revisions: `audits/model_revisions.json`.','',f'BF16 load audits: {len(audits)}, all pass: {bf16}. Test attack time: {runtime/3600:.2f} h.','', 'Method choice is frozen in `audits/method_lock.json`, selected exclusively from disjoint held-out surrogate development metrics. No final target feedback is used for selection. The representation hit metric used for development is not MLLM output ASR.','', 'Primary: unconditional targeted ASR on 30 source-ground-truth images/direction. Secondary conditional ASR uses clean-correct samples; zero denominators are N/A. Seeds 42,43,44 are reported independently. Multi-restart compute is separately labeled.','', '| Condition | Target | Direction | Hits/N | ASR |','|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['condition']} | {r['target']} | {r['transition']} | {r['hits']}/{r['denominator']} | {r['ASR'] if r['ASR'] is not None else 'N/A'} |")
    lines+=['','## Acceptance and limitations','']
    for cond in lock['allowed_test_conditions']:
        selected=[r for r in rows if r['condition']==cond]
        min_rate=min((r['ASR'] for r in selected if r['ASR'] is not None),default=None)
        passes=len(selected)==15 and all(r['passes_goal'] for r in selected)
        lines.append(f'- {cond}: minimum target-direction ASR={min_rate}; all_15_cells_above_50pct={passes}.')
    if complete:lines += ['', 'Post-hoc DeltaR, variance, margins, CKA/RSA, joint PCA/t-SNE and Spearman files are available under analysis/. Reverse-direction asymmetry is not estimated by this forward-only V7 matrix.']
    lines += ['','N=30 per direction gives wide uncertainty; confidence intervals and paired comparisons accompany exact masks. Statistical differences do not establish a causal mechanism. More proxies, lower surrogate loss and larger representation shifts do not guarantee targeted transfer. Any unmet 50% goal remains unmet; no post-test algorithm search is authorized by this report.','', 'Post-hoc feature measurements are recorded separately; missing or incompatible measurements must not be described as completed.']
    (OUT/'reports/final_report.md').write_text('\n'.join(lines)+'\n')
    if not evaluation_complete:raise RuntimeError('Final validation incomplete')

if __name__=='__main__':main()
