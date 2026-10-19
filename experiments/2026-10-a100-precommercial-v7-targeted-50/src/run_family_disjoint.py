#!/usr/bin/env python3
"""Fixed cross-family protocol correction, never target-guided method search."""
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
from scipy.stats import binomtest

from v7_common import OUT,EXP,REPO,REGISTRY,TIDS,records,sha,fingerprint,atomic_json,prepare_data
from family_protocol import corrected_conditions,condition_name,require_excluded_family,require_condition_target
from finalize_v7 import writecsv

R=OUT/'family_disjoint_recovery'


def execute(script,args,log):
    dest=R/'logs'/log;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('a') as stream:
        proc=subprocess.run([sys.executable,str(script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'Exit {proc.returncode}; inspect {dest}')


def validate_attack(c,root,smoke=False):
    count=0;peak=0.;seconds=0.;max_delta=0;budget=[]
    for tid in (['T01'] if smoke else TIDS):
        expected=[x['image_id'] for x in records(OUT/'manifests'/('dev' if smoke else 'test')/f'{tid}.jsonl')]
        if smoke:expected=expected[:2]
        ids=[]
        for path in sorted((root/tid).glob('batch_*/state.json')):
            d=json.loads(path.read_text());ids+=d['image_ids']
            require_excluded_family(d['models'],REGISTRY[c['target']]['family'],REGISTRY)
            assert d['models']==c['models'] and d['status']=='attack_complete'
            assert d['source_label']==int(tid[1:]) and d['target_label']==int(tid[1:])+1
            assert d['steps']==(2 if smoke else c['steps']) and d['seed']==c['seed'] and d['restarts']==1
            assert d['target_access'] is False
            assert d['config']['excluded_proxy_family']==REGISTRY[c['target']]['family']
            assert d['config']['input_diversity']==c['di']
            from run_v7_attack import METHODS
            expected_config={**METHODS[c['method']],**dict(fusion=c['fusion'],input_diversity=c['di'],smoothing=False),**c['options'],**dict(excluded_proxy_family=REGISTRY[c['target']]['family'])}
            if d['config']!=expected_config:raise RuntimeError('Actual objective differs from frozen condition')
            for rel,digest in d['png_hashes'].items():
                if sha(OUT/rel)!=digest:raise RuntimeError('Changed frozen PNG: '+rel)
            for clean,adv in zip(d['clean_paths'],d['adv_paths']):
                x=np.asarray(Image.open(OUT/clean)).astype('int16');y=np.asarray(Image.open(OUT/adv)).astype('int16')
                delta=int(np.max(np.abs(x-y)));max_delta=max(max_delta,delta)
                if delta>16:raise RuntimeError('L-infinity bound violation')
            if not smoke:
                for history in d['trace']:
                    assert len(history)==50
                    for step in history:
                        for key in ['losses','margins','gradient_l1','cancellation_ratio']:
                            if not np.isfinite(step[key]).all():raise RuntimeError('Nonfinite attack trace '+key)
            peak=max(peak,d['peak_reserved_gib']);seconds+=d['runtime_seconds']
            budget.append(dict(condition=condition_name(c),target=c['target'],transition=tid,images=len(d['image_ids']),steps=d['steps'],seed=d['seed'],seconds=d['runtime_seconds'],peak_reserved_gib=d['peak_reserved_gib'],model_calls=json.dumps(d['forward_counts'])))
        if ids!=expected:raise RuntimeError(f'Incomplete or mismatched frozen cohort: {root.name}/{tid}')
        count+=len(ids)
    if peak>75:raise RuntimeError(f'Peak reserved memory exceeded 75 GiB: {peak}')
    return dict(images=count,max_uint8_delta=max_delta,peak_reserved_gib=peak,attack_seconds=seconds,budget=budget)


def generate(c,smoke=False):
    family=REGISTRY[c['target']]['family'];require_excluded_family(c['models'],family,REGISTRY)
    cfg=R/'configs'/f"{c['id']}.json"
    if cfg.exists() and json.loads(cfg.read_text())!=c['options']:raise RuntimeError('Changed frozen condition config')
    atomic_json(cfg,c['options'])
    label=c['id']+('_smoke' if smoke else '')
    args=['--models',*c['models'],'--method',c['method'],'--fusion',c['fusion'],'--split','smoke' if smoke else 'test','--steps',2 if smoke else c['steps'],'--seed',c['seed'],'--restarts',1,'--batch-size',2,'--run-label',label,'--config-json',cfg,'--excluded-proxy-family',family]
    if c['di']:args.append('--input-diversity')
    if smoke:args+=['--transitions','T01','--limit-per-direction',2]
    execute(EXP/'src/run_v7_attack.py',args,label+'.log')
    name=condition_name(c)
    if smoke:name=name.replace(c['id']+'__',label+'__',1).replace('__i50__','__i2__')
    root=OUT/'attacks'/('smoke' if smoke else 'test')/name
    return root,validate_attack(c,root,smoke)


def valid_evaluation(c,lockpath):
    root=OUT/'attacks/test'/condition_name(c)
    found=0
    for tid in TIDS:
        ids=[]
        for path in sorted((root/tid).glob('batch_*/state.json')):
            d=json.loads(path.read_text())
            file=OUT/'evaluations'/root.name/c['target'].replace('/','_')/tid/path.parent.name/'metrics.json'
            if not file.exists():return False
            metric=json.loads(file.read_text())
            expected=fingerprint([d['identity'],d['png_hashes'],REGISTRY[c['target']],sha(lockpath)])
            if metric['identity']!=expected:return False
            ids += [r['image_id'] for r in metric['images']]
        if ids!=[r['image_id'] for r in records(OUT/'manifests/test'/f'{tid}.jsonl')]:return False
        found+=len(ids)
    return found==150


def report(lock,state):
    rows=[];detailed=[];masks={};budgets=[]
    for c in lock['configs']:
        name=condition_name(c);target=c['target']
        for tid in TIDS:
            images=[]
            for file in sorted((OUT/'evaluations'/name/target.replace('/','_')/tid).glob('batch_*/metrics.json')):
                metric=json.loads(file.read_text())
                if metric['method_lock_sha256']!=sha(R/'method_lock.json'):raise RuntimeError('Evaluation has different protocol lock')
                images+=metric['images']
            expected=[x['image_id'] for x in records(OUT/'manifests/test'/f'{tid}.jsonl')]
            actual=[x['image_id'] for x in images]
            if len(actual)!=len(set(actual)):raise RuntimeError('Duplicate evaluation image')
            complete=actual==expected
            if not set(actual)<=set(expected):raise RuntimeError('Evaluation contains unexpected cohort images')
            n=len(images);hits=sum(x['target_hit'] for x in images);cc=sum(x['clean_correct'] for x in images);hc=sum(x['target_hit'] and x['clean_correct'] for x in images)
            ci=binomtest(hits,n).proportion_ci() if n else None
            rows.append(dict(condition=name,id=c['id'],kind=c['kind'],target=target,transition=tid,seed=c['seed'],hits=hits,denominator=n,ASR=hits/n if n else None,CI_low=ci.low if ci else None,CI_high=ci.high if ci else None,clean_correct=cc,clean_correct_hits=hc,clean_correct_ASR=hc/cc if cc else None,parser_success=sum(x['adv_parser']=='ok' for x in images),legacy_conditional_TASR='N/A',status='complete' if complete else 'missing_or_partial'))
            for x in images:detailed.append(dict(condition=name,target=target,transition=tid,**x))
            masks[c['id'],tid]={x['image_id']:int(x['target_hit']) for x in images}
        if c['id'] in state.get('attacks',{}):budgets+=state['attacks'][c['id']]['budget']
    writecsv(R/'results/target_direction_results.csv',rows)
    writecsv(R/'results/per_image_results.csv',detailed)
    writecsv(R/'results/computation_budget.csv',budgets)
    writecsv(R/'reports/completion_matrix.csv',rows)
    for kind in {c['kind'] for c in lock['configs']}:
        writecsv(R/'results'/f'{kind}.csv',[x for x in rows if x['kind']==kind])
    paired=[];rng=np.random.default_rng(42)
    for main in [c for c in lock['configs'] if c['kind']=='primary_multi']:
        for single in [c for c in lock['configs'] if c['kind']=='matched_single' and c['target']==main['target']]:
            groups=[]
            for tid in TIDS:
                a=masks[main['id'],tid];b=masks[single['id'],tid]
                if set(a)!=set(b) or len(a)!=30:break
                groups.append(np.array([a[i]-b[i] for i in sorted(a)]))
            if len(groups)==5:
                boot=np.stack([g[rng.integers(0,30,(2000,30))].mean(1) for g in groups]).mean(0)
                paired.append(dict(target=main['target'],ensemble=main['id'],single=single['id'],N=150,ASR_difference=float(np.concatenate(groups).mean()),CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975))))
    writecsv(R/'results/paired_single_comparisons.csv',paired)
    # Keep the previously completed Gemma measurements, and explicitly relabel
    # old Qwen/InternVL as family-overlap rather than strict cross-family data.
    previous=list(csv.DictReader((OUT/'ten_hour_research/final_target_results.csv').open()))
    preserved=[x for x in previous if 'final_0__' in x['condition'] and x['target']=='Gemma 4 31B-it']
    primary=[dict(target=x['target'],transition=x['transition'],hits=x['hits'],denominator=x['denominator'],ASR=x['ASR'],source='corrected_family_disjoint',status=x['status']) for x in rows if x['kind']=='primary_multi']
    primary += [dict(target=x['target'],transition=x['transition'],hits=int(x['hits']),denominator=int(x['denominator']),ASR=float(x['ASR']),source='preserved_previous_Gemma_cross_family',status=x['status']) for x in preserved]
    writecsv(R/'results/combined_strict_cross_family_primary.csv',primary)
    completed=sum(x['status']=='complete' for x in rows)
    primary_pass=len(primary)==15 and all(x['status']=='complete' and x['denominator']==30 and x['hits']>=15 for x in primary)
    atomic_json(R/'completion.json',dict(completed_cells=completed,expected_cells=len(rows),expected_image_evaluations=len(rows)*30,measured_image_evaluations=len(detailed),all_complete=completed==len(rows),all_15_strict_cross_family_primary_cells_at_least_50pct=primary_pass,failed_attempts=len(state.get('failures',[]))))
    lines=['# Strict cross-family V7 correction','','Task: fixed ten-class targeted image recognition. No open-ended VQA.','',f"Completed cells: {completed}/{len(rows)}. Image evaluations: {len(detailed)}/{len(rows)*30}. Recorded failed attempts: {len(state.get('failures',[]))}.",'','Qwen27B excludes every Qwen proxy; InternVL14B excludes every InternVL proxy. Gemma31B prior cross-family primary results are preserved. Old Qwen/InternVL measurements are family-overlap experiments, not valid strict-cross-family replacements.','','The algorithm, proxy substitutions, all ablations and seeds were predeclared before corrected target evaluation. This is a protocol correction after earlier target results were observed; it is not an untouched new benchmark. No further target-feedback optimization is permitted. All ground-truth cohorts, references, prompts, epsilon16/255 and 50 steps are unchanged.','','## Primary results','','| Target | Direction | Hits/N | ASR |','|---|---|---|---|']
    for x in primary:lines.append(f"| {x['target']} | {x['transition']} | {x['hits']}/{x['denominator']} | {x['ASR']:.1%} |" if x['ASR'] is not None else f"| {x['target']} | {x['transition']} | pending | N/A |")
    lines+=['',f'All 15 strict-cross-family primary cells >=15/30: {primary_pass}.','','All matched singles, five individual layer depths, component ablations and primary seeds43/44 are reported in results/*.csv. No favorable seed replaces seed42. Original Pull/Push uses deepest layer without DI and is an historical algorithm control, not a single-component ablation. The no-patch and no-DI conditions provide matched component controls.','','Undefined conditional TASR is N/A. Missing results are not zero. Geometry is restricted to the two corrected primary ensembles. No reverse-direction cohort is present, so directional asymmetry is not estimated.']
    if 'finished' in state:lines+=['',f"Actual campaign wall seconds: {state['finished']-state['started']:.1f}."]
    (R/'reports').mkdir(exist_ok=True);(R/'reports/final_report.md').write_text('\n'.join(lines)+'\n')
    return rows


def archive(lock,state):
    events=json.loads((OUT/'audits/model_dtype_audit.json').read_text())
    relevant=set(lock['targets'])|{n for c in lock['configs'] for n in c['models']}
    selected=[e for e in events if e['model_name'] in relevant and e['audited_utc']>=state['started_utc']]
    audited={e['model_name'] for e in selected}
    passed=all(e['all_floating_parameters_bf16'] and e['status']=='pass' for e in selected) and audited==relevant
    atomic_json(R/'audits/bf16.json',dict(passed=passed,expected_models=sorted(relevant),audited_models=sorted(audited),load_events=selected))
    atomic_json(R/'audits/runtime.json',dict(started=state['started'],finished=state.get('finished'),wall_seconds=time.time()-state['started'],attack_seconds=sum(d['attack_seconds'] for d in state['attacks'].values()),peak_reserved_gib=max((d['peak_reserved_gib'] for d in state['attacks'].values()),default=0)))
    validation=dict(all_loaded_models_bf16=passed,all_attacks_validated=len(state['attacks'])==len(lock['configs']),family_disjoint=True,reference_and_attack_disjoint=True,target_feedback_for_correction=False,historical_target_results_previously_observed=True,algorithm_selected_before_corrected_target_evaluation=True)
    atomic_json(R/'audits/validation.json',validation)
    # Full PNGs and embeddings stay on persistent root storage. Push reports,
    # measured CSV/JSON data and a few plots/examples, never checkpoint weights.
    rows=list(csv.DictReader((R/'results/per_image_results.csv').open())) if (R/'results/per_image_results.csv').exists() else []
    for hit in ['True','False']:
        x=next((x for x in rows if x['target_hit']==hit and '_main__' in x['condition']),None)
        if not x:continue
        for f in (OUT/'attacks/test'/x['condition']/x['transition']).glob('batch_*/state.json'):
            d=json.loads(f.read_text())
            if x['image_id'] not in d['image_ids']:continue
            i=d['image_ids'].index(x['image_id']);dest=R/'examples';dest.mkdir(exist_ok=True)
            import shutil
            for kind,key in [('clean','clean_paths'),('adv','adv_paths')]:shutil.copyfile(OUT/d[key][i],dest/f'{hit}_{kind}.png')
            atomic_json(dest/f'{hit}_metadata.json',x);break
    files=[p for p in R.rglob('*') if p.is_file() and 'logs' not in p.relative_to(R).parts and 'embeddings' not in p.relative_to(R).parts and p.suffix in ['.csv','.json','.md','.yaml','.png']]
    rel=[str(p.relative_to(REPO)) for p in files]
    try:
        assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='experiment/a100-precommercial-v7-targeted-50'
        subprocess.run(['git','add','-f','--',*rel],cwd=REPO,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*rel],cwd=REPO).returncode:
            subprocess.run(['git','commit','--only','-m','Archive corrected strict cross-family V7 results and audits','--',*rel],cwd=REPO,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:experiment/a100-precommercial-v7-targeted-50'],cwd=REPO,check=True,timeout=120)
        atomic_json(R/'upload_status.json',dict(status='pushed',commit=commit))
    except Exception as exc:atomic_json(R/'upload_status.json',dict(status='failed',error=str(exc)))


def main():
    import torch
    R.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('Global blocker: CUDA unavailable')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Global blocker: BF16 unavailable')
    prepare_data();conditions=corrected_conditions()
    lockpath=R/'method_lock.json'
    spec=dict(protocol='strict_family_disjoint',task='fixed_10_class_recognition',configs=conditions,targets=['Qwen3.5-27B','InternVL3.5-14B-HF'],allowed_test_conditions=[condition_name(c) for c in conditions],condition_targets={condition_name(c):c['target'] for c in conditions},condition_models={condition_name(c):c['models'] for c in conditions},primary_conditions={c['target']:condition_name(c) for c in conditions if c['kind']=='primary_multi'},test_manifest_hashes={t:sha(OUT/'manifests/test'/f'{t}.jsonl') for t in TIDS},reference_hashes={str(i):sha(OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl') for i in range(1,11)},model_revisions=REGISTRY,algorithm_feedback_from_corrected_targets=False,historical_target_results_previously_observed=True,attack=dict(epsilon='16/255',step_size='1/255',steps=50,momentum=1,random_start=True,image_size=224),source_hashes={n:sha(EXP/'src'/n) for n in ['run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py','family_protocol.py']})
    if lockpath.exists():
        if json.loads(lockpath.read_text())!=spec:raise RuntimeError('Immutable cross-family lock or code/cohort changed; do not overwrite')
    else:atomic_json(lockpath,spec)
    lock=spec
    history=R/'state.json'
    state=json.loads(history.read_text()) if history.exists() else dict(started=time.time(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),smokes={},attacks={},evaluations={},failures=[])
    def save():atomic_json(history,state)
    def stage(name,condition=None):
        atomic_json(R/'current_stage.json',dict(stage=name,condition=condition,at=time.time()))
        print(f'[cross-family] {name} {condition or ""} attacks={len(state["attacks"])}/{len(conditions)} evaluations={len(state["evaluations"])}/{len(conditions)}',flush=True)
    def failed(stage,c,exc):
        state['failures'].append(dict(stage=stage,condition=c['id'],error=str(exc),traceback=traceback.format_exc()));save()
        print('[local-failure]',stage,c['id'],str(exc),flush=True)
    save();report(lock,state)
    # Real BF16 proxy gradients on two images/two steps for both corrected pools.
    for c in [x for x in conditions if x['kind']=='primary_multi']:
        if c['id'] in state['smokes']:continue
        stage('smoke',c['id'])
        root,audit=generate(c,smoke=True);state['smokes'][c['id']]=audit;save()
    # Predeclared priority: both multi methods and all matched singles first.
    # Later ablations never use these outcomes to revise their fixed settings.
    priority=[c for c in conditions if c['kind'] in ('primary_multi','matched_single')]
    remainder=[c for c in conditions if c not in priority]
    for block in [priority,remainder]:
        for c in block:
            if c['id'] in state['attacks']:
                validate_attack(c,OUT/'attacks/test'/condition_name(c));continue
            stage('attack',c['id'])
            try:
                root,audit=generate(c);state['attacks'][c['id']]=audit;save()
            except Exception as exc:failed('attack',c,exc)
        for target in lock['targets']:
            for c in [x for x in block if x['target']==target and x['id'] in state['attacks']]:
                if c['id'] in state['evaluations'] and valid_evaluation(c,lockpath):continue
                stage('target_evaluation',c['id'])
                try:
                    execute(EXP/'src/evaluate_v7.py',['--root',OUT/'attacks/test'/condition_name(c),'--target',target,'--method-lock',lockpath],c['id']+'_evaluation.log')
                    # Verify all five evaluation cells before considering resume valid.
                    measured=report(lock,state)
                    own=[x for x in measured if x['id']==c['id']]
                    if len(own)!=5 or not all(x['status']=='complete' for x in own):raise RuntimeError('Missing evaluation images')
                    state['evaluations'][c['id']]=dict(finished=time.time());save()
                except Exception as exc:failed('evaluation',c,exc)
    report(lock,state)
    if len(state['evaluations'])==len(conditions):
        stage('primary_representation_analysis')
        # Primary-only scope avoids repeating expensive inference for every loss
        # and preserves all previous Gemma analysis files.
        primary=[c for c in conditions if c['kind']=='primary_multi']
        analysis={**lock,'configs':primary,'allowed_test_conditions':[condition_name(c) for c in primary]}
        analysis_path=R/'analysis_lock.json';atomic_json(analysis_path,analysis)
        try:
            if not (R/'analysis/posthoc_analysis.json').exists():execute(EXP/'src/analyze_v7.py',['--method-lock',analysis_path,'--analysis-dir',R/'analysis','--results-csv',R/'results/target_direction_results.csv'],'analysis.log')
        except Exception as exc:failed('analysis',dict(id='primary_geometry'),exc)
    if len(state['evaluations'])==len(conditions) and (R/'analysis/posthoc_analysis.json').exists():state['finished']=time.time()
    save();report(lock,state);archive(lock,state)
    stage('complete' if 'finished' in state else 'incomplete')
    if 'finished' not in state:raise SystemExit(1)


if __name__=='__main__':main()
