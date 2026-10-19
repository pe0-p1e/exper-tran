#!/usr/bin/env python3
"""Gated, resumable held-out-category study; no target-based retuning."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from scipy.stats import binomtest
from v7_common import OUT,EXP,REGISTRY,atomic_json,sha,fingerprint
from category_data import validate_dictionary,validate_nested_dictionaries,classification_prompt,parse_class_code
from family_protocol import require_excluded_family
from proxy_selection_tables import writecsv


def require_selection_complete():
    p=OUT/'proxy_selection/completion.json'
    if not p.exists():raise RuntimeError('Proxy-selection measurements must finish before category generalization')
    d=json.loads(p.read_text())
    if not d.get('all_complete') or not d.get('feature_complete'):raise RuntimeError('Proxy-selection results or clean feature measurements are incomplete')


def validate_frozen_attacks(root,data,manifest_hash,expected_models=None):
    import hashlib
    import numpy as np
    from PIL import Image
    for task in data['transitions']:
        ids=[];cohort={r['image_id']:r for r in task['images']}
        for p in sorted((root/task['id']).glob('batch_*/state.json')):
            d=json.loads(p.read_text());ids+=d['image_ids']
            if d.get('status')!='attack_complete' or d['manifest_sha256']!=manifest_hash or d['steps']!=50 or d['seed']!=42 or d['restarts']!=1 or d['target_access']:raise RuntimeError('Unfrozen/budget-mismatched new-category attack')
            if expected_models is not None and d['models']!=expected_models:raise RuntimeError('Attack proxies differ from new-category method lock')
            if d['class_count']!=len(data['classes']) or d['config'].get('patch_weight')!=.35 or not d['config'].get('input_diversity'):raise RuntimeError('Changed class count or frozen loss')
            if d['source_label']!=task['source_label'] or d['target_label']!=task['target_label']:raise RuntimeError('Changed class mapping')
            if not len(d['image_ids'])==len(d['clean_paths'])==len(d['adv_paths']):raise RuntimeError('Mismatched image/path counts')
            for rel,h in d['png_hashes'].items():
                if sha(root/rel)!=h:raise RuntimeError('Changed new-category PNG')
            for iid,c,a in zip(d['image_ids'],d['clean_paths'],d['adv_paths']):
                clean=np.asarray(Image.open(root/c));adv=np.asarray(Image.open(root/a))
                if clean.shape!=(224,224,3) or adv.shape!=clean.shape:raise RuntimeError('Changed canonical image dimensions')
                if iid not in cohort or hashlib.sha256(clean.tobytes()).hexdigest()!=cohort[iid]['pixel_sha256']:raise RuntimeError('Frozen clean pixels differ from manifest')
                if np.abs(clean.astype('int16')-adv.astype('int16')).max()>16:raise RuntimeError('PNG perturbation violation')
        if ids!=[r['image_id'] for r in task['images']]:raise RuntimeError('Incomplete new-category attack cohort')


def evaluate_jobs(jobs,target,study):
    import gc
    import torch
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
    from strict_model_audit import audit_loaded_model
    for job in jobs:
        data=validate_dictionary(json.loads(Path(job['manifest']).read_text()),Path(job['manifest']).parents[1])
        validate_frozen_attacks(Path(job['attack_root']),data,job['manifest_sha256'],job['models'])
        require_excluded_family(job['models'],REGISTRY[target]['family'],REGISTRY)
    spec=REGISTRY[target];MODEL_REVISIONS[spec['repo_id']]=spec['revision']
    snap=local_snapshot(Path(os.environ['HF_HOME']),spec['repo_id'],spec['revision'])
    processor=load_processor(snap);model=load_target_for_generation(snap,torch.device('cuda'),precision='bf16')
    audit_loaded_model(model,model_name=target,revision=spec['revision'],role='held_out_category_target',repo_id=spec['repo_id'],output_dir=OUT)
    generator=TransformersTargetGenerator(model,processor)
    failures=[]
    try:
        for job in jobs:
            data=json.loads(Path(job['manifest']).read_text());root=Path(job['attack_root']);prompt=classification_prompt(data['classes'])
            validate_frozen_attacks(root,data,job['manifest_sha256'],job['models'])
            for task in data['transitions']:
                for statepath in sorted((root/task['id']).glob('batch_*/state.json')):
                    try:
                        d=json.loads(statepath.read_text());require_excluded_family(d['models'],REGISTRY[target]['family'],REGISTRY)
                        dest=study/'evaluations'/job['id']/task['id']/statepath.parent.name/'metrics.json'
                        identity=fingerprint([d['identity'],d['png_hashes'],spec,prompt,sha(study/'method_lock.json')])
                        if dest.exists() and json.loads(dest.read_text())['identity']==identity:continue
                        rows=[]
                        for i,iid in enumerate(d['image_ids']):
                            with torch.inference_mode():clean,adv=generator.generate_labels([root/d['clean_paths'][i],root/d['adv_paths'][i]],prompt)
                            cp=parse_class_code(clean.raw_output,len(data['classes']));ap=parse_class_code(adv.raw_output,len(data['classes']))
                            rows.append(dict(image_id=iid,clean_raw=clean.raw_output,adv_raw=adv.raw_output,clean_label=cp['label'],adv_label=ap['label'],clean_parser=cp['status'],adv_parser=ap['status'],clean_correct=cp['label']==task['source_label'],target_hit=ap['label']==task['target_label']))
                        atomic_json(dest,dict(identity=identity,method_lock_sha256=sha(study/'method_lock.json'),images=rows,target=target,class_count=len(data['classes']),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                    except Exception as exc:
                        failures.append(dict(job=job['id'],transition=task['id'],batch=statepath.parent.name,stage='evaluation',error=str(exc)))
                        torch.cuda.empty_cache()
    finally:del generator,model,processor;gc.collect();torch.cuda.empty_cache()
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=OUT/'category_generalization/data');p.add_argument('--study-root',type=Path,default=OUT/'category_generalization/study');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    a.data_root=a.data_root.resolve();a.study_root=a.study_root.resolve()
    if not a.study_root.is_relative_to(OUT.resolve()):raise ValueError('Study results must remain under the persistent V7 output root')
    manifests=sorted((a.data_root/'manifests').glob('classes_*.json'))
    if not manifests:raise RuntimeError('Dataset blocker: prepare new-class manifests with category_data.py first')
    dictionaries=[validate_dictionary(json.loads(path.read_text()),a.data_root) for path in manifests]
    order=sorted(zip(manifests,dictionaries),key=lambda x:len(x[1]['classes']))
    validate_nested_dictionaries([d for _,d in order])
    framework=yaml.safe_load((EXP/'latest_experiment_framework.yaml').read_text())
    profiles=framework['proxy_selection']['target_slots'];jobs=[]
    for path,data in order:
        for index,(target,slots) in enumerate(profiles.items()):
            names=[slots[k] for k in ['V','A','B']];require_excluded_family(names,REGISTRY[target]['family'],REGISTRY)
            id=f'C{len(data["classes"])}_T{index}'
            jobs.append(dict(id=id,target=target,models=names,manifest=str(path.resolve()),manifest_sha256=sha(path),attack_root=str((a.study_root/'attacks'/id).resolve())))
    plan=dict(protocol='held_out_categories_fixed_VAB',jobs=jobs,proxies_selected_from_actual_target_ranking=False,main_method=dict(layers=[.25,.5,.75,1.],patch_weight=.35,DI=True,seed=42,steps=50,epsilon='16/255'),model_revisions=REGISTRY,source_hashes={name:sha(EXP/'src'/name) for name in ['category_attack.py','category_features.py','category_data.py','run_category_generalization.py','run_v7_attack.py','vision_features.py','representation_objectives.py','attack_optimizers.py']})
    a.study_root.mkdir(parents=True,exist_ok=True)
    if a.prepare_only:atomic_json(a.study_root/'proposed_plan.json',plan);return
    require_selection_complete();lock=a.study_root/'method_lock.json'
    if lock.exists() and json.loads(lock.read_text())!=plan:raise RuntimeError('Category method lock changed')
    if not lock.exists():atomic_json(lock,plan)
    provenance=a.study_root/'data_manifests'
    for path in manifests:
        atomic_json(provenance/path.name,json.loads(path.read_text()))
    atomic_json(provenance/'data_lock.json',json.loads((a.data_root/'data_lock.json').read_text()))
    started_file=a.study_root/'execution_start.json'
    if not started_file.exists():atomic_json(started_file,dict(started=time.time()))
    failures=[];ready=[]
    # Both source/target classes are new; no image or hyperparameter screening.
    for job in jobs:
        log=a.study_root/'logs'/(job['id']+'_attack.log');log.parent.mkdir(exist_ok=True)
        try:
            # A real2-image/2-step smoke, isolated from all full50-step artifacts.
            for smoke in [True,False]:
                output=job['attack_root']+('_smoke' if smoke else '')
                cmd=[sys.executable,str(EXP/'src/category_attack.py'),'--manifest',job['manifest'],'--data-root',str(a.data_root),'--output',output,'--target',job['target'],'--models',*job['models']]+(['--smoke'] if smoke else [])
                with log.open('a') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
            data=json.loads(Path(job['manifest']).read_text());validate_frozen_attacks(Path(job['attack_root']),data,job['manifest_sha256'],job['models']);ready.append(job)
        except Exception as exc:failures.append(dict(job=job['id'],stage='attack',error=str(exc)))
    for target in profiles:
        selected=[j for j in ready if j['target']==target]
        if not selected:continue
        try:failures.extend(evaluate_jobs(selected,target,a.study_root))
        except Exception as exc:failures.append(dict(target=target,stage='evaluation',error=str(exc)))
    rows=[];detailed=[]
    for job in jobs:
        data=json.loads(Path(job['manifest']).read_text())
        for task in data['transitions']:
            images=[]
            for file in sorted((a.study_root/'evaluations'/job['id']/task['id']).glob('batch_*/metrics.json')):
                if job not in ready:continue
                m=json.loads(file.read_text());statefile=Path(job['attack_root'])/task['id']/file.parent.name/'state.json'
                state=json.loads(statefile.read_text())
                expected=fingerprint([state['identity'],state['png_hashes'],REGISTRY[job['target']],classification_prompt(data['classes']),sha(lock)])
                if m['identity']!=expected or m['method_lock_sha256']!=sha(lock):raise RuntimeError('Stale or changed category evaluation identity')
                images+=m['images']
            n=len(images);h=sum(x['target_hit'] for x in images);cc=sum(x['clean_correct'] for x in images);hc=sum(x['target_hit'] and x['clean_correct'] for x in images)
            ci=binomtest(h,n).proportion_ci() if n else None
            expected_ids=[r['image_id'] for r in task['images']];actual_ids=[x['image_id'] for x in images]
            if actual_ids not in ([],expected_ids):
                if len({x['image_id'] for x in images})!=n or not {x['image_id'] for x in images}<={r['image_id'] for r in task['images']}:raise RuntimeError('Wrong category evaluation cohort')
                if n==30:raise RuntimeError('Category evaluation reordered frozen cohort')
            rows.append(dict(condition=job['id'],target=job['target'],class_count=len(data['classes']),transition=task['id'],hits=h,denominator=n,ASR=h/n if n else None,CI_low=ci.low if ci else None,CI_high=ci.high if ci else None,clean_correct=cc,clean_accuracy=cc/n if n else None,clean_correct_ASR=hc/cc if cc else None,parser_success=sum(x['adv_parser']=='ok' for x in images),status='complete' if actual_ids==expected_ids else 'pending_or_partial'))
            detailed += [dict(condition=job['id'],target=job['target'],transition=task['id'],**x) for x in images]
    writecsv(a.study_root/'results/target_direction_results.csv',rows);writecsv(a.study_root/'results/per_image_results.csv',detailed)
    audit_events=json.loads((OUT/'audits/model_dtype_audit.json').read_text());names=set(profiles)|{n for j in jobs for n in j['models']}
    latest={e['model_name']:e for e in audit_events if e['model_name'] in names}
    bf16=set(latest)==names and all(e['all_floating_parameters_bf16'] and e['status']=='pass' and e['revision']==REGISTRY[n]['revision'] for n,e in latest.items())
    atomic_json(a.study_root/'audits/bf16.json',dict(all_expected_models_verified=bf16,latest_model_audits=latest))
    atomic_json(a.study_root/'completion.json',dict(all_complete=all(r['status']=='complete' for r in rows) and bf16,expected_cells=len(jobs)*5,completed_cells=sum(r['status']=='complete' for r in rows),BF16=bf16,failures=failures,category_generalization_only=True,open_ended_VQA=False))
    lines=['# Held-out-category generalization results','','Frozen V+A+B proxy sets, original50-step visual objective, no target-feedback tuning. New references are supplied; this is reference-conditioned category generalization, not proof of open-ended generalization.','', '| Target | Classes | Direction | Hits/N | ASR | Status |','|---|---|---|---|---|---|']
    for r in rows:
        score=f"{r['ASR']:.1%}" if r['ASR'] is not None else 'N/A'
        lines.append(f"| {r['target']} | {r['class_count']} | {r['transition']} | {r['hits']}/{r['denominator']} | {score} | {r['status']} |")
    lines+=['',f'BF16 audit passed: {bf16}. Recorded failures: {len(failures)}. Missing cells are not zero. Exact source/target class names and reference/image IDs are in the frozen manifests.']
    reports=a.study_root/'reports';reports.mkdir(exist_ok=True);(reports/'final_report.md').write_text('\n'.join(lines)+'\n')
    start=json.loads(started_file.read_text())['started']
    atomic_json(a.study_root/'audits/runtime.json',dict(started=start,finished=time.time(),elapsed_wall_including_any_interruptions=time.time()-start))
    from archive_next_stage import upload_compact
    upload_compact(a.study_root)
    if not all(r['status']=='complete' for r in rows) or not bf16:raise SystemExit(1)


if __name__=='__main__':main()
