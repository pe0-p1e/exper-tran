#!/usr/bin/env python3
"""Complete the predeclared seven-subset ablation after corrected recovery."""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from v7_common import OUT,EXP,REGISTRY,TIDS,records,sha,fingerprint,atomic_json
from family_protocol import condition_name,require_condition_target
from proxy_selection_tables import ROOT,matrix,prepare,writecsv


def require_recovery_complete():
    path=OUT/'family_disjoint_recovery/state.json'
    audit=OUT/'family_disjoint_recovery/audits/validation.json'
    if not path.exists() or not audit.exists():raise RuntimeError('Prerequisite: corrected recovery and audits must finish first')
    state=json.loads(path.read_text());checked=json.loads(audit.read_text())
    if not state.get('finished') or not checked.get('all_attacks_validated') or not checked.get('all_loaded_models_bf16'):
        raise RuntimeError('Corrected recovery is unfinished or failed validation')


def command(script,args,log):
    file=ROOT/'logs'/log;file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('a') as stream:
        proc=subprocess.run([sys.executable,str(script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'Exit {proc.returncode}: {file}')


def raw_results(plan,lock):
    rows=[];detailed=[]
    for item in plan:
        if item['source']!='new_subset_measurement':continue
        c=item['config'];name=condition_name(c)
        for tid in TIDS:
            data=[]
            for path in sorted((OUT/'evaluations'/name/c['target'].replace('/','_')/tid).glob('batch_*/metrics.json')):
                m=json.loads(path.read_text())
                if m['method_lock_sha256']!=sha(ROOT/'method_lock.json'):raise RuntimeError('Mismatched selection evaluation lock')
                state=json.loads((OUT/m['source_state']).read_text())
                expected=fingerprint([state['identity'],state['png_hashes'],REGISTRY[c['target']],sha(ROOT/'method_lock.json')])
                if m['identity']!=expected:raise RuntimeError('Changed evaluation identity')
                data+=m['images']
            ids=[x['image_id'] for x in data]
            expected=[r['image_id'] for r in records(OUT/'manifests/test'/f'{tid}.jsonl')]
            if len(ids)!=len(set(ids)) or not set(ids)<=set(expected):raise RuntimeError('Unexpected/duplicate evaluation images')
            n=len(data);hits=sum(x['target_hit'] for x in data)
            rows.append(dict(condition=name,target=c['target'],transition=tid,hits=hits,denominator=n,ASR=hits/n if n else None,status='complete' if ids==expected else 'pending_or_partial'))
            detailed += [dict(condition=name,target=c['target'],transition=tid,**x) for x in data]
    writecsv(ROOT/'results/target_direction_results.csv',rows);writecsv(ROOT/'results/per_image_results.csv',detailed)
    return prepare(plan)


def verify_reused(item):
    """Verify actual frozen pixels/cohort/answers, not only a summary CSV."""
    import numpy as np
    from PIL import Image
    c=item['config'];name=item['measurement_condition']
    source_lock=OUT/('family_disjoint_recovery/method_lock.json' if item['source']=='corrected_family_disjoint' else 'ten_hour_research/selected_method_lock.json')
    for tid in TIDS:
        ids=[];evaluated=[]
        for file in sorted((OUT/'attacks/test'/name/tid).glob('batch_*/state.json')):
            d=json.loads(file.read_text());ids+=d['image_ids']
            require_condition_target(name,c['target'],d['models'],dict(protocol='strict_family_disjoint',condition_targets={name:c['target']},condition_models={name:c['models']}),REGISTRY)
            if d['steps']!=50 or d['seed']!=42 or d['restarts']!=1 or d['target_access']:raise RuntimeError('Incompatible reuse budget/isolation')
            if d['config'].get('patch_weight')!=.35 or not d['config']['input_diversity']:raise RuntimeError('Incompatible reused objective')
            for rel,h in d['png_hashes'].items():
                if sha(OUT/rel)!=h:raise RuntimeError('Changed reused PNG')
            for clean,adv in zip(d['clean_paths'],d['adv_paths']):
                a=np.asarray(Image.open(OUT/clean)).astype('int16');b=np.asarray(Image.open(OUT/adv)).astype('int16')
                if np.abs(a-b).max()>16:raise RuntimeError('Reused epsilon violation')
            metric=OUT/'evaluations'/name/c['target'].replace('/','_')/tid/file.parent.name/'metrics.json'
            m=json.loads(metric.read_text())
            if m['method_lock_sha256']!=sha(source_lock):raise RuntimeError('Changed reused evaluation lock')
            if m['identity']!=fingerprint([d['identity'],d['png_hashes'],REGISTRY[c['target']],sha(source_lock)]):raise RuntimeError('Changed reused evaluation identity')
            for x in m['images']:
                if x['target_hit']!=(x['adv_label']==d['target_label']):raise RuntimeError('Wrong reused success numerator')
            evaluated += [x['image_id'] for x in m['images']]
        expected=[x['image_id'] for x in records(OUT/'manifests/test'/f'{tid}.jsonl')]
        if ids!=expected or evaluated!=expected:raise RuntimeError('Incomplete reused cohort/measurements')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();return
    require_recovery_complete();plan=matrix();ROOT.mkdir(parents=True,exist_ok=True)
    lock=dict(protocol='strict_family_disjoint',configs=[x['config'] for x in plan],allowed_test_conditions=[x['measurement_condition'] for x in plan],condition_targets={x['measurement_condition']:x['config']['target'] for x in plan},condition_models={x['measurement_condition']:x['config']['models'] for x in plan},targets=list(dict.fromkeys(x['config']['target'] for x in plan)),plan=plan,model_revisions=REGISTRY,manifest_hashes={t:sha(OUT/'manifests/test'/f'{t}.jsonl') for t in TIDS},source_hashes={p:sha(EXP/'src'/p) for p in ['run_proxy_selection.py','proxy_selection_tables.py','measure_selection_clean.py','run_v7_attack.py','vision_features.py','representation_objectives.py','family_protocol.py']},target_ranking_descriptive_only=True)
    path=ROOT/'method_lock.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise RuntimeError('Frozen proxy-selection matrix changed')
    if not path.exists():atomic_json(path,lock)
    prepare(plan)
    statepath=ROOT/'state.json';state=json.loads(statepath.read_text()) if statepath.exists() else dict(started=time.time(),attacks={},failures=[])
    from run_family_disjoint import validate_attack
    for item in plan:
        c=item['config']
        if item['source']!='new_subset_measurement':verify_reused(item);continue
        atomic_json(ROOT/'current_stage.json',dict(stage='attack',condition=c['id']))
        print('[proxy-selection] attack',c['id'],flush=True)
        try:
            cfg=ROOT/'configs'/f"{c['id']}.json";atomic_json(cfg,c['options'])
            command(EXP/'src/run_v7_attack.py',['--models',*c['models'],'--method','A3','--fusion','equal','--split','test','--steps',50,'--seed',42,'--restarts',1,'--batch-size',2,'--run-label',c['id'],'--config-json',cfg,'--excluded-proxy-family',REGISTRY[c['target']]['family'],'--input-diversity'],c['id']+'_attack.log')
            state['attacks'][c['id']]=validate_attack(c,OUT/'attacks/test'/condition_name(c))
        except Exception as exc:state['failures'].append(dict(stage='attack',id=c['id'],error=str(exc),traceback=traceback.format_exc()))
        atomic_json(statepath,state)
    for target in lock['targets']:
        for item in plan:
            c=item['config']
            if item['source']!='new_subset_measurement' or c['target']!=target or c['id'] not in state['attacks']:continue
            try:
                atomic_json(ROOT/'current_stage.json',dict(stage='evaluation',condition=c['id']))
                command(EXP/'src/evaluate_v7.py',['--root',OUT/'attacks/test'/condition_name(c),'--target',target,'--method-lock',path],c['id']+'_evaluation.log')
            except Exception as exc:state['failures'].append(dict(stage='evaluation',id=c['id'],error=str(exc)))
            atomic_json(statepath,state);raw_results(plan,lock)
    # Clean features are proxy-only post-hoc measurements; never model selection
    # by actual target results. Content-matched caches may already cover them.
    try:command(EXP/'src/measure_selection_clean.py',['--plan',ROOT/'planned_matrix.json'],'clean_features.log')
    except Exception as exc:state['failures'].append(dict(stage='clean_features',error=str(exc)))
    table=raw_results(plan,lock)
    complete=all(x['status']=='complete' for x in table)
    features=all(x['feature_status'] in ('complete','not_applicable_single','degenerate_features') for x in table)
    from archive_next_stage import model_audits,upload_compact
    bf16=model_audits(set(lock['targets'])|{n for c in lock['configs'] for n in c['models']},ROOT/'audits/bf16.json')
    if complete and features and bf16:state['finished']=time.time()
    atomic_json(statepath,state)
    atomic_json(ROOT/'completion.json',dict(all_complete=complete and features and bf16,all_ASR_complete=complete,conditions_complete=sum(x['status']=='complete' for x in table),conditions_expected=21,feature_complete=features,BF16=bf16,failed_attempts=state['failures']))
    atomic_json(ROOT/'audits/runtime.json',dict(started=state['started'],finished=state.get('finished'),wall_seconds=time.time()-state['started'],new_attack_seconds=sum(x['attack_seconds'] for x in state['attacks'].values()),reuse_runtime_is_reported_in_table_not_charged_as_new_work=True))
    upload_compact(ROOT)
    print('Proxy selection tables:',ROOT/'tables/proxy_selection.md',flush=True)
    if not complete or not features or not bf16:raise SystemExit(1)


if __name__=='__main__':main()
