#!/usr/bin/env python3
"""Budgeted loss/composition research; final targets accessed only after locking."""
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from v7_common import OUT,EXP,TARGETS,TIDS,REGISTRY,atomic_json,sha,records,fingerprint,prepare_data
from focus_search import launch,score,INITIAL,VALIDATORS
from campaign_v7 import ensure_models

R=OUT/'ten_hour_research'


def savecsv(name,rows):
    if not rows:return
    p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    R.mkdir(parents=True,exist_ok=True);prepare_data()
    clock=R/'budget.json'
    if not clock.exists():atomic_json(clock,dict(started=time.time(),deadline=time.time()+36000,hours=10))
    budget=json.loads(clock.read_text());deadline=budget['deadline'];search_end=deadline-5400
    history=R/'state.json';state=json.loads(history.read_text()) if history.exists() else dict(results=[],failures=[],skipped=[])
    results=state['results'];failures=state['failures'];done={r['config']['id'] for r in results}
    def write_report():
        rows=[]
        for item in results:
            c=item['config'];rows.append(dict(condition=c['id'],family=c.get('family','unknown'),models='+'.join(c['models']),method=c.get('method','A3'),options=json.dumps(c.get('options',{}),sort_keys=True),fusion=c.get('fusion','equal'),seed=c.get('seed',42),steps=c.get('steps',50),restarts=c.get('restarts',1),N_per_direction=c.get('n_per_direction',2),minimum_native_ASR=item['score'][0],pooled_native_ASR=item['score'][1],seconds=item['seconds'],root=item['root']))
        savecsv('development_results.csv',rows)
        for family,filename in [('loss','loss_comparison.csv'),('combination','loss_combination_ablation.csv'),('composition','proxy_composition_comparison.csv'),('optimizer','optimizer_comparison.csv')]:savecsv(filename,[r for r in rows if r['family']==family])
        raw=[];compute=[]
        for item in results:
            root=Path(item['root'])
            for p in (OUT/'focused_optimization/native_validation').glob('*/'+root.name+'/*.json'):
                d=json.loads(p.read_text())
                for row in d['images']:raw.append(dict(condition=item['config']['id'],model=d['model'],**row))
            for p in root.glob('T*/batch_*/state.json'):
                d=json.loads(p.read_text());compute.append(dict(condition=item['config']['id'],transition=d['transition_id'],images=len(d['image_ids']),steps=d['steps'],seed=d['seed'],restarts=d['restarts'],seconds=d['runtime_seconds'],peak_reserved_gib=d['peak_reserved_gib'],counts=json.dumps(d['forward_counts'])))
        savecsv('per_image_results.csv',raw);savecsv('compute_budget.csv',compute)
        lines=['# Ten-hour V7 research progress','',f"Completed research conditions: {len(results)}. Failed attempts: {len(failures)}. Remaining wall time: {max(0,(deadline-time.time())/3600):.2f} h.",'', 'Development uses actual native outputs of held-out Qwen9B, InternVL8B and Gemma E4B. These are not final Qwen27B/InternVL14B/Gemma31B results. Initial pilot is ten GT-selected images, two per direction; larger development confirmation is explicitly marked. No prediction-based screening.','', '| Condition | Worst model-direction ASR | Pooled native ASR | Images/direction |','|---|---|---|---|']
        lines += [f"| {r['condition']} | {r['minimum_native_ASR']:.1%} | {r['pooled_native_ASR']:.1%} | {r['N_per_direction']} |" for r in rows]
        lines += ['', 'A high two-image pilot is not evidence for the formal >=15/30 criterion. Undefined conditional TASR is N/A. All added EOT/variance/restart computation must be interpreted with compute_budget.csv. Failures and budget-skipped conditions are retained.']
        (R/'final_report.md').write_text('\n'.join(lines)+'\n')
        atomic_json(history,state);atomic_json(R/'failures.json',failures)
    def run(script,args,log,cutoff):
        if time.time()>=cutoff:raise TimeoutError('Budget exhausted before starting job')
        path=R/'logs'/log;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            proc=subprocess.Popen([sys.executable,str(script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=max(1,cutoff-time.time()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise TimeoutError('Research deadline; completed batches retained')
        if proc.returncode:raise RuntimeError(f'Exit {proc.returncode}: {path}')
    current_cutoff=[search_end]
    import focus_search
    focus_search.command=lambda script,args,log:run(script,args,'attack_'+log,current_cutoff[0])
    def trial(cfg,cutoff=search_end):
        current_cutoff[0]=cutoff
        if cfg['id'] in done:return next(r for r in results if r['config']['id']==cfg['id'])
        if time.time()>cutoff-180:
            state['skipped'].append(dict(config=cfg,reason='insufficient time before phase deadline'));write_report();return None
        atomic_json(R/'current_stage.json',dict(config=cfg,started=time.time(),remaining_hours=(deadline-time.time())/3600))
        print('[ten-hour]',cfg['id'],cfg['models'],cfg.get('options',{}),flush=True);start=time.time()
        try:
            # launch() uses the known audited runner; subprocess-scoped watchdog
            # is enforced at the campaign level and native-validation boundary.
            root=launch(cfg)
            run(EXP/'src/evaluate_surrogate_outputs.py',['--root',root,'--models',*VALIDATORS,'--limit-per-direction',cfg.get('n_per_direction',2)],cfg['id']+'_native.log',cutoff)
            rank=score(root);item=dict(config=cfg,root=str(root),score=rank,seconds=time.time()-start)
            results.append(item);done.add(cfg['id']);print('[native-result]',cfg['id'],rank,flush=True);return item
        except Exception as exc:
            failures.append(dict(config=cfg,error=str(exc),traceback=traceback.format_exc()));print('[local-failure]',cfg['id'],exc,flush=True)
        finally:write_report()
    # Existing frozen focus pilots are valid initial evidence; never regenerate them.
    if not results:
        old=OUT/'focused_optimization/search_state.json'
        if old.exists():
            for item in json.loads(old.read_text())['results']:
                if item['config']['id'] not in done:
                    results.append({**item,'seconds':0.,'config':{**item['config'],'family':'loss','reused_existing_result':True}});done.add(item['config']['id'])
        write_report()
    lock_path=R/'selected_method_lock.json'
    if not lock_path.exists():
        if list((OUT/'evaluations').glob('*/*/T*/batch_*/metrics.json')) or (OUT/'focused_optimization/audits/final_target_access.json').exists():raise RuntimeError('Final targets already accessed; adaptive tuning prohibited')
        atomic_json(R/'experiment_plan.json',dict(budget=budget,validators=VALIDATORS,targets=TARGETS,selection='worst native surrogate direction ASR, pooled native ASR; cost tie break',initial_images_per_direction=2,reference_count=48,parameter_dtype='bfloat16',final_target_feedback=False))
        losses=[('L00','A0',{}),('L01','A1',{}),('L02','A2',{}),('L03','A3',{}),('L04','A4',{}),('L05','A3',{'projector_weight':.35}),('L06','A3',{'feature_transform':'center'}),('L07','A3',{'temperature':.005,'margin':.005,'negative_mode':'hardest'})]
        for id,method,options in losses:trial(dict(id='T10_'+id,models=INITIAL,method=method,options=options,family='loss'))
        def best(pool=None):return max(pool if pool is not None else [r for r in results if len(r['config']['models'])>1],key=lambda r:(*r['score'],-r['seconds']))
        if not results:raise RuntimeError('No valid research conditions')
        champion=best();base=champion['config'];baseopt=base.get('options',{})
        mixtures=[{'loss':'combined','margin_weight':w} for w in [.25,.5,.75]]+[{'patch_weight':w} for w in [.1,.35]]+[{'projector_weight':w} for w in [.25,.75]]
        for i,extra in enumerate(mixtures):trial(dict(id=f'T10_M{i:02d}',models=INITIAL,method='A3',options={**baseopt,**extra},family='combination'))
        champion=best();baseopt=champion['config'].get('options',{});method=champion['config'].get('method','A3')
        pools=[['Qwen3.5-2B','InternVL3.5-2B-HF'],['CLIP ViT-L/14','InternVL3.5-2B-HF','Gemma 4 E2B-it'],['SigLIP2-So400m-patch14-384','Qwen3.5-2B','InternVL3.5-2B-HF'],['CLIP ViT-L/14','Qwen3.5-4B','InternVL3.5-4B-HF'],['CLIP ViT-L/14','SigLIP2-So400m-patch14-384','InternVL3.5-2B-HF','Gemma 4 E2B-it'],['CLIP ViT-L/14','DINOv2-Large','InternVL3.5-4B-HF'],['CLIP ViT-L/14','Qwen3.5-2B','Gemma 4 26B-A4B-it'],['SigLIP2-So400m-patch14-384','InternVL3.5-4B-HF','Gemma 4 26B-A4B-it']]
        for i,names in enumerate(pools):trial(dict(id=f'T10_C{i:02d}',models=names,method=method,options=baseopt,family='composition'))
        champion=best();base=champion['config'];baseopt=base.get('options',{})
        for n in base['models']:trial(dict(id='T10_single_'+fingerprint(n)[:8],models=[n],method=base.get('method','A3'),options=baseopt,family='composition'))
        variants=[('equal',{}),('weighted',{'adaptive_temperature':.005}),('pcgrad',{}),('weighted_pcgrad',{'adaptive_temperature':.005}),('equal',{'nesterov':True}),('equal',{'variance_samples':2}),('equal',{'frequency_rho':.15,'eot_samples':2})]
        for i,(fusion,extra) in enumerate(variants):trial(dict(id=f'T10_O{i:02d}',models=base['models'],method=base.get('method','A3'),options={**baseopt,**extra},fusion=fusion,family='optimizer'))
        trial(dict(id='T10_O_noDI',models=base['models'],method=base.get('method','A3'),options=baseopt,di=False,family='optimizer'))
        trial(dict(id='T10_O_TI',models=base['models'],method=base.get('method','A3'),options=baseopt,ti=True,family='optimizer'))
        # Predeclared refinement queue; spend remaining development budget on real
        # alternatives, not repeated identical runs or target-result-guided search.
        extras=list(itertools.product([.005,.02,.05,.1],[.0,.02,.05],['hardest','logsumexp']))
        for i,(temperature,margin,negative) in enumerate(extras):
            if time.time()>search_end-1500:break
            champion=best();c=champion['config']
            trial(dict(id=f'T10_R{i:02d}',models=c['models'],method=c.get('method','A3'),options={**c.get('options',{}),'temperature':temperature,'margin':margin,'negative_mode':negative},fusion=c.get('fusion','equal'),family='loss'))
        # Confirm only top two on nested 8-per-direction dev sets, with overlap disclosed.
        ranked=sorted([r for r in results if len(r['config']['models'])>1],key=lambda r:tuple(r['score']),reverse=True)
        confirmed=[]
        for i,item in enumerate(ranked[:2]):
            cfg={**item['config'],'id':f'T10_confirm_{i}','n_per_direction':8,'family':'confirmation'}
            result=trial(cfg,cutoff=deadline-4500)
            if result:confirmed.append(result)
        champion=best(confirmed) if confirmed else best()
        for seed in [43,44]:trial({**champion['config'],'id':f'T10_seed_{seed}','seed':seed,'n_per_direction':2,'family':'seed_confirmation'},cutoff=deadline-4200)
        # Selection is frozen here. Subsequent final target outputs cannot revise it.
        singles=[x for x in results if len(x['config']['models'])==1]
        selected=[champion]
        if singles:selected.append(best(singles))
        configs=[{**x['config'],'id':f'T10_final_{i}','n_per_direction':30,'seed':42} for i,x in enumerate(selected)]
        configs.append(dict(id='T10_final_pullpush',models=champion['config']['models'],method='A0',options={},n_per_direction=30,di=False))
        allowed=[]
        for c in configs:
            name='focus_'+c['id']+'__'+c.get('method','A3')+'__'+c.get('fusion','equal')+'__'+'+'.join(n.replace('/','_').replace(' ','_') for n in c['models'])
            if c.get('di',True):name+='__DI'
            if c.get('ti'):name+='__TI'
            name+=f"__s42__i{c.get('steps',50)}__r{c.get('restarts',1)}";allowed.append(name)
        atomic_json(lock_path,dict(selected=champion,configs=configs,allowed_test_conditions=allowed,targets=TARGETS,selection='held-out surrogate native outputs only',final_target_feedback=False,validation_confirmation_is_nested=True,test_manifest_hashes={t:sha(OUT/'manifests/test'/f'{t}.jsonl') for t in TIDS}))
    lock=json.loads(lock_path.read_text())
    # Final generation uses the original 30-image manifests; no small-sample shortcut.
    for cfg,name in zip(lock['configs'],lock['allowed_test_conditions']):
        try:
            cp=R/'configs'/f"{cfg['id']}.json";atomic_json(cp,cfg.get('options',{}));ensure_models(cfg['models'])
            args=['--models',*cfg['models'],'--method',cfg.get('method','A3'),'--fusion',cfg.get('fusion','equal'),'--split','test','--steps',cfg.get('steps',50),'--seed',42,'--batch-size',2,'--run-label','focus_'+cfg['id'],'--config-json',cp]
            if cfg.get('di',True):args+=['--input-diversity']
            if cfg.get('ti'):args+=['--smoothing']
            run(EXP/'src/run_v7_attack.py',args,cfg['id']+'_test.log',deadline-1800)
        except Exception as exc:failures.append(dict(condition=cfg['id'],stage='final_attack',error=str(exc)))
    atomic_json(R/'final_target_access.json',dict(method_lock_sha256=sha(lock_path),search_closed=True))
    for target in TARGETS:
        try:ensure_models([target])
        except Exception as exc:failures.append(dict(model=target,stage='download',error=str(exc)));continue
        for name in lock['allowed_test_conditions']:
            try:run(EXP/'src/evaluate_v7.py',['--root',OUT/'attacks/test'/name,'--target',target,'--method-lock',lock_path],fingerprint([target,name])[:16]+'.log',deadline-120)
            except Exception as exc:failures.append(dict(model=target,condition=name,stage='final_evaluation',error=str(exc)))
    write_report()
    target_rows=[]
    for name in lock['allowed_test_conditions']:
        for target in TARGETS:
            for tid in TIDS:
                rows=[]
                for p in (OUT/'evaluations'/name/target.replace('/','_')/tid).glob('batch_*/metrics.json'):rows+=json.loads(p.read_text())['images']
                n=len(rows);hits=sum(r['target_hit'] for r in rows)
                target_rows.append(dict(condition=name,target=target,transition=tid,hits=hits,denominator=n,ASR=hits/n if n else None,status='complete' if n==30 else 'missing_or_partial'))
    savecsv('final_target_results.csv',target_rows)
    with (R/'final_report.md').open('a') as f:
        f.write('\n## Frozen final target evaluation\n\n| Condition | Model | Direction | Hits/N |\n|---|---|---|---|\n')
        for r in target_rows:f.write(f"| {r['condition']} | {r['target']} | {r['transition']} | {r['hits']}/{r['denominator']} |\n")
        f.write('\nMissing/partial conditions are not zeros. Nested dev confirmation overlaps the initial pilot and is not an untouched generalization set. The final test cohort is independent and unchanged.\n')
    atomic_json(R/'completion.json',dict(finished=time.time(),budget=budget,completed_research_conditions=len(results),failed_attempts=len(failures),complete_target_cells=sum(r['status']=='complete' for r in target_rows),expected_target_cells=len(target_rows),all_complete=all(r['status']=='complete' for r in target_rows)))
    print('[ten-hour] budgeted campaign finished; inspect',R/'final_report.md',flush=True)

if __name__=='__main__':main()
