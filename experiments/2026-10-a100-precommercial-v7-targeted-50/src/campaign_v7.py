#!/usr/bin/env python3
"""Long-running staged V7 campaign. All selection precedes final target access."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from v7_common import EXP, REPO, V6, OUT, BASE, TIDS, TARGETS, REGISTRY, prepare_data, atomic_json, fingerprint, sha
from run_v7_attack import METHODS

POOL=['CLIP ViT-L/14','SigLIP2-So400m-patch14-384','DINOv2-Large','Qwen3.5-2B','InternVL3.5-2B-HF','Gemma 4 E2B-it']
INITIAL=['CLIP ViT-L/14','Qwen3.5-2B','InternVL3.5-2B-HF']
VALIDATORS=['Qwen3.5-4B','InternVL3.5-4B-HF']


def command(script,args,log):
    log=OUT/'logs'/log;log.parent.mkdir(parents=True,exist_ok=True)
    print('[command]',script,' '.join(map(str,args)),flush=True)
    with log.open('a') as f:
        p=subprocess.Popen([sys.executable,str(script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        while p.poll() is None:time.sleep(2)
    if p.returncode:raise RuntimeError(f'Command exited {p.returncode}; see {log}')


def condition_name(cfg):
    s=f"{cfg['method']}__{cfg.get('fusion','equal')}__"+'+'.join(n.replace('/','_').replace(' ','_') for n in cfg['models'])
    if cfg.get('di'):s+='__DI'
    if cfg.get('ti'):s+='__TI'
    return s+f"__s{cfg.get('seed',42)}__i{cfg.get('steps',50)}__r{cfg.get('restarts',1)}"


def attack(cfg,split,smoke=False):
    batch=2
    perf=OUT/'audits/performance_benchmark.json'
    if perf.is_file():
        audited={r['model']:r for r in json.loads(perf.read_text())['results']}
        safe=[]
        for name in cfg['models']:
            safe.append(max((r['batch'] for r in audited.get(name,{}).get('trials',[]) if r.get('safe') and r.get('peak_reserved_gib',100)<55),default=2))
        batch=min(30,min(safe))
    args=['--models',*cfg['models'],'--method',cfg['method'],'--fusion',cfg.get('fusion','equal'),'--split',split,'--steps',2 if smoke else cfg.get('steps',50),'--seed',cfg.get('seed',42),'--restarts',cfg.get('restarts',1),'--batch-size',os.environ.get('V7_BATCH_SIZE',str(batch))]
    if smoke:args+=['--transitions','T01']
    if cfg.get('di'):args+=['--input-diversity']
    if cfg.get('ti'):args+=['--smoothing']
    command(EXP/'src/run_v7_attack.py',args,f'{split}_{condition_name(cfg)}.log')
    return OUT/'attacks'/split/condition_name({**cfg,'steps':2} if smoke else cfg)


def available(name):
    """Require the pinned config AND all declared checkpoint shards before a load."""
    from primary_ml_cka.models.common.loading import local_snapshot
    try:
        spec=REGISTRY[name];path=local_snapshot(Path(os.environ['HF_HOME']),spec['repo_id'],spec['revision'])
        if not (path/'config.json').is_file():return False
        index=path/'model.safetensors.index.json'
        if index.exists():return all((path/f).is_file() for f in set(json.loads(index.read_text())['weight_map'].values()))
        return (path/'model.safetensors').is_file() or (path/'pytorch_model.bin').is_file()
    except (FileNotFoundError,ValueError,RuntimeError):return False


def ensure_models(names):
    for name in names:
        if available(name):continue
        # Independent download errors do not prevent other available model experiments.
        command(V6/'src/download_models.py',['--models',name,'--output-dir',OUT],f'download_{name.replace("/","_")}.log')
        if not available(name):raise RuntimeError(f'Missing complete pinned snapshot: {name}')


def evaluate_dev(root):
    ensure_models(VALIDATORS)
    command(EXP/'src/evaluate_v7.py',['--root',root,'--validators',*VALIDATORS],f'dev_eval_{root.name}.log')


def dev_score(cfg):
    files=list((OUT/'development_evaluations'/condition_name(cfg)).glob('*/T*/batch_*/metrics.json'))
    rows=[];directions={};models=set()
    for p in files:
        d=json.loads(p.read_text());models.add(d['validator']);tid=p.parents[1].name
        for x in d['images']:rows.append(x);directions.setdefault((d['validator'],tid),[]).append(x)
    if models!=set(VALIDATORS) or len(directions)!=10 or any(len(v)!=10 for v in directions.values()):raise RuntimeError(f'Incomplete dev metrics for {condition_name(cfg)}')
    # Worst held-out-surrogate/direction representation hit rate, then overall hit rate,
    # then mean target-vs-hardest-rival margin. No real-target output is available here.
    return (min(sum(r['representation_hit'] for r in v)/len(v) for v in directions.values()),sum(r['representation_hit'] for r in rows)/len(rows),sum(r['target_margin'] for r in rows)/len(rows))


def aggregate():
    entries=[]
    for p in (OUT/'evaluations').glob('*/*/T*/batch_*/metrics.json'):
        d=json.loads(p.read_text());st=json.loads((OUT/d['source_state']).read_text())
        for r in d['images']:entries.append(dict(condition=p.parents[3].name,target=d['target'],transition=st['transition_id'],seed=st['seed'],**r))
    (OUT/'reports').mkdir(parents=True,exist_ok=True)
    if entries:
        with (OUT/'reports/per_image_target_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    groups={}
    for r in entries:groups.setdefault((r['condition'],r['target'],r['transition']),[]).append(r)
    summary=[]
    for (condition,target,tid),rows in groups.items():
        n=len(rows);hits=sum(r['target_hit'] for r in rows);clean=sum(r['clean_correct'] for r in rows);ch=sum(r['target_hit'] and r['clean_correct'] for r in rows)
        summary.append(dict(condition=condition,target=target,transition=tid,N=n,hits=hits,ASR=hits/n,clean_correct=clean,clean_correct_hits=ch,conditional_ASR=ch/clean if clean else None,complete=n==30,passes_goal=n==30 and hits>=15))
    if summary:
        with (OUT/'reports/target_direction_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    failures=json.loads((OUT/'audits/campaign_failures.json').read_text()) if (OUT/'audits/campaign_failures.json').exists() else []
    report=['# V7 live progress report','','Status: IN PROGRESS until every predeclared cell, analysis and validation passes.','',f'Evaluated target-direction conditions: {len(summary)}. Complete: {sum(r["complete"] for r in summary)}.',f'Recorded failed stage attempts: {len(failures)}.','','Primary metric: unconditional targeted ASR on exactly 30 locked ground-truth images. Undefined conditional metrics are N/A.','', 'Method selection uses disjoint dev images and held-out surrogate models only. Target feedback does not enter selection.','', '| Condition | Target | Direction | Hits/N | ASR |','|---|---|---|---|---|']
    report += [f'| {r["condition"]} | {r["target"]} | {r["transition"]} | {r["hits"]}/{r["N"]} | {r["ASR"]:.3f} |' for r in summary]
    (OUT/'reports/final_report.md').write_text('\n'.join(report)+'\n')


def execute_frozen_tests(tests, task):
    for cfg in tests:
        def generate(c=cfg):
            ensure_models(c['models']);return attack(c,'test')
        task('test-attacks '+condition_name(cfg),generate)
    for target in TARGETS:
        if task('target-download '+target,lambda n=target:ensure_models([n])) is None:continue
        for cfg in tests:
            root=OUT/'attacks/test'/condition_name(cfg)
            task('target-eval '+target+' '+root.name,lambda r=root,n=target:command(EXP/'src/evaluate_v7.py',['--root',r,'--target',n],f'target_{n}_{r.name}.log'))
    aggregate()
    # Completion includes explicitly checked expected cells, never just process exit status.
    expected=len(tests)*len(TARGETS)*len(TIDS)
    with (OUT/'reports/target_direction_results.csv').open() as f:done=list(csv.DictReader(f))
    complete=len(done)==expected and all(r['complete']=='True' for r in done)
    atomic_json(OUT/'audits/execution_completion.json',dict(expected_cells=expected,observed_cells=len(done),all_complete=complete,failed_attempts=len(failures)))
    if not complete:raise RuntimeError(f'Campaign incomplete: {len(done)}/{expected} cells; see failure audit')
    command(EXP/'src/finalize_v7.py',[],'final_validation.log')
    command(EXP/'src/analyze_v7.py',[],'posthoc_analysis.log')
    command(EXP/'src/finalize_v7.py',[],'final_validation.log')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','smoke','pilot','full'],default='full');p.add_argument('--resume',action='store_true')
    a=p.parse_args();prepare_data();failures=[]
    def task(label,fn):
        atomic_json(OUT/'audits/current_stage.json',dict(stage=label,started=time.time()))
        print(f'[stage] {label}',flush=True)
        try:
            value=fn();aggregate();return value
        except Exception as exc:
            failures.append(dict(stage=label,error=str(exc),time=time.time(),traceback=traceback.format_exc()))
            atomic_json(OUT/'audits/campaign_failures.json',failures);aggregate();print(f'[failed-local] {label}: {exc}',flush=True);return None
    if not torch_preflight():raise RuntimeError('CUDA/BF16 global blocker')
    subprocess.run([sys.executable,'-m','pytest','-q',str(EXP/'tests')],check=True)
    atomic_json(OUT/'audits/preflight.json',dict(status='pass',timestamp=time.time(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()))
    if a.mode=='preflight':return
    lock_path=OUT/'audits/method_lock.json'
    if a.mode=='full' and lock_path.is_file():
        frozen=json.loads(lock_path.read_text())
        execute_frozen_tests(frozen['test_conditions'],task)
        return
    ensure_models(INITIAL)
    command(EXP/'src/diagnose_v7.py',['--models',*INITIAL],'diagnostics.log')
    command(EXP/'src/calibrate_v7.py',['--models',*INITIAL],'calibration.log')
    smoke_cfgs=[dict(models=INITIAL,method=x) for x in ['A0','A1','A2','A3','A4']]+[dict(models=INITIAL,method='A1',fusion=x) for x in ['weighted','pcgrad']]+[dict(models=INITIAL,method='A1',di=True)]
    for cfg in smoke_cfgs:
        if task('smoke '+condition_name(cfg),lambda c=cfg:attack(c,'smoke',True)) is None:raise RuntimeError('Smoke failed; full campaign held')
    # Verify completed-batch resumption without modifying PNGs.
    root=attack(smoke_cfgs[0],'smoke',True);before={str(p):sha(p) for p in root.rglob('*adv.png')};attack(smoke_cfgs[0],'smoke',True)
    assert before=={str(p):sha(p) for p in root.rglob('*adv.png')}
    atomic_json(OUT/'audits/resume_smoke.json',dict(pass_=True,artifacts=before))
    if a.mode=='smoke':return
    # Ordered objectives first, fusion/input-search second. All arms are predeclared.
    loss_configs=[dict(models=INITIAL,method=m) for m in METHODS]
    evaluated=[]
    for cfg in loss_configs:
        def run(c=cfg):
            root=attack(c,'dev');evaluate_dev(root);return c
        result=task('loss-dev '+condition_name(cfg),run)
        if result is not None:evaluated.append(result)
    if not evaluated:raise RuntimeError('No complete surrogate dev loss experiment')
    best_loss=max(evaluated,key=dev_score)['method']
    if a.mode=='pilot':return
    # Best loss is determined before fusion study, never from final targets.
    candidates=[]
    for names in [INITIAL]+[[n] for n in POOL]+[['Qwen3.5-2B','InternVL3.5-2B-HF'],['CLIP ViT-L/14','SigLIP2-So400m-patch14-384']]:
        candidates.append(dict(models=names,method=best_loss))
    for fusion in ['weighted','pcgrad','weighted_pcgrad','consensus']:candidates.append(dict(models=INITIAL,method=best_loss,fusion=fusion))
    candidates += [dict(models=INITIAL,method=best_loss,di=True),dict(models=INITIAL,method=best_loss,ti=True),dict(models=INITIAL,method=best_loss,restarts=3)]
    for cfg in candidates:
        def run(c=cfg):
            ensure_models(c['models']);root=attack(c,'dev');evaluate_dev(root);return c
        result=task('fusion-dev '+condition_name(cfg),run)
        if result is not None:evaluated.append(result)
    finalist=max((c for c in evaluated if len(c['models'])>1),key=dev_score)
    # Finalist & matched individual proxies & original equal ensemble, seeds predeclared.
    tests=[{**finalist,'seed':s} for s in [42,43,44]]
    tests+=[dict(models=[n],method=finalist['method']) for n in set(finalist['models']+INITIAL)]
    tests+=[dict(models=[n],method='A0') for n in set(finalist['models']+INITIAL)]
    tests += loss_configs
    tests += [c for c in candidates if len(c['models'])>1 and c.get('restarts',1)==1]
    tests+=[dict(models=INITIAL,method='A0'),dict(models=INITIAL,method=best_loss)]
    unique={condition_name(c):c for c in tests};tests=list(unique.values())
    lock=dict(selection='held_out_surrogate_representation_metrics',validators=VALIDATORS,dev_manifest_hashes={t:sha(OUT/'manifests/dev'/f'{t}.jsonl') for t in TIDS},selected=finalist,scores={condition_name(c):dev_score(c) for c in evaluated},allowed_test_conditions=list(unique),test_conditions=tests,targets=TARGETS,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    lock_path=OUT/'audits/method_lock.json'
    if lock_path.exists():
        old=json.loads(lock_path.read_text())
        if old!=lock:raise RuntimeError('Refusing to revise method lock after target access')
    else:atomic_json(lock_path,lock)
    execute_frozen_tests(tests,task)


def torch_preflight():
    import torch
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

if __name__=='__main__':main()
