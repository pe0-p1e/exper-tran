#!/usr/bin/env python3
"""Small-cohort algorithm/composition research before any further broad ablation."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from v7_common import OUT,EXP,REGISTRY,TIDS,TARGETS,records,fingerprint,atomic_json,prepare_data,sha
from campaign_v7 import ensure_models

VALIDATORS=['Qwen3.5-9B','InternVL3.5-8B-HF','Gemma 4 E4B-it']
INITIAL=['CLIP ViT-L/14','Qwen3.5-2B','InternVL3.5-2B-HF']
FOCUS=OUT/'focused_optimization'


def command(script,args,log):
    path=OUT/'logs/focus'/log;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        result=subprocess.run([sys.executable,str(script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Exit {result.returncode}: {path}')


def launch(cfg, smoke=False):
    method=cfg.get('method','A3');options=cfg.get('options',{})
    label='focus_'+cfg['id']+('_smoke' if smoke else '')
    cp=FOCUS/'configs'/f'{label}.json'
    if cp.exists() and json.loads(cp.read_text())!=options:raise RuntimeError('Never overwrite an existing research condition')
    atomic_json(cp,options)
    args=['--models',*cfg['models'],'--method',method,'--fusion',cfg.get('fusion','equal'),'--split','dev','--limit-per-direction',cfg.get('n_per_direction',2),'--steps',2 if smoke else cfg.get('steps',50),'--seed',cfg.get('seed',42),'--restarts',cfg.get('restarts',1),'--batch-size',2,'--run-label',label,'--config-json',cp]
    if smoke:args+=['--transitions','T01']
    if cfg.get('di',True):args+=['--input-diversity']
    if cfg.get('ti'):args+=['--smoothing']
    ensure_models(cfg['models'])
    command(EXP/'src/run_v7_attack.py',args,label+'.log')
    name=f"{label}__{method}__{cfg.get('fusion','equal')}__"+'+'.join(n.replace('/','_').replace(' ','_') for n in cfg['models'])
    if cfg.get('di',True):name+='__DI'
    if cfg.get('ti'):name+='__TI'
    name+=f"__s{cfg.get('seed',42)}__i{2 if smoke else cfg.get('steps',50)}__r{cfg.get('restarts',1)}"
    return OUT/'attacks/dev'/name


def evaluate(root):
    ensure_models(VALIDATORS)
    command(EXP/'src/evaluate_surrogate_outputs.py',['--root',root,'--models',*VALIDATORS,'--limit-per-direction',2],root.name+'_native.log')


def score(root):
    files=list((FOCUS/'native_validation').glob('*/'+root.name+'/*.json'))
    ds=[json.loads(p.read_text()) for p in files]
    if set(d['model'] for d in ds)!=set(VALIDATORS):raise RuntimeError('Missing independent surrogate native outputs')
    fractions=[c['hits']/c['denominator'] for d in ds for c in d['directions']]
    if len(fractions)!=15:raise RuntimeError('Incomplete directions')
    return (min(fractions),sum(d['hits'] for d in ds)/sum(d['denominator'] for d in ds))


def report(results,failed):
    rows=[]
    for item in results:
        s=score(Path(item['root']));rows.append(dict(condition=item['config']['id'],models='+'.join(item['config']['models']),options=json.dumps(item['config'].get('options',{}),sort_keys=True),minimum_surrogate_direction_ASR=s[0],pooled_surrogate_native_ASR=s[1],images_per_model=10,root=item['root']))
    dst=FOCUS/'reports';dst.mkdir(parents=True,exist_ok=True)
    if rows:
        with (dst/'native_pilot_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Focused V7 pilot research','','Status: ongoing small-cohort optimization. Five directions x two independent development images = 10 images/model.','', 'Native generated output is evaluated on held-out Qwen9B, InternVL8B and Gemma E4B. This is NOT final Qwen27B/InternVL14B/Gemma31B performance. Cohorts are ground-truth-only; no samples removed.','', '| Condition | Minimum direction ASR | Pooled native ASR |','|---|---|---|']
    for r in rows:lines.append(f"| {r['condition']} | {r['minimum_surrogate_direction_ASR']:.1%} | {r['pooled_surrogate_native_ASR']:.1%} |")
    lines+=['',f'Failed research conditions: {len(failed)}. Every failure is retained in failures.json.','', 'N=2/direction is only a pilot. Native surrogate ASR and reference-feature class hits are distinct; no official 30-image result or legacy conditional TASR is implied. More computation is explicitly recorded; EOT and variance tuning add forward/backward calls.','', 'Final target feedback is prohibited during this search. No large ablation or full final-target matrix is launched automatically by the focused pilot.']
    (dst/'progress.md').write_text('\n'.join(lines)+'\n')


def main():
    prepare_data();FOCUS.mkdir(parents=True,exist_ok=True)
    # Earlier final-target lock and attacks remain immutable; no target output exists.
    if (FOCUS/'final_pilot_lock.json').is_file():
        from evaluate_focus_targets import main as final_pilot
        final_pilot()
        return
    if list((OUT/'evaluations').glob('*/*/T*/batch_*/metrics.json')):
        raise RuntimeError('Final targets have been accessed: stop adaptive development and audit isolation before a new study')
    history=FOCUS/'search_state.json'
    prior=json.loads(history.read_text()) if history.exists() else dict(results=[],failed=[])
    results=prior['results'];failed=prior['failed'];done={r['config']['id'] for r in results}
    def trial(cfg):
        if cfg['id'] in done:return next(r for r in results if r['config']['id']==cfg['id'])
        print('[focused-pilot]',cfg['id'],cfg['models'],cfg.get('options',{}),flush=True)
        atomic_json(FOCUS/'current_stage.json',dict(condition=cfg,started=time.time(),images_per_model=10))
        try:
            root=launch(cfg);evaluate(root);rank=score(root)
            item=dict(config=cfg,root=str(root),score=rank);results.append(item);done.add(cfg['id'])
            print('[native-score]',cfg['id'],rank,flush=True)
            return item
        except Exception as exc:
            failed.append(dict(config=cfg,error=str(exc),traceback=traceback.format_exc()))
            atomic_json(FOCUS/'failures.json',failed);print('[local-failure]',cfg['id'],str(exc),flush=True)
        finally:
            atomic_json(history,dict(results=results,failed=failed));report(results,failed)
    # Audit new differentiable mechanisms on 2 images x 2 steps before 50 steps.
    for sid,options in [('center',{'feature_transform':'center'}),('projector',{'projector_weight':.35}),('spectral',{'frequency_rho':.15,'eot_samples':2}),('variance',{'variance_samples':2}),('nesterov',{'nesterov':True})]:
        marker=FOCUS/'smoke'/f'{sid}.json'
        signature=fingerprint([options,sha(EXP/'src/run_v7_attack.py'),sha(EXP/'src/vision_features.py')])
        if marker.exists() and json.loads(marker.read_text()).get('signature')==signature:continue
        root=launch(dict(id='smoke_'+sid,models=INITIAL,options=options),smoke=True)
        states=list(root.glob('T*/batch_*/state.json'))
        if len(states)!=1:raise RuntimeError('Missing focused smoke')
        state=json.loads(states[0].read_text())
        if state['linf_png']>16/255+1e-7:raise RuntimeError('Smoke epsilon failure')
        atomic_json(marker,dict(signature=signature,status='pass',state=str(states[0])))
    objectives=[
      ('P00',{}),
      ('P01',{'feature_transform':'center'}),
      ('P02',{'feature_transform':'standardize'}),
      ('P03',{'temperature':.005,'margin':.005,'negative_mode':'hardest'}),
      ('P04',{'aggregation':'soft_reference','reference_temperature':.005,'temperature':.01}),
      ('P05',{'projector_weight':.35}),
      ('P06',{'layer_weights':[.1,.2,.3,.4]}),
    ]
    for id,options in objectives:trial(dict(id=id,models=INITIAL,options=options))
    if not results:raise RuntimeError('No valid native-output pilot')
    best=max(results,key=lambda r:tuple(r['score']))
    options=dict(best['config'].get('options',{}))
    # Different complementarity hypotheses, all evaluated on exactly the same images.
    compositions=[
      ['CLIP ViT-L/14','InternVL3.5-2B-HF','Gemma 4 E2B-it'],
      ['SigLIP2-So400m-patch14-384','Qwen3.5-2B','InternVL3.5-2B-HF'],
      ['CLIP ViT-L/14','Qwen3.5-4B','InternVL3.5-4B-HF'],
      ['Qwen3.5-4B','InternVL3.5-2B-HF','Gemma 4 E2B-it'],
      ['CLIP ViT-L/14','SigLIP2-So400m-patch14-384','InternVL3.5-2B-HF','Gemma 4 E2B-it'],
      ['CLIP ViT-L/14','DINOv2-Large','InternVL3.5-4B-HF'],
      ['CLIP ViT-L/14','Qwen3.5-2B','Gemma 4 26B-A4B-it'],
      ['SigLIP2-So400m-patch14-384','InternVL3.5-4B-HF','Gemma 4 26B-A4B-it'],
      ['Qwen3.5-4B','InternVL3.5-4B-HF'],
    ]
    for i,names in enumerate(compositions,1):trial(dict(id=f'C{i:02d}',models=names,options=options))
    best=max(results,key=lambda r:tuple(r['score']));names=best['config']['models'];options=best['config'].get('options',{})
    optimizers=[('O01',{'nesterov':True}),('O02',{'frequency_rho':.15,'eot_samples':2}),('O03',{'variance_samples':2})]
    for id,extra in optimizers:trial(dict(id=id,models=names,options={**options,**extra}))
    for mode in ['weighted','pcgrad','weighted_pcgrad']:
        trial(dict(id='G_'+mode,models=names,options={**options,'adaptive_temperature':.005},fusion=mode))
    best=max(results,key=lambda r:tuple(r['score']))
    atomic_json(FOCUS/'best_surrogate_pilot.json',dict(**best,final_target_feedback=False,small_sample_only=True,final_target_50pct_goal_achieved=False))
    # User requested optimization first. Small ablations are gated on actual native
    # validation success, never on representation-only metrics.
    passed=best['score'][0]>=.5 and best['score'][1]>=.9
    atomic_json(FOCUS/'research_status.json',dict(native_pilot_gate_passed=passed,conditions_completed=len(results),conditions_failed=len(failed),large_ablation_deferred=True,final_target_evaluation_pending=True))
    if not passed:
        print('Native pilot goal not met. Best measured configuration retained; no broad ablation started.',flush=True)
        return
    # Lock the winning algorithm BEFORE the only final-target pilot.
    pilot_ids={t:[r['image_id'] for r in records(OUT/'manifests/test'/f'{t}.jsonl')[:2]] for t in TIDS}
    atomic_json(FOCUS/'final_pilot_lock.json',dict(selected=best,validators=VALIDATORS,pilot_ids=pilot_ids,selection='native held-out-surrogate outputs only',primary_budget=dict(steps=50,epsilon=16/255,seed=42),target_feedback_used=False))
    from evaluate_focus_targets import main as final_pilot
    final_pilot()
    pilot_status=json.loads((FOCUS/'reports/final_target_pilot_status.json').read_text())
    if not pilot_status['high_native_pilot_gate']:
        print('Final target pilot remains below the research threshold; no large ablation or target-guided search.',flush=True)
        return
    # At most two prespecified small ablations, still ten dev images/model.
    base=best['config']
    trial({**base,'id':'ABL_no_DI','di':False})
    trial({**base,'id':'ABL_deepest','method':'A1','options':{k:v for k,v in base.get('options',{}).items() if k!='layer_weights'}})
    print('Focused research queue finished; see actual native-output pilot report. No final-target success claimed.',flush=True)

if __name__=='__main__':main()
