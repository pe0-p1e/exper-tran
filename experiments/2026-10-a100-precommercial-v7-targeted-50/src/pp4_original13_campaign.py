"""Original13 full test rows; phased memory reservations; grouped evaluation."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import run_proxy_study as resources
from pp4_campaign import table
from pp4_original13_protocol import ROOT,PREVIOUS,OUT,EXP,freeze,smoke,evidence,sha,atomic_json,registry,TIDS


def historical(row):
    import supplement_protocol
    try:
        d=supplement_protocol.evidence(row['previous_A3'])
        a3=d['detailed'];error=''
    except Exception as exc:a3=[];error=str(exc)
    v6file=OUT.parent/'a100_precommercial_v6_multi_repr_recovery/analysis/per_image_metrics.csv'
    v6=[]
    if v6file.exists():
        for r in csv.DictReader(v6file.open()):
            if r['ensemble_id'].split('_',1)[0]==row['id'] and r['target']==row['target']:
                v6.append(dict(image_id=r['image_id'],transition=r['transition_id'],target_hit=r['target_hit'] in ['1','True','true']))
    return a3,v6,error


def report(rows,state):
    summary=[];per_image=[];directed=[];paired=[];baselines=[]
    for row in rows:
        c=row['condition'];d=evidence(c);a3,v6,error=historical(row)
        new=d['measured'];complete=d['evaluation_complete']
        hits=sum(r['target_hit'] for r in new)
        summary.append(dict(ensemble=row['id'],target=row['target'],models=' + '.join(c['models']),
            status='COMPLETE' if complete else 'ATTACK_COMPLETE' if d['attack_complete'] else 'PARTIAL' if d['attack_images'] else 'QUEUED',
            attack_images=d['attack_images'],evaluated_images=len(new),hits=hits if complete else None,denominator=150,
            ASR=hits/150 if complete else None,old_A3_hits=sum(r['target_hit'] for r in a3) if len(a3)==150 else None,
            old_A3_N=len(a3),old_V6_hits=sum(r['target_hit'] for r in v6) if len(v6)==150 else None,
            old_V6_N=len(v6),attack_seconds=d['attack_seconds'],peak_reserved_gib=d['peak_reserved_gib'],old_A3_error=error))
        per_image.extend(dict(ensemble=row['id'],target=row['target'],**r) for r in new)
        for tid in TIDS:
            part=[r for r in new if r['transition']==tid]
            directed.append(dict(ensemble=row['id'],target=row['target'],transition=tid,observed=len(part),
                hits=sum(r['target_hit'] for r in part) if len(part)==30 else None,denominator=30))
        for label,old in [('A3_margin_patch_DI',a3),('V6_deep_PP_noDI',v6)]:
            oldmap={r['image_id']:r for r in old};newmap={r['image_id']:r for r in new}
            baselines.extend({**r,'ensemble':row['id'],'baseline':label,'target':row['target']} for r in old)
            if not complete or set(oldmap)!=set(newmap) or len(oldmap)!=150:continue
            delta=np.array([int(newmap[k]['target_hit'])-int(oldmap[k]['target_hit']) for k in sorted(newmap)])
            boot=np.random.default_rng(42).choice(delta,size=(4000,150),replace=True).mean(1)
            paired.append(dict(ensemble=row['id'],target=row['target'],baseline=label,N=150,
                paired_wins=int((delta>0).sum()),paired_losses=int((delta<0).sum()),delta_ASR=float(delta.mean()),
                image_bootstrap_low=float(np.quantile(boot,.025)),image_bootstrap_high=float(np.quantile(boot,.975)),
                interpretation='Method-level contrast; not a weight-only ablation'))
    table(ROOT/'reports/completion.csv',summary);table(ROOT/'reports/per_image.csv',per_image)
    table(ROOT/'reports/directions.csv',directed);table(ROOT/'reports/paired_comparisons.csv',paired)
    # Baseline schemas differ; preserve separate raw sources instead of silently joining columns.
    atomic_json(ROOT/'reports/baseline_observations.json',baselines)
    atomic_json(ROOT/'progress.json',dict(at=time.time(),conditions=summary,active=state.get('active',[]),
        failures=state.get('failures',[]),expected_conditions=13,expected_directions=65,expected_image_rows=1950,
        commercial_testing=False,target_feedback_for_tuning=False))
    lines=['# Original Multiple Proxy table: four-layer balanced Pull/Push','',
        'Fixed original13 unique workbook rows,65 directions,1950 image-evaluation rows. Same ground-truth cohorts, revisions, prompt and strict parser. New settings: uniform four-layer pure Pull/Push.5/.5, DI, equal normalized gradients,50 steps, BF16 weights. No commercial evaluation.', '',
        '| Ensemble | Target | Status | New hits /150 | Old A3 hits /150 | V6 hits /150 |','|---|---|---|---:|---:|---:|']
    lines += [f"| {r['ensemble']} | {r['target']} | {r['status']} | {r['hits'] if r['hits'] is not None else 'N/A'}/150 | {r['old_A3_hits'] if r['old_A3_hits'] is not None else 'N/A'}/150 | {r['old_V6_hits'] if r['old_V6_hits'] is not None else 'N/A'}/150 |" for r in summary]
    lines += ['', 'Missing/partial results are not zero. Old A3 changes global loss and patch weighting relative to new PP; historical V6 additionally changes layers/DI. Neither contrast isolates the Pull/Push coefficient. Use the matched development diagnostic studies for component-specific evidence. Paired comparisons require identical150 image IDs. These final-target results do not tune the next attack or commercial proxy selection.']
    (ROOT/'reports/report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    rows=freeze();s=smoke()
    from pp4_original13_evaluator import lock as evaluation_lock
    evaluation_lock()
    if args.preflight:
        report(rows,{})
        print('PASS original13 /65 directions /1950 rows; pinned checkpoints present; GT cohorts/reference bank disjoint; commercial off');return
    handle=(PREVIOUS/'scheduler.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    file=ROOT/'scheduler_state.json';state=json.loads(file.read_text()) if file.exists() else dict(active=[],failures=[],attempts={})
    jobs=[]
    for j in state['active']:
        if resources.alive(j['pid']):
            cmd=(Path('/proc')/str(j['pid'])/'cmdline').read_text()
            if not any(n in cmd for n in ['pp4_original13_worker.py','pp4_original13_evaluator.py']):raise RuntimeError('Foreign resumed PID')
            jobs.append(j)
    stop=False;last_report=0
    def halt(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def save():
        state['active']=[{k:v for k,v in j.items() if k not in ['proc','stream']} for j in jobs];atomic_json(file,state)
    while True:
        for j in list(jobs):
            proc=j.get('proc');done=proc.poll() is not None if proc else not resources.alive(j['pid'])
            if not done:continue
            jobs.remove(j)
            if j.get('stream'):j['stream'].close()
            if j['mode']=='attack':
                c=s if j['key']=='smoke' else next(r['condition'] for r in rows if r['id']==j['key'])
                good=evidence(c,deep=True)['attack_complete']
            else:good=all(evidence(r['condition'])['evaluation_complete'] for r in rows if r['target']==j['key'])
            if not good:
                key=j['mode']+':'+j['key'];state['attempts'][key]=state['attempts'].get(key,0)+1
                state['failures'].append(dict(task=key,at=time.time(),log=j['log'],error=Path(j['log']).read_text()[-2400:]))
            print('finished',j['mode'],j['key'],good,flush=True)
        # Initialization weights can exceed steady-state vision-only residence.
        for j in jobs:
            if j['mode']!='attack':continue
            marker=ROOT/'runtime'/(j['condition_id']+'.json')
            if marker.exists():
                d=json.loads(marker.read_text())
                if d['initialization_complete']:
                    j['reserve']=min(j['initial_reserve'],d['steady_reservation_gib'])
                    j['host_reserve']=min(j['initial_host_reserve'],d['steady_cpu_parameter_gib']+4)
        save()
        if time.time()-last_report>60:
            report(rows,state);atomic_json(ROOT/'resource_status.json',dict(at=time.time(),gpu=resources.gpu(),active=state['active'],
                max_workers=4,predictive_gpu_cap_gib=72,predictive_host_cap_gib=96));last_report=time.time()
        if stop:return
        active={(j['mode'],j['key']) for j in jobs};candidates=[]
        if not evidence(s)['attack_complete']:
            candidates=[('attack','smoke',s['models'],s)]
        else:
            for target in sorted({r['target'] for r in rows}):
                group=[r for r in rows if r['target']==target]
                if all(evidence(r['condition'])['attack_complete'] for r in group) and not all(evidence(r['condition'])['evaluation_complete'] for r in group):
                    candidates.append(('evaluation',target,[target],None))
            # Large initialization jobs start early; independent smaller jobs fill safe slack.
            events=resources.weight_events()
            pending=[r for r in rows if not evidence(r['condition'])['attack_complete']]
            pending.sort(key=lambda r:resources.reservation(r['condition']['models'],'attack',events),reverse=True)
            candidates += [('attack',r['id'],r['condition']['models'],r['condition']) for r in pending]
        if all(evidence(r['condition'])['evaluation_complete'] for r in rows):
            report(rows,state);atomic_json(ROOT/'completion.json',dict(at=time.time(),status='COMPLETE',conditions=13,directions=65,image_rows=1950,commercial_testing=False));return
        gpu=resources.gpu();events=resources.weight_events();dispatched=False
        for mode,key,models,c in candidates:
            task=mode+':'+key
            if (mode,key) in active or state['attempts'].get(task,0)>=1:continue
            reserve=resources.reservation(models,mode,events)
            sizes=[sum(events[n]['parameter_bytes_by_dtype'].values())/2**30 for n in models]
            gemma=sum(v for n,v in zip(models,sizes) if registry()[n]['family']=='gemma4')
            other=max([v for n,v in zip(models,sizes) if registry()[n]['family']!='gemma4'] or [0])
            host=(max(sizes) if mode=='evaluation' else gemma+other)+4
            available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/2**20
            if not resources.eligible(0,[j['reserve'] for j in jobs],reserve,max(len(gpu['pids']),len(jobs))):continue
            if gpu['used_gib']+reserve>72 or sum(j['host_reserve'] for j in jobs)+host>96 or available<host+8:continue
            log=ROOT/'logs'/(mode+'_'+key.replace('/','_')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('a')
            if mode=='attack':cmd=[sys.executable,str(EXP/'src/pp4_original13_worker.py'),'attack','--condition',c['id']]
            else:cmd=[sys.executable,str(EXP/'src/pp4_original13_evaluator.py'),'--target',key]
            env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(reserve-.4)}
            proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            jobs.append(dict(mode=mode,key=key,models=models,condition_id=c['id'] if c else None,pid=proc.pid,
                reserve=reserve,initial_reserve=reserve,host_reserve=host,initial_host_reserve=host,started=time.time(),
                log=str(log),proc=proc,stream=stream));save();dispatched=True
            print('dispatch',mode,key,'GPU',round(reserve,2),'host',round(host,2),flush=True);break
        if not candidates and not jobs:
            report(rows,state);atomic_json(ROOT/'completion.json',dict(at=time.time(),status='PARTIAL_FAILURE',commercial_testing=False));return
        # Failed tasks stay documented while independent rows continue; don't spin forever after all are attempted.
        if candidates and not jobs and not dispatched and all(state['attempts'].get(m+':'+k,0)>=1 for m,k,_,_ in candidates):
            report(rows,state);atomic_json(ROOT/'completion.json',dict(at=time.time(),status='PARTIAL_FAILURE',commercial_testing=False));return
        time.sleep(4)


if __name__=='__main__':main()
