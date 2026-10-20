"""Bounded surrogate diagnostics. One GPU admission owner; no commercial invocation."""
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

import run_proxy_study as previous
from pp4_protocol import ROOT,PREVIOUS,OUT,EXP,VALIDATORS,plan,freeze,attack_status,validation_path,sha,atomic_json,development_ids,fingerprint

WORKER_SCRIPT=EXP/'src/pp4_worker.py'


def reuse_controls(p):
    from unified_research_plan import conditions,condition_name,status
    source_conditions={c['id']:c for c in conditions()}
    for row in p['ladder']:
        if row['group']!='A' or row['label'] not in ['four_margin_DI','four_margin_patch_DI']:continue
        c=row['condition'];old=source_conditions['U02' if row['label']=='four_margin_DI' else 'U00']
        old_status=status(old)
        if not old_status['attack_complete'] or not all(old_status['validations'].values()):continue
        if c['models']!=old['models'] or old['fusion']!='equal':raise RuntimeError('Baseline models/fusion differ')
        old_root=OUT/'attacks/dev'/condition_name(old)
        states=[]
        for path in sorted(old_root.glob('T*/batch_*/state.json')):
            d=json.loads(path.read_text());cfg=d['config']
            if cfg['loss']!='margin' or cfg['layers']!=[.25,.5,.75,1.] or cfg.get('patch_weight',0)!=c['options']['patch_weight'] or not cfg['input_diversity']:
                raise RuntimeError('External baseline setting differs')
            states.append(dict(path=str(path),sha256=sha(path)))
        for model in VALIDATORS:
            path=OUT/'focused_optimization/native_validation'/fingerprint(development_ids())[:12]/condition_name(old)/(model.replace('/','_')+'.json')
            d=json.loads(path.read_text())
            if d['denominator']!=10:raise RuntimeError('External baseline cohort incomplete')
            if {r['image_id'] for r in d['images']}!={i for ids in development_ids().values() for i in ids}:raise RuntimeError('External baseline images differ')
            d.update(pp4_reused_from=str(path),pp4_reused_sha256=sha(path),new_inference=False)
            atomic_json(validation_path(c,model),d)
        atomic_json(ROOT/'reuse'/(c['id']+'.json'),dict(source_condition=old['id'],states=states,N=10,
            shared_cohort=development_ids(),new_attack_generated=False))


def evidence(c):
    reuse=ROOT/'reuse'/(c['id']+'.json')
    if reuse.exists():
        d=json.loads(reuse.read_text())
        for s in d['states']:
            if sha(Path(s['path']))!=s['sha256']:raise RuntimeError('Changed external baseline')
        attack=dict(complete=True,N=10,expected_N=10,seconds=0.,peak_reserved_gib=0.,reused=True)
    else:attack=attack_status(c)
    validations={}
    for model in VALIDATORS:
        path=validation_path(c,model)
        if path.exists():
            d=json.loads(path.read_text())
            if d['model']!=model or d['denominator']!=10:raise RuntimeError('Invalid PP4 validation')
            validations[model]=d
    return attack,validations


def table(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def report(p,state):
    rows=[];observations=[];complete={}
    for row in p['ladder']+p['weights']:
        c=row['condition'];a,v=evidence(c);done=len(v)==len(VALIDATORS)
        local=[]
        for model,d in v.items():
            for i in d['images']:
                r=dict(group=row['group'],label=row['label'],condition=c['id'],validator=model,**i)
                observations.append(r);local.append(r)
        hits=sum(r['target_hit'] for r in local)
        rows.append(dict(group=row['group'],label=row['label'],condition=c['id'],models=' + '.join(c['models']),
            loss=c['options']['loss'],pull=c['options']['pull_weight'],push=c['options']['push_weight'],
            layers=json.dumps(c['options']['layers']),DI=c['di'],patch_weight=c['options']['patch_weight'],
            status='COMPLETE' if done else 'PARTIAL' if a['N'] else 'QUEUED',
            unique_attack_images=a['N'],validated_model_images=len(local),hits=hits if done else None,
            denominator=30,ASR=hits/30 if done else None,reused=bool(a.get('reused')),attack_seconds=a['seconds']))
        if done:complete[(row['group'],row['label'])]={(r['validator'],r['image_id']):r for r in local}
    contrasts=[]
    for group in ['A','B']:
        for factor,old,new in [('layers','deep_PP_noDI','four_PP_noDI'),('DI','four_PP_noDI','four_PP_DI'),
            ('global_loss','four_PP_DI','four_margin_DI'),('patch','four_margin_DI','four_margin_patch_DI')]:
            if (group,old) not in complete or (group,new) not in complete:continue
            left,right=complete[(group,old)],complete[(group,new)]
            if left.keys()!=right.keys():raise RuntimeError('Paired observations differ')
            wins=sum(right[k]['target_hit'] and not left[k]['target_hit'] for k in left)
            losses=sum(left[k]['target_hit'] and not right[k]['target_hit'] for k in left)
            ids=sorted({k[1] for k in left})
            per_image=np.array([np.mean([int(right[k]['target_hit'])-int(left[k]['target_hit']) for k in left if k[1]==iid]) for iid in ids])
            rng=np.random.default_rng(42);boot=rng.choice(per_image,size=(4000,len(ids)),replace=True).mean(1)
            contrasts.append(dict(group=group,factor=factor,before=old,after=new,N_unique_images=len(ids),
                N_model_image_pairs=len(left),paired_wins=wins,paired_losses=losses,delta_ASR=float(per_image.mean()),
                image_cluster_bootstrap_low=float(np.quantile(boot,.025)),image_cluster_bootstrap_high=float(np.quantile(boot,.975))))
    table(ROOT/'reports/conditions.csv',rows);table(ROOT/'reports/per_image.csv',observations);table(ROOT/'reports/paired_components.csv',contrasts)
    atomic_json(ROOT/'progress.json',dict(at=time.time(),conditions=rows,failures=state.get('failures',[]),
        active=state.get('active',[]),commercial_testing='AWAITING_USER_CONFIRMATION',full_matrix_expansion=False))
    lines=['# Which component accounts for the improvement?','',
        'Bounded development diagnostics: two fixed proxy combinations,10 unique images each, three held-out validators. No final target or commercial calls. Scores are exploratory and do not establish equivalence or causality.','',
        '| Group | Condition | Status | Hits /30 model-image observations |','|---|---|---|---|']
    lines += [f"| {r['group']} | {r['label']} | {r['status']} | {r['hits'] if r['hits'] is not None else 'N/A'}/30 |" for r in rows]
    lines += ['', 'Matched comparisons change only layer count, DI, global loss, or patch term. Pure Pull/Push uses weights summing to1; no CLS or output head. Five ratios are tested only on combination A. Existing compatible U00/U02 records are reused, with provenance.', '',
        'No automatic full-matrix rerun. Small differences on10 images are inconclusive. Candidate proxy lists/settings must be presented to the user for explicit approval before any new commercial evaluation.']
    (ROOT/'reports/report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--takeover',action='store_true');args=parser.parse_args()
    p=freeze();reuse_controls(p)
    if args.preflight:
        unique={r['condition']['id'] for r in p['ladder']+p['weights']}
        print(f'PASS: {len(unique)} unique development conditions; no target/commercial calls')
        report(p,{});return
    old_state_path=PREVIOUS/'unified_scheduler/state.json'
    if args.takeover:
        owners=[]
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:cmd=path.read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if str(EXP/'src/unified_scheduler.py').encode() in cmd:owners.append(int(path.parent.name))
        if len(owners)>1:raise RuntimeError('Multiple GPU admission owners')
        for pid in owners:
            os.kill(pid,signal.SIGSTOP)  # Freeze admissions before reading adopted state.
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        for _ in range(50):
            if not any(previous.alive(pid) for pid in owners):break
            time.sleep(.2)
        atomic_json(PREVIOUS/'unified_scheduler/handoff_to_pp4.json',dict(at=time.time(),old_owners=owners,
            new_owner=os.getpid(),old_pending_admissions_paused=True,running_children_preserved=True))
    handle=(PREVIOUS/'scheduler.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=ROOT/'scheduler_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else dict(active=[],failures=[],attempts={})
    jobs=[];seen=set()
    adopted=state['active']
    if args.takeover and old_state_path.exists():adopted+= [dict(j,kind='previous') for j in json.loads(old_state_path.read_text())['active']]
    for j in adopted:
        if j['pid'] in seen or not previous.alive(j['pid']):continue
        cmd=(Path('/proc')/str(j['pid'])/'cmdline').read_text()
        if not any(n in cmd for n in ['proxy_study_worker.py','unified_research_worker.py','pp4_worker.py',WORKER_SCRIPT.name]):raise RuntimeError('Foreign adopted child')
        jobs.append(j);seen.add(j['pid'])
    conditions={r['condition']['id']:r['condition'] for r in p['ladder']+p['weights']}
    # First resolve pure four-layer PP versus margin on both compositions.
    order=[r['condition'] for r in p['ladder'] if r['label'] in ['four_PP_DI','four_margin_DI']]
    order += [r['condition'] for r in p['ladder']+p['weights']]
    order=list({c['id']:c for c in order}.values())
    smoke=p['smoke'];conditions[smoke['id']]=smoke;last_report=0;stopping=False
    def request_stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def save():
        state['active']=[{k:v for k,v in j.items() if k not in ['proc','stream']} for j in jobs]
        atomic_json(state_path,state)
    while True:
        for j in list(jobs):
            proc=j.get('proc');finished=proc.poll() is not None if proc else not previous.alive(j['pid'])
            if not finished:continue
            jobs.remove(j)
            if j.get('stream'):j['stream'].close()
            if j.get('kind')=='previous':continue
            c=conditions[j['condition']]
            good=attack_status(c,deep=True)['complete'] if j['phase']=='attack' else validation_path(c,j['models'][0]).exists()
            if not good:
                key=j['condition']+':'+j['phase']+':'+','.join(j['models'])
                state['attempts'][key]=state['attempts'].get(key,0)+1
                state['failures'].append(dict(key=key,at=time.time(),log=j['log'],reason=Path(j['log']).read_text()[-2000:]))
        save()
        if time.time()-last_report>30:
            report(p,state);g=previous.gpu()
            atomic_json(ROOT/'resource_status.json',dict(at=time.time(),gpu=g,active=state['active'],max_gpu_workers=4,
                reserved_limit_gib=72,commercial_workers=0,previous_pending_paused=True))
            last_report=time.time()
        if stopping:save();return  # Children remain resumable/adoptable; never signal their process groups.
        active={j['condition'] for j in jobs if j.get('kind')!='previous'}
        smoke_done=attack_status(smoke)['complete']
        todo=[smoke] if not smoke_done else order
        candidates=[]
        for c in todo:
            if c['id'] in active:continue
            a,v=evidence(c)
            if not a['complete']:phase,models='attack',c['models']
            elif c['steps']!=50:continue
            else:
                pending=[m for m in VALIDATORS if m not in v]
                if not pending:continue
                phase,models='validate',[pending[0]]
            key=c['id']+':'+phase+':'+','.join(models)
            if state['attempts'].get(key,0)>=1:continue
            candidates.append((c,phase,models))
        if not candidates and not jobs:
            report(p,state);atomic_json(ROOT/'completion.json',dict(at=time.time(),status='COMPLETE' if not state['failures'] else 'PARTIAL_FAILURE',
                full_matrix_run=False,commercial_testing='AWAITING_USER_CONFIRMATION'))
            break
        gpu=previous.gpu();reserves=[j['reserve'] for j in jobs]
        for c,phase,models in candidates:
            reserve=previous.reservation(models,'evaluation' if phase=='validate' else 'attack',previous.weight_events())
            if not previous.eligible(0,reserves,reserve,max(len(gpu['pids']),len(jobs))):continue
            if gpu['used_gib']+reserve>72:continue
            mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and len(line.split())>1 and line.split()[1].isdigit()}
            if mem['MemAvailable']/2**20<12:continue
            log=ROOT/'logs'/(c['id']+'_'+phase+'_'+models[0].replace('/','_')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('a');cmd=[sys.executable,str(WORKER_SCRIPT),phase,'--condition',c['id']]
            if phase=='validate':cmd+=['--model',models[0]]
            env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(reserve-.4)}
            proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            jobs.append(dict(kind='pp4',condition=c['id'],phase=phase,models=models,reserve=reserve,
                pid=proc.pid,proc=proc,stream=stream,log=str(log),started=time.time()))
            print('PP4 dispatch',c['id'],phase,round(reserve,2),'GiB',flush=True);save();break
        time.sleep(3)


if __name__=='__main__':main()
