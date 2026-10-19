"""One GPU admission owner, plus independent CPU analysis and commercial CLI lanes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_proxy_study as legacy
from proxy_study_protocol import ROOT,OUT,EXP,atomic_json,plan,checked,lock,report
from unified_research_plan import RESEARCH,VALIDATORS,conditions,condition_name,status as research_status,lock as research_lock

CONTROL=ROOT/'unified_scheduler'


def safe_stop(job):
    if not legacy.alive(job['pid']):return
    cmd=(Path('/proc')/str(job['pid'])/'cmdline').read_text()
    if not any(x in cmd for x in ['proxy_study_worker.py','unified_research_worker.py']):
        raise RuntimeError('Refuse to signal a foreign process')
    os.killpg(job['pid'],signal.SIGTERM)


def next_research(c,s):
    if not s['attack_complete']:return 'attack',c['models']
    for model in VALIDATORS:
        if not s['validations'][model]:return 'validate',[model]
    for model in sorted(set(c['models']+VALIDATORS)):
        if not s['embeddings'][model]:return 'embeddings',[model]
    return None,[]


def research_report(cs,state):
    output=[]
    for c in cs:
        s=research_status(c);complete=all(s['validations'].values())
        row=dict(condition=c['id'],models=c['models'],fusion=c['fusion'],options=c['options'],
              budget_class=c['budget_class'],status='COMPLETE' if complete and all(s['embeddings'].values()) else 'PARTIAL' if s['attack_images'] else 'QUEUED',**s)
        if complete:
            from unified_research_plan import cohort,fingerprint
            pid=fingerprint(cohort())[:12];results=[json.loads((OUT/'focused_optimization/native_validation'/pid/condition_name(c)/(m.replace('/','_')+'.json')).read_text()) for m in VALIDATORS]
            row['minimum_surrogate_model_direction_ASR']=min(v['hits']/v['denominator'] for r in results for v in r['directions'])
            row['pooled_surrogate_ASR']=sum(r['hits'] for r in results)/sum(r['denominator'] for r in results)
        output.append(row)
    atomic_json(RESEARCH/'progress.json',dict(conditions=output,failures=state.get('research_failures',[]),
           final_target_feedback=False,commercial_feedback=False,small_sample_per_direction=2))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');args=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    rows=plan();lock(rows);research_lock();cs=conditions();by_id={r['item']['config']['id']:r for r in rows}
    if args.preflight:
        print('PASS: unchanged96-condition lock,11 separate surrogate-only controls, disjoint commercial/dev cohorts; GPU cap4 /72GiB')
        return
    handle=(ROOT/'scheduler.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((ROOT/'budget.json').read_text());deadline=budget['deadline']
    old_state=json.loads((ROOT/'state.json').read_text())
    file=CONTROL/'state.json';state=json.loads(file.read_text()) if file.exists() else dict(active=[],failures=[],research_failures=[],retries={},dispatch_legacy=0)
    jobs=[];known=set()
    for j in state.get('active',[])+[dict(j,kind='legacy') for j in old_state.get('active',[])]:
        if j['pid'] not in known and legacy.alive(j['pid']):
            cmd=(Path('/proc')/str(j['pid'])/'cmdline').read_text()
            if not any(v in cmd for v in ['proxy_study_worker.py','unified_research_worker.py']):raise RuntimeError('Foreign adopted PID')
            jobs.append(j);known.add(j['pid'])
    atomic_json(CONTROL/'adoption.json',dict(at=time.time(),adopted=jobs,old_method_lock_preserved=True,deadline_unchanged=deadline))
    global_lock=(OUT/'campaign.lock').open('a');last_report=0;cpu=[]

    def save():
        state['active']=[{k:v for k,v in j.items() if k not in ['proc','stream']} for j in jobs]
        atomic_json(file,state)
        old_state['active']=[dict(j) for j in state['active'] if j['kind']=='legacy']
        atomic_json(ROOT/'state.json',old_state)

    # Independent account/API work never reserves local CUDA memory.
    if not state.get('commercial_finished'):
        stream=(CONTROL/'commercial.log').open('a')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
        proc=subprocess.Popen([sys.executable,str(EXP/'src/commercial_validation_campaign.py')],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        cpu.append(dict(kind='commercial',proc=proc,stream=stream));state['commercial_pid']=proc.pid
    try:
        while time.time()<deadline:
            now=time.time()
            try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);exclusive=True
            except BlockingIOError:exclusive=False
            g=legacy.gpu();primary=legacy.primary_stage()
            yielding=(not exclusive and primary['stage']!='attack') or g['used_gib']>72 or len(g['pids'])>4
            if yielding:
                for j in jobs:j['yielded']=True;safe_stop(j)
            for j in list(jobs):
                process=j.get('proc');finished=process.poll() is not None if process else not legacy.alive(j['pid'])
                if not finished:continue
                jobs.remove(j)
                if j.get('stream'):j['stream'].close()
                if j.get('yielded'):continue
                ok=process.returncode==0 if process else True
                if j['kind']=='legacy':
                    if j['phase']=='smoke':old_state['smokes'][j['condition']]=ok
                    elif j['phase'] in ['attack','evaluation']:
                        done=checked(by_id[j['condition']]['item'],deep=True)
                        ok=done['evaluation_complete' if j['phase']=='evaluation' else 'attack_complete']
                elif j['phase']=='attack':ok=research_status(next(c for c in cs if c['id']==j['condition']))['attack_complete']
                if not ok:
                    error=Path(j['log']).read_text()[-1800:]
                    key=j['kind']+':'+j['condition']+':'+j['phase']+':'+','.join(j['models'])
                    record=dict(at=now,key=key,error=error)
                    state['research_failures' if j['kind']=='research' else 'failures'].append(record)
                    retry=state['retries'].setdefault(key,dict(count=0,reserve_floor=0));retry['count']+=1;retry['after']=now+120
                    if 'out of memory' in error.lower():retry['reserve_floor']=j['reserve']+4
                print('[unified complete]',j['kind'],j['condition'],j['phase'],ok,flush=True)
            for c in list(cpu):
                if c['proc'].poll() is not None:
                    state[c['kind']+'_finished']=dict(at=now,returncode=c['proc'].returncode)
                    c['stream'].close();cpu.remove(c)
            save()
            if now-last_report>180:
                report(rows,old_state);legacy.paired_report(rows);research_report(cs,state)
                metadata=sorted((RESEARCH/'embeddings').glob('*/*.json'))
                signature=tuple((str(p),p.stat().st_mtime_ns) for p in metadata)
                if metadata and str(signature)!=state.get('last_mechanism_signature') and not any(c['kind']=='mechanism' for c in cpu):
                    stream=(CONTROL/'mechanism.log').open('a')
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
                    proc=subprocess.Popen([sys.executable,str(EXP/'src/analyze_surrogate_followup.py')],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    cpu.append(dict(kind='mechanism',proc=proc,stream=stream));state['last_mechanism_signature']=str(signature)
                atomic_json(CONTROL/'resource_status.json',dict(at=now,gpu=g,primary=primary,
                    active_gpu_workers=state['active'],max_gpu_workers=4,global_reserved_budget_gib=72,
                    commercial_pid=state.get('commercial_pid'),commercial_status=state.get('commercial_finished'),
                    original_scientific_sources_unchanged=True))
                last_report=now
            if yielding:time.sleep(3);continue
            events=legacy.weight_events();reserves=[j['reserve'] for j in jobs]
            primary_reserve=0 if exclusive else max(10.5,g['used_gib']-sum(reserves))
            active_names={(j['kind'],j['condition']) for j in jobs};candidates=[]
            # Keep the original relative queue order; interleave one research job
            # per two original dispatches, with at most one research GPU job.
            allow_research=not any(j['kind']=='research' for j in jobs)
            if allow_research:
                for c in cs:
                    if ('research',c['id']) in active_names:continue
                    phase,models=next_research(c,research_status(c))
                    if phase:candidates.append(dict(kind='research',condition=c['id'],phase=phase,models=models,config=c))
                    if phase:break
            legacy_candidates=[];core_remaining=any(r['tier']=='core' and not checked(r['item'])['evaluation_complete'] for r in rows)
            for r in rows:
                n=r['item']['config']['id']
                if ('legacy',n) in active_names or n in old_state.get('blocked',{}):continue
                if core_remaining and r['tier']!='core':continue
                if r['item']['source']=='corrected_family_disjoint' and not exclusive:continue
                d=checked(r['item']);phase,models=legacy.phase_for(r,d,old_state)
                if phase:legacy_candidates.append(dict(kind='legacy',condition=n,phase=phase,models=models,row=r))
            if state['dispatch_legacy']%3!=2:candidates=legacy_candidates+candidates
            else:candidates=candidates+legacy_candidates
            for task in candidates:
                models=task['models'];phase=task['phase'];kind=task['kind'];name=task['condition']
                key=kind+':'+name+':'+phase+':'+','.join(models);retry=state['retries'].get(key,{})
                if retry.get('count',0)>=2 or retry.get('after',0)>now:continue
                if phase=='calibration' and any(j['phase']=='calibration' or set(models)&set(j['models']) for j in jobs):continue
                try:request=max(retry.get('reserve_floor',0),legacy.reservation(models,'evaluation' if phase in ['evaluation','validate'] else 'attack',events))
                except Exception as exc:
                    state.setdefault('resource_blockers',{})[key]=str(exc);continue
                if not legacy.eligible(primary_reserve,reserves,request,max(len(g['pids']),len(jobs)+(0 if exclusive else 1))):continue
                if deadline-now<900 or phase=='attack' and deadline-now<2700:continue
                if kind=='legacy':cmd=[sys.executable,str(EXP/'src/proxy_study_worker.py'),*map(str,legacy.args_for(task['row'],phase,models))]
                else:
                    cmd=[sys.executable,str(EXP/'src/unified_research_worker.py'),phase,'--condition',name]
                    if phase!='attack':cmd+=['--model',models[0]]
                log=CONTROL/(kind+'_'+name+'_'+phase+'_'+models[0].replace('/','_')+'.log');stream=log.open('a')
                env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(request-.4)}
                proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                jobs.append(dict(kind=kind,condition=name,phase=phase,models=models,reserve=request,pid=proc.pid,
                     proc=proc,stream=stream,log=str(log),started=now))
                if kind=='legacy':state['dispatch_legacy']+=1
                else:state['dispatch_legacy']=0
                atomic_json(CONTROL/'last_admission.json',dict(at=now,task={k:v for k,v in task.items() if k not in ['row']},
                   predicted_reserved_gib=request,primary_reserved_gib=primary_reserve,other_reservations=reserves,gpu_before=g))
                print('[unified dispatch]',kind,name,phase,round(request,2),'GiB',flush=True);save();break
            if not legacy_candidates and not candidates and not jobs:break
            time.sleep(4)
    finally:
        for j in jobs:safe_stop(j)
        for j in jobs:
            if j.get('proc'):
                try:j['proc'].wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(j['pid'],signal.SIGKILL)
        wait_until=time.time()+20
        while any(legacy.alive(j['pid']) for j in jobs) and time.time()<wait_until:time.sleep(.5)
        jobs.clear();save();old_state['window_finished']=time.time();atomic_json(ROOT/'state.json',old_state)
        research_report(cs,state);report(rows,old_state)
        atomic_json(CONTROL/'completion.json',dict(at=time.time(),remaining_work_resumable=True,commercial_may_continue_independently=True))


if __name__=='__main__':main()
