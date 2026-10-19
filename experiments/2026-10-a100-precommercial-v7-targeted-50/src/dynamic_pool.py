"""Four-worker maximum with predictive reservations and measured memory feedback."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from v7_common import OUT,EXP,REGISTRY,atomic_json,sha
from supplement_protocol import ROOT,candidates,evidence,inventory,snapshot
from family_protocol import condition_name
from run_parallel_subsets import parent_progress,gpu_status


def fits(primary,active,requested,max_workers=4,total_budget=72):
    return 1+len(active)<max_workers and primary+sum(active)+requested<=total_budget


def evaluation_can_share(primary,required,total_budget=72):
    # An evaluation needing exclusive GPU access must not stall all smaller
    # attacks while the original owner continues its long frozen campaign.
    return primary+required<=total_budget


def weights():
    result={}
    for path in [OUT/'audits/model_dtype_audit.json',ROOT/'audits/model_dtype_audit.json']:
        if path.exists():
            for e in json.loads(path.read_text()):
                if e['status']=='pass' and e['all_floating_parameters_bf16']:
                    result[e['model_name']]=e
    return result


def estimate(models,phase,events,calibration):
    totals=[];resident=[];activation=[]
    for n in models:
        e=events.get(n)
        if e is None or e['revision']!=REGISTRY[n]['revision']:raise RuntimeError('Missing pinned weight-size audit: '+n)
        size=sum(e['parameter_bytes_by_dtype'].values())/2**30;totals.append(size)
        r=sum(e['vision_encoder_parameter_bytes_by_dtype'].values())/2**30 if REGISTRY[n]['family']=='gemma4' else size
        resident.append(r)
        a=calibration.get(n,{})
        measurements=[m['peak_reserved_gib'] for m in a.get('measurements',[]) if m.get('safe') and m['batch']==2]
        activation.append(max(.8,max(measurements,default=r+2)-r))
    if phase=='evaluation':return totals[0]+2.6
    return max(max(totals)+.4,sum(resident)+max(activation)) +1.2


def alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split()[2]!='Z'


def recover_jobs(saved,tasks):
    """Adopt orphaned scientific children after a controller-only interruption."""
    result=[]
    for job in saved:
        if not alive(job['pid']):continue
        cmd=(Path('/proc')/str(job['pid'])/'cmdline').read_text()
        if not any(s in cmd for s in ['pool_entry.py','run_supplement_attack.py','evaluate_supplement.py']):raise RuntimeError('Stale PID now belongs to another process')
        if job['condition'] not in tasks:raise RuntimeError('Orphan condition outside frozen pool')
        result.append({**job,'item':tasks[job['condition']]['item'],'adopted':True})
    return result


def stop_owned(job):
    if not alive(job['pid']):return
    cmd=(Path('/proc')/str(job['pid'])/'cmdline').read_text()
    if not any(s in cmd for s in ['pool_entry.py','run_supplement_attack.py','evaluate_supplement.py']):raise RuntimeError('Refuse to signal unrecognized PID')
    os.kill(job['pid'],signal.SIGTERM)


def checkpoint(state,jobs):
    state['active']=[{k:v for k,v in j.items() if k not in ('proc','stream','item')} for j in jobs]
    atomic_json(ROOT/'pool_state.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--adopt-pid',type=int);p.add_argument('--adopt-condition',default='PSQ_VA');a=p.parse_args()
    lock=(ROOT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=json.loads((ROOT/'budget.json').read_text())['deadline'];items=candidates()
    # Preserve the already-running v1 runtime lock. Only controller recovery
    # changes; scientific loss/runner/cohort and method locks stay unchanged.
    protocol=ROOT/'pool_runtime_lock_v2.json'
    settings=dict(runtime_version=2,previous_runtime_lock_sha256=sha(ROOT/'pool_runtime_lock.json') if (ROOT/'pool_runtime_lock.json').exists() else None,max_total_GPU_workers=4,global_memory_budget_gib=72,primary_settings_unchanged=True,method_lock_sha256=sha(ROOT/'method_lock.json'),source_hashes={n:sha(EXP/'src'/n) for n in ['dynamic_pool.py','pool_entry.py']},target_feedback_for_scheduling=False,batch=2)
    if protocol.exists() and json.loads(protocol.read_text())!=settings:raise RuntimeError('Changed runtime pool protocol')
    if not protocol.exists():atomic_json(protocol,settings)
    statepath=ROOT/'pool_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else dict(started=time.time(),conditions={},failures=[],memory_events=[],completed_phases=[])
    order=['PSQ_VA','PSI_VA','PSG_VB','PSG_VA','PSQ_VB','PSI_VB','PSQ_AB','PSI_AB','PSG_AB']
    queue=[next(r for r in items if r['config']['id']==n) for n in order]
    queue+=sorted([r for r in items if not r['optional'] and r['source']=='new_supplement' and len(r['config']['models'])>1],key=lambda r:(r['priority'],r['config']['id']))
    jobs=[];tasks={};calib_path=ROOT/'audits/model_gradients.json'
    for item in queue:
        n=item['config']['id'];e=evidence(item,deep=True)
        tasks[n]=dict(item=item,attack_complete=e['attack_complete'],evaluation_complete=e['evaluation_complete'],smoke_complete=False,retry_after=0)
    jobs=recover_jobs(state.get('active',[]),tasks)
    if a.adopt_pid and alive(a.adopt_pid) and all(j['pid']!=a.adopt_pid for j in jobs):
        task=tasks[a.adopt_condition]
        cmd=(Path('/proc')/str(a.adopt_pid)/'cmdline').read_text()
        if 'run_supplement_attack.py' not in cmd or a.adopt_condition not in cmd:raise RuntimeError('Adopted PID does not match expected attack')
        jobs.append(dict(pid=a.adopt_pid,condition=a.adopt_condition,phase='attack',reserved_gib=8,started=time.time(),item=task['item'],adopted=True))
        task['smoke_complete']=True
    global_lock=(OUT/'campaign.lock').open('a');exclusive=False;last_snapshot=0;last_benchmark=0;samples=[]
    while time.time()<deadline:
        try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);exclusive=True
        except BlockingIOError:pass
        gpu=gpu_status();primary=parent_progress();now=time.time()
        samples.append(dict(at=now,**gpu,primary=primary,active=[dict(condition=j['condition'],phase=j['phase'],reserved_gib=j['reserved_gib']) for j in jobs]))
        if not exclusive and primary['stage']!='attack' or gpu['used_gib']>72 or gpu['worker_count']>4:
            # Only stop OUR workers; frozen batches survive and the original owner is untouched.
            for j in jobs:stop_owned(j)
            state['memory_events'].append(dict(at=now,reason='yield_to_primary_or_memory_guard',gpu=gpu,primary=primary))
        for j in list(jobs):
            proc=j.get('proc');done=proc.poll() is not None if proc else not alive(j['pid'])
            if not done:continue
            jobs.remove(j);task=tasks[j['condition']]
            if 'stream' in j:j['stream'].close()
            if proc:success=proc.returncode==0
            elif j['phase']=='calibration':
                a=json.loads(calib_path.read_text()) if calib_path.exists() else {}
                success=all(a.get(n,{}).get('passed') for n in j['item']['config']['models'])
            elif j['phase']=='smoke':
                files=list((OUT/'attacks/smoke').glob(j['condition']+'_smoke__*/T01/batch_*/state.json'))
                success=sum(len(json.loads(p.read_text())['image_ids']) for p in files)==2
            else:
                checked=evidence(j['item'],deep=True)
                success=checked['evaluation_complete' if j['phase']=='evaluation' else 'attack_complete']
            if success:
                if j['phase']=='smoke':task['smoke_complete']=True
                elif j['phase']=='attack':task['attack_complete']=evidence(j['item'],deep=True)['attack_complete']
                elif j['phase']=='evaluation':task['evaluation_complete']=evidence(j['item'],deep=True)['evaluation_complete']
                state['completed_phases'].append(dict(condition=j['condition'],phase=j['phase'],finished=now))
                state['conditions'][j['condition']]={k:task[k] for k in ['attack_complete','evaluation_complete']}
            else:
                log=ROOT/'logs'/(j['condition']+'_pool_'+j['phase']+'.log')
                text=log.read_text()[-5000:] if log.exists() else ''
                state['failures'].append(dict(condition=j['condition'],phase=j['phase'],at=now,error=text[-1200:]))
                task['retry_after']=now+120
                # Never silently switch dtype/budget after OOM; request more memory next time.
                if 'out of memory' in text.lower():task['reserve_floor']=j['reserved_gib']+3
                else:task['retry_after']=now+600
        if now-last_benchmark>30:
            atomic_json(ROOT/'audits/dynamic_memory_benchmark.json',dict(samples=samples[-180:],max_global_used_gib=max(s['used_gib'] for s in samples),max_total_GPU_workers=max(s['worker_count'] for s in samples),mean_GPU_utilization=sum(s['utilization'] for s in samples)/len(samples),original_primary_unchanged=True))
            checkpoint(state,jobs);last_benchmark=now
        if now-last_snapshot>180:
            inventory();snapshot(items,owner_active=not exclusive);last_snapshot=now
        if not exclusive and primary['stage']!='attack':time.sleep(3);continue
        calibration=json.loads(calib_path.read_text()) if calib_path.exists() else {};events=weights()
        active_names={j['condition'] for j in jobs};active_reserves=[j['reserved_gib'] for j in jobs]
        # Initial full Gemma loading also counts; reserve more than current live tensors.
        primary_reserve=0 if exclusive else max(10.5,gpu['used_gib']-sum(j['reserved_gib'] for j in jobs))
        waiting_eval=[(n,t) for n,t in tasks.items() if t['attack_complete'] and not t['evaluation_complete'] and n not in active_names and t['retry_after']<=now]
        feasible_eval=[]
        for n,t in waiting_eval:
            try:requested=max(t.get('reserve_floor',0),estimate([t['item']['config']['target']],'evaluation',events,calibration))
            except Exception:continue
            if evaluation_can_share(primary_reserve,requested):feasible_eval.append((n,t))
        pending=waiting_eval+[(n,t) for n,t in tasks.items() if not t['attack_complete'] and n not in active_names and t['retry_after']<=now]
        if gpu['worker_count']>=4:pending=[]
        dispatched=False
        for name,task in pending:
            item=task['item'];c=item['config']
            if task['attack_complete']:phase='evaluation';models=[c['target']]
            elif any(not calibration.get(n,{}).get('passed') for n in c['models']):
                if any(j['phase']=='calibration' for j in jobs):continue
                phase='calibration';models=c['models']
            else:phase='attack' if task['smoke_complete'] else 'smoke';models=c['models']
            try:reserve=max(task.get('reserve_floor',0),estimate(models,phase,events,calibration))
            except Exception as exc:
                state['failures'].append(dict(condition=name,error=str(exc),at=now));task['retry_after']=now+3600;continue
            if not fits(primary_reserve,active_reserves,reserve,max_workers=4 if not exclusive else 5):continue
            if phase not in ('evaluation',) and feasible_eval:continue # drain only when evaluation can fit beside primary
            if deadline-now<900 or phase=='attack' and deadline-now<1800:continue
            cfg=ROOT/'configs'/(name+'.json');atomic_json(cfg,c['options'])
            common=['--models',*c['models'],'--method','A3','--fusion','equal','--seed',42,'--restarts',1,'--batch-size',2,'--config-json',cfg,'--excluded-proxy-family',REGISTRY[c['target']]['family'],'--input-diversity']
            if phase=='calibration':args=['calibration','--models',*c['models']]
            elif phase=='evaluation':args=['evaluation','--root',OUT/'attacks/test'/item['measurement_condition'],'--target',c['target'],'--method-lock',OUT/'proxy_selection/method_lock.json' if item['source']=='new_subset_measurement' else ROOT/'method_lock.json']
            else:args=['attack',*common,'--split','smoke' if phase=='smoke' else 'test','--steps',2 if phase=='smoke' else 50,'--run-label',name+('_smoke' if phase=='smoke' else '')]+(['--transitions','T01','--limit-per-direction',2] if phase=='smoke' else [])
            log=(ROOT/'logs'/(name+'_pool_'+phase+'.log')).open('a')
            env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(max(2,reserve-.6))}
            proc=subprocess.Popen([sys.executable,str(EXP/'src/pool_entry.py'),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(dict(proc=proc,pid=proc.pid,condition=name,phase=phase,reserved_gib=reserve,started=now,item=item,stream=log))
            atomic_json(ROOT/'audits'/(name+'_pool_admission.json'),dict(at=now,phase=phase,weight_and_activation_estimate_gib=reserve,primary_reserved_gib=primary_reserve,other_reservations_gib=active_reserves,total_reserved_gib=primary_reserve+sum(active_reserves)+reserve,parameter_dtype='bf16',max_total_workers=4))
            print('[pool dispatch]',name,phase,'reserved GiB',round(reserve,2),flush=True)
            checkpoint(state,jobs);dispatched=True;break
        if all(t['evaluation_complete'] for t in tasks.values()):break
        time.sleep(2 if dispatched else 5)
    for j in jobs:stop_owned(j)
    checkpoint(state,[]);inventory();table=snapshot(items,owner_active=not exclusive,deep=True)
    atomic_json(ROOT/'pool_completion.json',dict(finished=time.time(),complete_unique_conditions=sum(r['status']=='COMPLETE' for r in table),total_unique_conditions=47,remaining_work_is_resumable=True,failures=state['failures']))
    from archive_next_stage import upload_compact
    upload_compact(ROOT)


if __name__=='__main__':main()
