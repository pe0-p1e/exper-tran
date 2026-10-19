"""User-authorized sidecar: one extra GPU worker, hard memory cap and yield guard."""
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from v7_common import OUT,EXP,atomic_json
from supplement_protocol import ROOT,evidence,inventory,snapshot


class YieldForOwner(RuntimeError):pass


def gpu_status():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(values)!=1:raise RuntimeError('Shared worker expects exactly one verified GPU')
    memory,util=[float(x.strip()) for x in values[0].split(',')]
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
    return dict(used_gib=memory/1024,utilization=util,worker_count=len([p for p in pids if p.strip()]))


def parent_progress():
    root=OUT/'family_disjoint_recovery';stage=json.loads((root/'current_stage.json').read_text())
    state=json.loads((root/'state.json').read_text());condition=stage.get('condition')
    files=[]
    if condition:
        for p in (OUT/'attacks/test').glob(condition+'__*'):files+=list(p.glob('T*/batch_*/state.json'))
    recent=sorted(files,key=lambda p:p.stat().st_mtime)[-5:]
    times=[json.loads(p.read_text())['runtime_seconds'] for p in recent]
    active_images=2*len(files) if condition not in state['attacks'] else 0
    return dict(stage=stage['stage'],condition=condition,attack_images=150*len(state['attacks'])+active_images,completed_conditions=len(state['attacks']),recent_batch_seconds=statistics.median(times) if times else None)


def worker_guard(progress,gpu,exclusive):
    if gpu.get('worker_count',0)>4:return 'More than four GPU workers; stop supplementary worker'
    if gpu['used_gib']>72:return 'Global GPU usage exceeded72GiB; yield to primary owner'
    if not exclusive and progress['stage']!='attack':return 'Primary owner entering inference/analysis; release sidecar GPU'
    return None


def terminate_child(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def supervised(script,args,log,cap,exclusive,deadline,monitor):
    path=ROOT/'logs'/log;path.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(cap)}
    if time.time()>=deadline:raise YieldForOwner('Eight-hour deadline reached before dispatch')
    with path.open('a') as stream:
        proc=subprocess.Popen([sys.executable,str(EXP/'src'/script),*map(str,args)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                gpu=gpu_status();progress=parent_progress()
                reason=worker_guard(progress,gpu,exclusive)
                if time.time()>=deadline:reason='Eight-hour deadline reached; keep completed batches'
                if reason:
                    terminate_child(proc);raise YieldForOwner(reason)
                monitor.append(dict(at=time.time(),script=script,child_pid=proc.pid,memory_cap_gib=cap,**gpu,primary=progress))
                time.sleep(2)
        finally:
            if proc.poll() is None:terminate_child(proc)
    if proc.returncode:raise RuntimeError(f'Exit{proc.returncode}: {path}')


def main(items,deadline):
    from run_multiproxy_supplement import lock_protocol,admission
    from proxy_selection_tables import matrix,prepare
    from run_proxy_selection import raw_results,selection_lock
    from family_protocol import condition_name
    lock_protocol(items);prepare(matrix())
    statepath=ROOT/'state.json';state=json.loads(statepath.read_text()) if statepath.exists() else dict(started=time.time(),failures=[],blocked={},attacks={},evaluations={})
    state['user_authorized_shared_GPU']=True;state['max_new_GPU_workers']=1
    atomic_json(statepath,state)
    baseline=parent_progress();monitor=[];bench=ROOT/'audits/concurrency_benchmark.json'
    global_lock=(OUT/'campaign.lock').open('a');exclusive=False
    # Predeclared order: light CLIP/VLM pairs first, then the complementary pairs.
    order=['PSQ_VA','PSI_VA','PSG_VA','PSQ_VB','PSI_VB','PSG_VB','PSQ_AB','PSI_AB','PSG_AB']
    queue=[next(r for r in items if r['config']['id']==name) for name in order]
    full_queue=[r for r in items if not r['optional'] and len(r['config']['models'])>1 and r not in queue and r['source']=='new_supplement']
    queue+=sorted(full_queue,key=lambda r:(r['priority'],r['config']['id']))
    for item in queue:
        c=item['config'];name=c['id']
        while time.time()<deadline-900:
            try:
                if evidence(item,deep=True)['evaluation_complete']:break
                try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);exclusive=True
                except BlockingIOError:pass
                phase=parent_progress();gpu=gpu_status()
                if worker_guard(phase,gpu,exclusive) or gpu['used_gib']>50:
                    atomic_json(ROOT/'current_stage.json',dict(stage='YIELDING_TO_PRIMARY',condition=name,primary=phase,at=time.time()))
                    time.sleep(30);continue
                atomic_json(ROOT/'current_stage.json',dict(stage='PARALLEL_PROXY_CALIBRATION',condition=name,at=time.time()))
                # Extra worker alone is limited to16GiB; original owner unchanged.
                attack_cap=min(70-gpu['used_gib'],65) if exclusive else 16
                supervised('smoke_supplement_models.py',['--models',*c['models']],name+'_shared_calibration.log',attack_cap,exclusive,deadline,monitor)
                audit=json.loads((ROOT/'audits/model_gradients.json').read_text());batch=min(audit[n]['chosen_batch'] for n in c['models'])
                eta=150*50*sum(audit[n]['seconds_per_image_step'] for n in c['models'])*1.5+60*len(c['models'])
                atomic_json(ROOT/'audits'/(name+'_admission.json'),dict(estimated_attack_seconds=eta,remaining_seconds=deadline-time.time(),admitted=admission(deadline-time.time(),eta),shared_GPU=True,extra_worker_memory_cap_gib=16,batch=batch))
                e=evidence(item,deep=True)
                if not e['attack_complete']:
                    if not admission(deadline-time.time(),eta):break
                    cfg=ROOT/'configs'/(name+'.json');atomic_json(cfg,c['options'])
                    common=['--models',*c['models'],'--method','A3','--fusion','equal','--seed',42,'--restarts',1,'--batch-size',batch,'--config-json',cfg,'--excluded-proxy-family',__import__('v7_common').REGISTRY[c['target']]['family'],'--input-diversity']
                    atomic_json(ROOT/'current_stage.json',dict(stage='PARALLEL_ATTACK',condition=name,at=time.time()))
                    supervised('run_supplement_attack.py',[*common,'--split','smoke','--steps',2,'--run-label',name+'_smoke','--transitions','T01','--limit-per-direction',2],name+'_shared_smoke.log',attack_cap,exclusive,deadline,monitor)
                    supervised('run_supplement_attack.py',[*common,'--split','test','--steps',50,'--run-label',name],name+'_shared_attack.log',attack_cap,exclusive,deadline,monitor)
                    e=evidence(item,deep=True)
                    if not e['attack_complete']:raise RuntimeError('Incomplete scientific attack cohort')
                state['attacks'][name]=dict(finished=time.time());atomic_json(statepath,state)
                # Target only consumes this condition's already frozen PNGs;
                # no outputs enter selection or the fixed subsequent attack queue.
                phase=parent_progress();gpu=gpu_status()
                cap=min(60,70-gpu['used_gib'])
                if cap<30 or worker_guard(phase,gpu,exclusive):raise YieldForOwner('Insufficient safe target-evaluation headroom')
                lock=OUT/'proxy_selection/method_lock.json' if item['source']=='new_subset_measurement' else ROOT/'method_lock.json'
                atomic_json(ROOT/'current_stage.json',dict(stage='PARALLEL_FROZEN_PNG_EVALUATION',condition=name,cap_gib=cap,at=time.time()))
                supervised('evaluate_supplement.py',['--root',OUT/'attacks/test'/item['measurement_condition'],'--target',c['target'],'--method-lock',lock],name+'_shared_evaluation.log',cap,exclusive,deadline,monitor)
                final=evidence(item,deep=True)
                if not final['evaluation_complete']:raise RuntimeError('Missing target responses')
                state['evaluations'][name]=dict(finished=time.time());state['blocked'].pop(name,None)
                if item['source']=='new_subset_measurement':raw_results(matrix(),selection_lock(matrix()))
                print('[parallel supplement] completed',name,flush=True)
                break
            except YieldForOwner as exc:
                state['last_yield']=dict(condition=name,reason=str(exc),at=time.time());atomic_json(statepath,state)
                if time.time()>=deadline:break
                time.sleep(30)
            except Exception as exc:
                state['failures'].append(dict(condition=name,error=str(exc),at=time.time()));state['blocked'][name]=str(exc)
                print('[parallel supplement local failure]',name,str(exc),flush=True);break
            finally:
                atomic_json(statepath,state)
                recent=monitor[-180:]
                atomic_json(bench,dict(baseline=baseline,monitor=recent,peak_global_used_gib=max((r['used_gib'] for r in monitor),default=None),mean_GPU_utilization=statistics.mean(r['utilization'] for r in monitor) if monitor else None,original_primary_settings_changed=False,extra_GPU_workers=1,upper_worker_limit=4,partial_batches_are_resumable=True))
        snapshot(items,owner_active=not exclusive)
        if time.time()>=deadline-900:break
    inventory();table=snapshot(items,owner_active=not exclusive,deep=True)
    state['window_finished']=time.time();atomic_json(statepath,state)
    atomic_json(ROOT/'completion.json',dict(window_finished=True,complete_unique_conditions=sum(r['status']=='COMPLETE' for r in table),total_unique_conditions=47,new_evaluated_conditions=len(state['evaluations']),all47_complete=all(r['status']=='COMPLETE' for r in table),shared_GPU_authorized=True,failures=state['failures'],remaining_work_is_resumable=True))
    from archive_next_stage import upload_compact
    upload_compact(ROOT)
