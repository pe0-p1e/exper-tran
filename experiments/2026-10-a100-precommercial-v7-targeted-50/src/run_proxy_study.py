"""18 elapsed hours, four total GPU workers, predictive admission, local failures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from proxy_study_protocol import (ROOT, OUT, EXP, atomic_json, plan, checked, lock,
                                  report, sha, records, TIDS)
from supplement_registry import registry, verify_existing_data


def gpu():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(text) != 1:
        raise RuntimeError('This launcher requires exactly one verified GPU')
    memory, util = map(float, text[0].split(','))
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return dict(used_gib=memory/1024, utilization=util,
                pids=[int(p.strip()) for p in pids if p.strip().isdigit()])


def primary_stage():
    path = OUT / 'family_disjoint_recovery/current_stage.json'
    return json.loads(path.read_text()) if path.exists() else {'stage': 'unknown'}


def alive(pid):
    path = Path('/proc') / str(pid) / 'stat'
    return path.exists() and path.read_text().split()[2] != 'Z'


def stop(job):
    if not alive(job['pid']):
        return
    cmd = (Path('/proc') / str(job['pid']) / 'cmdline').read_text()
    if 'proxy_study_worker.py' not in cmd:
        raise RuntimeError('Refuse to signal a foreign PID')
    os.killpg(job['pid'], signal.SIGTERM)


def weight_events():
    result = {}
    for p in [OUT / 'audits/model_dtype_audit.json',
              OUT / 'multiproxy_8h_supplement/audits/model_dtype_audit.json',
              ROOT / 'audits/model_dtype_audit.json']:
        if p.exists():
            for e in json.loads(p.read_text()):
                if e.get('status') == 'pass' and e.get('all_floating_parameters_bf16'):
                    result[e['model_name']] = e
    return result


def reservation(names, phase, events):
    specs = registry(); full = []; resident = []; activation = []
    for n in names:
        event = events[n]
        if event['revision'] != specs[n]['revision']:
            raise RuntimeError('Pinned revision differs from memory audit: ' + n)
        size = sum(event['parameter_bytes_by_dtype'].values()) / 2**30
        visual = sum(event['vision_encoder_parameter_bytes_by_dtype'].values()) / 2**30
        r = visual if specs[n]['family'] == 'gemma4' else size
        full.append(size); resident.append(r)
        p = ROOT / 'calibration' / (n.replace('/', '_') + '.json')
        measurements = json.loads(p.read_text())['measurements'] if p.exists() else []
        activation.append(max(1.5, max((m['peak_reserved_gib']-r for m in measurements), default=2.5)))
    if phase == 'evaluation':
        return full[0] + 3.5
    # Frozen runner offloads proxies sequentially if resident weights exceed 35GiB.
    held = sum(resident) if sum(resident) <= 35 else max(resident)
    return max(max(full)+1.0, held+max(activation)) + 1.5


def eligible(primary_reserved, active_reservations, request, workers):
    return workers < 4 and primary_reserved + sum(active_reservations) + request <= 72


def checkpoint(state, jobs):
    state['active'] = [{k: v for k, v in j.items() if k not in ('proc', 'stream')} for j in jobs]
    atomic_json(ROOT / 'state.json', state)


def phase_for(row, done, state):
    c = row['item']['config']; name = c['id']
    if done['evaluation_complete']:
        return None, []
    if done['attack_complete']:
        return 'evaluation', [c['target']]
    missing = [n for n in c['models'] if not (ROOT / 'calibration' / (n.replace('/', '_') + '.json')).exists()]
    if missing:
        return 'calibration', missing
    if not state.get('smokes', {}).get(name):
        return 'smoke', c['models']
    return 'attack', c['models']


def args_for(row, phase, names):
    item = row['item']; c = item['config']
    if phase == 'calibration':
        return ['calibration', *names]
    if phase == 'evaluation':
        source = item['source']
        path = (ROOT / 'method_lock.json' if source == 'study18' else
                OUT / 'proxy_selection/method_lock.json' if source == 'new_subset_measurement' else
                OUT / 'family_disjoint_recovery/method_lock.json' if source == 'corrected_family_disjoint' else
                OUT / 'ten_hour_research/selected_method_lock.json' if source == 'historical_valid_Gemma_cross_family' else
                OUT / 'multiproxy_8h_supplement/method_lock.json')
        return ['evaluation', '--root', OUT / 'attacks/test' / item['measurement_condition'],
                '--target', c['target'], '--method-lock', path]
    cfg = ROOT / 'configs' / (c['id'] + '.json')
    atomic_json(cfg, c['options'])
    args = ['attack', '--models', *c['models'], '--method', 'A3', '--fusion', 'equal',
            '--seed', 42, '--restarts', 1, '--batch-size', 2, '--config-json', cfg,
            '--excluded-proxy-family', registry()[c['target']]['family'], '--input-diversity',
            '--split', 'smoke' if phase == 'smoke' else 'test', '--steps', 2 if phase == 'smoke' else 50,
            '--run-label', c['id'] + ('_smoke' if phase == 'smoke' else '')]
    if phase == 'smoke':
        args += ['--transitions', 'T01', '--limit-per-direction', 2]
    return args


def paired_report(rows):
    """Paired successes, bootstrap uncertainty and exact discordant-binomial tests."""
    import numpy as np
    from scipy.stats import binomtest
    from proxy_selection_tables import writecsv
    contrasts = [('M07','M16'), ('M07','M17'), ('M08','M09'), ('M08','M10'),
                 ('M13','M14'), ('M13','M15'), ('M14','M15'), ('M15','M18'),
                 ('M15','M27'), ('M15','M28'), ('M15','M29'), ('M15','M30')]
    by = {(r['template'], r['item']['config']['target']): r['item'] for r in rows}
    output = []; rng = np.random.default_rng(42)
    for a, b in contrasts:
        for target in sorted({k[1] for k in by}):
            x = checked(by[(a,target)]); y = checked(by[(b,target)])
            if not x['evaluation_complete'] or not y['evaluation_complete']:
                continue
            xx = {(v['transition'],v['image_id']): int(v['target_hit']) for v in x['detailed']}
            yy = {(v['transition'],v['image_id']): int(v['target_hit']) for v in y['detailed']}
            if xx.keys() != yy.keys() or len(xx) != 150:
                raise RuntimeError('Paired cohort differs')
            delta = np.array([yy[k]-xx[k] for k in sorted(xx)]).reshape(5,30)
            boot = np.stack([delta[i, rng.integers(0,30,(4000,30))].mean(1) for i in range(5)]).mean(0)
            gains = int((delta==1).sum()); losses = int((delta==-1).sum())
            output.append(dict(target=target, baseline=a, comparison=b, N=150,
                               ASR_difference=float(delta.mean()), CI_low=float(np.quantile(boot,.025)),
                               CI_high=float(np.quantile(boot,.975)), paired_gains=gains, paired_losses=losses,
                               discordant_binomial_p=binomtest(gains,gains+losses).pvalue if gains+losses else 1.,
                               multiple_comparisons='exploratory unadjusted; no significance-based selection'))
    if output:
        writecsv(ROOT / 'reports/paired_contrasts.csv', output)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preflight', action='store_true')
    a = p.parse_args(); ROOT.mkdir(parents=True, exist_ok=True)
    handle = (ROOT / 'scheduler.lock').open('a'); fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_existing_data(); rows = plan(); lock(rows)
    if len(rows) != 96:
        raise RuntimeError('Expected 32 templates x three targets')
    if a.preflight:
        print('PASS: manifests, 96 unique predeclared conditions, source lock, strict family caps')
        return
    budget = ROOT / 'budget.json'
    if not budget.exists():
        atomic_json(budget, dict(started=time.time(), deadline=time.time()+18*3600, hours=18))
    b = json.loads(budget.read_text()); deadline = b['deadline']
    statepath = ROOT / 'state.json'
    state = json.loads(statepath.read_text()) if statepath.exists() else dict(started=b['started'], smokes={}, failures=[], blocked={}, retries={}, active=[], memory_events=[])
    # Resume adoption, never duplicate a live child after controller interruption.
    jobs = []
    for j in state.get('active', []):
        if alive(j['pid']):
            cmd = (Path('/proc') / str(j['pid']) / 'cmdline').read_text()
            if 'proxy_study_worker.py' not in cmd:
                raise RuntimeError('Saved PID is foreign')
            jobs.append(j)
    tasks = {r['item']['config']['id']: r for r in rows}
    done = {}
    for r in rows:
        n = r['item']['config']['id']
        try:
            e = checked(r['item'], deep=True)
            done[n] = {k:e[k] for k in ['attack_complete','evaluation_complete']}
        except Exception as exc:
            done[n] = dict(attack_complete=False,evaluation_complete=False)
            state['blocked'][n] = 'Evidence validation failed: '+str(exc)
    atomic_json(ROOT / 'audits/start_inventory.json',dict(at=time.time(),gpu=gpu(),primary=primary_stage(),
                completed=sum(d['evaluation_complete'] for d in done.values()), original_outputs_unchanged=True))
    print('[study] validated existing complete conditions:',sum(d['evaluation_complete'] for d in done.values()),flush=True)
    global_lock = (OUT / 'campaign.lock').open('a'); exclusive = False; last_report = 0; samples = []
    try:
        while time.time() < deadline:
            now=time.time()
            try:
                fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB); exclusive=True
            except BlockingIOError:
                exclusive=False
            g=gpu(); primary=primary_stage()
            yielding=(not exclusive and primary['stage']!='attack') or g['used_gib']>72 or len(g['pids'])>4
            if yielding:
                for j in jobs:
                    j['yielded']=True; stop(j)
            for j in list(jobs):
                proc=j.get('proc')
                finished=proc.poll() is not None if proc else not alive(j['pid'])
                if not finished:
                    continue
                jobs.remove(j)
                if j.get('stream'):
                    j['stream'].close()
                n=j['condition']; row=tasks[n]
                if j.get('yielded'):
                    state['memory_events'].append(dict(at=now,condition=n,reason='yield to original owner or global memory guard'))
                    continue
                ok=proc.returncode==0 if proc else False
                if not proc:
                    if j['phase']=='calibration':
                        ok=all((ROOT/'calibration'/(m.replace('/','_')+'.json')).exists() for m in j['models'])
                    elif j['phase'] in ['attack','evaluation']:
                        ok=checked(row['item'])['evaluation_complete' if j['phase']=='evaluation' else 'attack_complete']
                    else:
                        ok=False  # A stopped smoke is safe to repeat.
                if ok:
                    if j['phase']=='smoke':
                        state['smokes'][n]=True
                    if j['phase'] in ['attack','evaluation']:
                        e=checked(row['item'],deep=True)
                        done[n]={k:e[k] for k in ['attack_complete','evaluation_complete']}
                        state['blocked'].pop(n,None)
                    print('[study completed]',n,j['phase'],flush=True)
                else:
                    log=Path(j['log']); error=log.read_text()[-1800:] if log.exists() else 'No child log'
                    state['failures'].append(dict(at=now,condition=n,phase=j['phase'],error=error))
                    retry=state['retries'].setdefault(n,dict(count=0,reserve_floor=0))
                    retry['count']+=1; retry['after']=now+120
                    if 'out of memory' in error.lower():
                        retry['reserve_floor']=j['reserve']+4
                    elif retry['count']>=2:
                        state['blocked'][n]=error
            checkpoint(state,jobs)
            if now-last_report>180:
                table=report(rows,state);paired_report(rows)
                print('[study progress]',sum(x['status']=='COMPLETE' for x in table),'/96 conditions; remaining hours',round((deadline-now)/3600,2),flush=True)
                last_report=now
            samples.append(dict(at=now,**g,primary=primary,active=[dict(condition=j['condition'],phase=j['phase'],reserved_gib=j['reserve']) for j in jobs]))
            atomic_json(ROOT/'audits/performance_benchmark.json',dict(samples=samples[-240:],max_global_used_gib=max(s['used_gib'] for s in samples),max_total_gpu_workers=max(len(s['pids']) for s in samples),strict_bf16=True,primary_source_unchanged=True))
            if yielding:
                time.sleep(2); continue
            names={j['condition'] for j in jobs}; reserves=[j['reserve'] for j in jobs]
            primary_reserve=0 if exclusive else max(10.5,g['used_gib']-sum(reserves))
            events=weight_events()
            # Fixed template order; finish all targets for a contrast before extensions.
            pending=[r for r in rows if not done[r['item']['config']['id']]['evaluation_complete'] and r['item']['config']['id'] not in names]
            core_remaining=any(r['tier']=='core' and not done[r['item']['config']['id']]['evaluation_complete'] for r in rows)
            dispatched=False
            for r in pending:
                c=r['item']['config']; n=c['id']
                if n in state['blocked']:
                    continue
                if core_remaining and r['tier']!='core':
                    continue
                retry=state['retries'].get(n,{})
                if retry.get('after',0)>now or retry.get('count',0)>=3:
                    continue
                phase,models=phase_for(r,done[n],state)
                # Coordinate old jobs rather than starting duplicates.
                if r['item']['source']=='corrected_family_disjoint' and not exclusive:
                    continue
                if phase=='calibration' and any(j['phase']=='calibration' for j in jobs):
                    continue
                if phase=='calibration' and any(set(models)&set(j['models']) for j in jobs):
                    continue
                try:
                    request=max(retry.get('reserve_floor',0),reservation(models,'evaluation' if phase=='evaluation' else 'attack',events))
                except Exception as exc:
                    state['blocked'][n]='No safe pinned-model memory estimate: '+str(exc);continue
                # Permit at most four in total, counting primary and foreign GPU users.
                workers=max(len(g['pids']),len(jobs)+(0 if exclusive else 1))
                if not eligible(primary_reserve,reserves,request,workers):
                    continue
                if deadline-now<900 or phase=='attack' and deadline-now<2700:
                    continue
                log=ROOT/'logs'/(n+'_'+phase+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                stream=log.open('a')
                env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(request-.4)}
                proc=subprocess.Popen([sys.executable,str(EXP/'src/proxy_study_worker.py'),*map(str,args_for(r,phase,models))],
                                      env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                jobs.append(dict(proc=proc,stream=stream,pid=proc.pid,condition=n,phase=phase,
                                 models=models,reserve=request,log=str(log),started=now))
                atomic_json(ROOT/'admissions'/(n+'_'+phase+'.json'),dict(at=now,estimated_reserved_gib=request,
                            primary_reserved_gib=primary_reserve,other_reservations=reserves,max_workers=4,global_budget_gib=72))
                print('[study dispatch]',n,phase,'reserved',round(request,2),'GiB',flush=True)
                checkpoint(state,jobs);dispatched=True;break
            if not pending:
                break
            time.sleep(2 if dispatched else 5)
    finally:
        for j in jobs:
            stop(j)
        for j in jobs:
            try:
                if j.get('proc'):
                    j['proc'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(j['pid'],signal.SIGKILL);j['proc'].wait()
        checkpoint(state,[])
        table=report(rows,state);paired_report(rows)
        state['window_finished']=time.time();state['budget_expired']=time.time()>=deadline
        atomic_json(ROOT/'state.json',state)
        atomic_json(ROOT/'completion.json',dict(complete=sum(x['status']=='COMPLETE' for x in table),total=96,
                    core_complete=sum(x['status']=='COMPLETE' and x['tier']=='core' for x in table),core_total=54,
                    elapsed_hours=(time.time()-b['started'])/3600,remaining_work_resumable=True,failures=state['failures']))


if __name__=='__main__':
    main()
