"""Bounded, single-owner supplement; preserve current owner and checkpoint locally."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from v7_common import OUT,EXP,REGISTRY as BASE_REGISTRY,atomic_json,sha,fingerprint,TIDS
from family_protocol import condition_name
from supplement_registry import registry
from supplement_protocol import ROOT,candidates,evidence,inventory,snapshot
from proxy_selection_tables import matrix,prepare


def admission(remaining_seconds,attack_seconds,evaluation_seconds=1200,safety_seconds=600):
    return remaining_seconds>=attack_seconds+evaluation_seconds+safety_seconds


def command(script,args,log):
    path=ROOT/'logs'/log;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        p=subprocess.run([sys.executable,str(EXP/'src'/script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Exit{p.returncode}; {path}')


def freeze(path,payload):
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('Changed immutable supplement protocol: '+str(path))
    if not path.exists():atomic_json(path,payload)


def lock_protocol(items):
    reg=registry();configs=[r['config'] for r in items if r['source']=='new_supplement']
    lock=dict(protocol='strict_family_disjoint',configs=configs,allowed_test_conditions=[condition_name(c) for c in configs],condition_targets={condition_name(c):c['target'] for c in configs},condition_models={condition_name(c):c['models'] for c in configs},model_revisions=reg,manifest_hashes={t:sha(OUT/'manifests/test'/f'{t}.jsonl') for t in TIDS},reference_hashes={str(i):sha(OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl') for i in range(1,11)},candidate_matrix_sha256=sha(EXP/'multiproxy_8h_supplement_matrix.csv'),source_hashes={n:sha(EXP/'src'/n) for n in ['run_multiproxy_supplement.py','run_parallel_subsets.py','supplement_protocol.py','supplement_registry.py','run_supplement_attack.py','smoke_supplement_models.py','evaluate_supplement.py','run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py']},target_feedback_for_selection=False)
    freeze(ROOT/'method_lock.json',lock)
    from run_proxy_selection import selection_lock
    freeze(OUT/'proxy_selection/method_lock.json',selection_lock(matrix()))


def calibrate(names):
    command('smoke_supplement_models.py',['--models',*names],'model_calibration.log')
    return json.loads((ROOT/'audits/model_gradients.json').read_text())


def estimate(c,audits):
    # Actual per-proxy backward timings; include conservative reload/dispatch overhead.
    return 150*50*sum(audits[n]['seconds_per_image_step'] for n in c['models'])*1.35+60*len(c['models'])


def generate(item,smoke=False,split=None):
    c=item['config'];label=c['id']+('_smoke' if smoke else '_pilot' if split=='dev' else '')
    audits=json.loads((ROOT/'audits/model_gradients.json').read_text())
    batch=min(audits[n]['chosen_batch'] for n in c['models'])
    config=ROOT/'configs'/(c['id']+'.json');freeze(config,c['options'])
    args=['--models',*c['models'],'--method','A3','--fusion','equal','--split','smoke' if smoke else split or 'test','--steps',2 if smoke else 50,'--seed',42,'--restarts',1,'--batch-size',batch,'--run-label',label,'--config-json',config,'--excluded-proxy-family',BASE_REGISTRY[c['target']]['family'],'--input-diversity']
    if smoke:args+=['--transitions','T01','--limit-per-direction',2]
    if split=='dev':args+=['--limit-per-direction',2]
    command('run_supplement_attack.py',args,label+'.log')
    from run_family_disjoint import validate_attack
    if split!='dev':
        # Validation imports are process-local and already use base names for
        # target family. Extra proxies require an explicit combined lookup.
        import run_family_disjoint
        run_family_disjoint.REGISTRY=registry()
        name=condition_name(c)
        if smoke:name=name.replace(c['id']+'__',label+'__',1).replace('__i50__','__i2__')
        return validate_attack(c,OUT/'attacks'/('smoke' if smoke else 'test')/name,smoke=smoke)
    return None


def evaluate(item):
    c=item['config'];lock=OUT/'proxy_selection/method_lock.json' if item['source']=='new_subset_measurement' else ROOT/'method_lock.json'
    command('evaluate_supplement.py',['--root',OUT/'attacks/test'/item['measurement_condition'],'--target',c['target'],'--method-lock',lock],c['id']+'_evaluation.log')


def run_optional(items,state,deadline):
    # Only predeclared N4/N5/N6: compare on disjoint DEV, never target ASR.
    optional=[r for r in items if r['optional'] and any(a.endswith(('_D2','_D2_A','_D2_AB')) for a in r['aliases'])]
    bytarget={}
    for item in optional:bytarget.setdefault(item['config']['target'],[]).append(item)
    validators={'Qwen3.5-27B':['InternVL3.5-4B-HF','Gemma 4 E4B-it'],'InternVL3.5-14B-HF':['Qwen3.5-4B','Gemma 4 E4B-it'],'Gemma 4 31B-it':['Qwen3.5-4B','InternVL3.5-4B-HF']}
    for target,rows in bytarget.items():
        # Do not start pilot family without conservative full-evaluation reserve.
        if deadline-time.time()<3*3600:break
        scores=[]
        for item in rows:
            c=item['config']
            try:
                audit=calibrate(c['models'])
                eta=estimate(c,audit)
                if not admission(deadline-time.time(),eta+eta/15*3):break
                generate(item,smoke=True);generate(item,split='dev')
                pilot=condition_name({**c,'id':c['id']+'_pilot'})
                command('evaluate_supplement.py',['--root',OUT/'attacks/dev'/pilot,'--validators',*validators[target]],c['id']+'_surrogate_validation.log')
                measurements=[]
                for v in validators[target]:
                    for f in (OUT/'development_evaluations'/pilot/v.replace('/','_')).glob('T*/batch_*/metrics.json'):measurements+=json.loads(f.read_text())['images']
                if len(measurements)!=20:raise RuntimeError('Incomplete ten-image/two-validator pilot')
                scores.append(dict(key=item['key'],surrogate_hit_rate=sum(r['representation_hit'] for r in measurements)/20,mean_margin=sum(r['target_margin'] for r in measurements)/20,estimated_full_seconds=eta))
            except Exception as exc:state['failures'].append(dict(condition=c['id'],stage='optional_dev',error=str(exc)))
        if not scores:continue
        chosen=max(scores,key=lambda r:(r['surrogate_hit_rate'],r['mean_margin'],-r['estimated_full_seconds']))
        decision=dict(target=target,dev_scores=scores,selected_key=chosen['key'],target_outputs_used=False,criterion='representation_hit_then_margin_then_lower_runtime')
        freeze(ROOT/'decisions'/(target.replace('/','_')+'.json'),decision)
        item=next(r for r in rows if r['key']==chosen['key'])
        try:
            if admission(deadline-time.time(),chosen['estimated_full_seconds']):
                generate(item);evaluate(item);checked=evidence(item,deep=True)
                if checked['evaluation_complete']:state['evaluations'][item['config']['id']]=dict(finished=time.time())
        except Exception as exc:state['failures'].append(dict(condition=item['config']['id'],stage='optional_full',error=str(exc)))
        atomic_json(ROOT/'state.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory-only',action='store_true');p.add_argument('--share-current-gpu',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);budget=ROOT/'budget.json'
    if not budget.exists():atomic_json(budget,dict(started=time.time(),deadline=time.time()+8*3600,hours=8,existing_owner_must_not_be_interrupted=True))
    deadline=json.loads(budget.read_text())['deadline'];items=candidates();inventory()
    if a.share_current_gpu and not a.inventory_only:
        from run_parallel_subsets import main as shared_main
        shared_main(items,deadline);return
    lockfile=(OUT/'campaign.lock').open('a')
    try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
    except BlockingIOError:owned=False
    snapshot(items,owner_active=not owned,deep=True)
    if a.inventory_only:return
    while not owned and time.time()<deadline:
        atomic_json(ROOT/'current_stage.json',dict(stage='WAITING_FOR_EXISTING_GPU_OWNER',at=time.time(),deadline=deadline))
        print('Waiting for existing GPU owner; remaining hours',(deadline-time.time())/3600,flush=True)
        time.sleep(60)
        try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
        except BlockingIOError:pass
        if int(time.time())%300<65:inventory();snapshot(items,owner_active=not owned)
    statepath=ROOT/'state.json';state=json.loads(statepath.read_text()) if statepath.exists() else dict(started=time.time(),failures=[],blocked={},attacks={},evaluations={})
    if owned and deadline-time.time()>1800:
        from run_proxy_selection import require_recovery_complete
        try:require_recovery_complete()
        except Exception as exc:state['blocked']['global_prerequisite']=str(exc);atomic_json(statepath,state);snapshot(items);return
        # Besides the flock, refuse foreign CUDA users not participating in this owner.
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if processes:raise RuntimeError('Uncoordinated CUDA owner still present: '+processes)
        lock_protocol(items)
        # Missing singles are deliberately NOT launched in this supplement.
        required=[r for r in items if not r['optional'] and len(r['config']['models'])>1 and r['source'] not in ('corrected_family_disjoint','historical_valid_Gemma_cross_family')]
        required.sort(key=lambda r:(r['priority'],0 if 'selection_7subsets' in r['buckets'] else 1,r['config']['id']))
        for item in required:
            c=item['config'];name=c['id']
            try:
                e=evidence(item,deep=True)
                if e['evaluation_complete']:continue
                if not admission(deadline-time.time(),0):break
                atomic_json(ROOT/'current_stage.json',dict(stage='CALIBRATION',condition=name,at=time.time()))
                audit=calibrate(c['models']);eta=estimate(c,audit)
                atomic_json(ROOT/'audits'/f'{name}_admission.json',dict(estimated_attack_seconds=eta,remaining_seconds=deadline-time.time(),admitted=admission(deadline-time.time(),eta),method='measured proxy gradients plus35pct runtime margin and60s/model reload'))
                if not e['attack_complete']:
                    if not admission(deadline-time.time(),eta):continue
                    atomic_json(ROOT/'current_stage.json',dict(stage='ATTACK',condition=name,at=time.time()))
                    generate(item,smoke=True);generate(item)
                state['attacks'][name]=evidence(item,deep=True)['attack_complete'];atomic_json(statepath,state)
                atomic_json(ROOT/'current_stage.json',dict(stage='TARGET_EVALUATION',condition=name,at=time.time()))
                evaluate(item);final=evidence(item,deep=True)
                if not final['evaluation_complete']:raise RuntimeError('Missing target evaluation cells')
                state['evaluations'][name]=dict(finished=time.time())
                state['blocked'].pop(name,None)
                if item['source']=='new_subset_measurement':
                    from run_proxy_selection import raw_results,selection_lock
                    plan=matrix();raw_results(plan,selection_lock(plan))
            except Exception as exc:
                state['failures'].append(dict(condition=name,stage='supplement',error=str(exc)));state['blocked'][name]=str(exc);print('[local failure]',name,str(exc),flush=True)
            atomic_json(statepath,state);snapshot(items)
        # Optional work is gated on completion of the required multi-proxy rows.
        measured=snapshot(items)
        if all(any(q['unique_key']==r['key'] and q['status']=='COMPLETE' for q in measured) for r in required):run_optional(items,state,deadline)
    state['window_finished']=time.time();state['budget_expired']=time.time()>=deadline;atomic_json(statepath,state)
    inventory();table=snapshot(items,owner_active=not owned,deep=True)
    events=json.loads((OUT/'audits/model_dtype_audit.json').read_text())
    atomic_json(ROOT/'audits/bf16_events.json',[e for e in events if e.get('audited_utc','')>=time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(json.loads(budget.read_text())['started']))])
    atomic_json(ROOT/'completion.json',dict(window_finished=True,all47_complete=all(r['status']=='COMPLETE' for r in table),complete_unique_conditions=sum(r['status']=='COMPLETE' for r in table),total_unique_conditions=47,elapsed_since_window_start=time.time()-json.loads(budget.read_text())['started'],new_evaluated_conditions=len(state['evaluations']),failures=state['failures'],remaining_work_is_resumable=True))
    from archive_next_stage import upload_compact
    upload_compact(ROOT)
    print('Supplement window finished; complete/partial statuses:',ROOT/'reports/multiproxy_8h_report.md',flush=True)


if __name__=='__main__':main()
