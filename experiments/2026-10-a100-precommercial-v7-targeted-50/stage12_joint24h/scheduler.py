"""One admission owner, existing GPU lock/predictions, restartable job leases."""
from common import *
import argparse,subprocess,signal,traceback,random
from resources import telemetry,estimate,admit,failure_kind
from stage1_methods.parallel_owner import research_pending,stage2_progress,stop_stage2
STOP=False

def fair_pending(primary,background,active,legacy_ids):
 # Fair queue lanes, single admission owner; no second scheduler or forced parallelism.
 queues=[list(primary),list(background)];counts=[sum(j['condition'] not in legacy_ids for j in active),sum(j['condition'] in legacy_ids for j in active)]
 lane=0 if counts[0]<=counts[1] else 1;out=[];seen=set()
 while queues[0] or queues[1]:
  if not queues[lane]:lane=1-lane
  c,mode=queues[lane].pop(0);key=(c['id'],mode)
  if key not in seen:out.append((c,mode));seen.add(key)
  lane=1-lane
 return out

def save_job(c,mode):
 legal(c['layers'],c['validator']);cid=c['id'];condition_path=ROOT/'conditions'/(cid+'.json')
 existing=read(condition_path)
 if existing and scientific(existing)!=scientific(c):raise RuntimeError('Condition ID collision')
 if not existing:atomic(condition_path,c)
 else:c=existing
 jid=cid+'_'+mode;freeze(ROOT/'jobs'/(jid+'.json'),dict(condition=c,mode=mode));return jid

def complete(j):
 c=read(ROOT/'jobs'/(j['id']+'.json'))['condition']
 if j['mode']=='calibrate_resident':return (ROOT/'calibration_resident_v2'/(c['id']+'.json')).exists()
 if j['mode']=='evaluate30':
  from stage1_methods.early_confirm import done
  return done(c)
 if j['mode']=='ila':
  from ila_selection import done
  return done(c)
 if j['mode']=='ila_smoke':
  from ila_selection import smoke_done
  return smoke_done(c)
 return (ROOT/'calibration'/(c['id']+'.json')).exists() if j['mode']=='calibrate' else attack_done(c) if j['mode']=='attack' else eval_done(c)

def state_write(s):
 atomic(ROOT/'scheduler.json',s)
 try:t=telemetry()
 except Exception:t={}
 deadline=read(ROOT/'window.json',{}).get('deadline',time.time())
 status=dict(at=time.time(),state=s.get('state','RUNNING'),stage=s.get('phase'),active=s.get('active',[]),failures=s.get('failures',[]),
  gpu=t,remaining_hours=max(0,deadline-time.time())/3600,conditions_admitted=len(s.get('admitted',[])),completed_jobs=len(s.get('completed',[])),
  max_workers=s.get('worker_limit',1),method_frozen=(ROOT/'method_lock.json').exists(),stage2=read(ROOT/'stage2/summary.json',{}))
 from stage1_methods.early_confirm import summarize as early_summary
 status['confirmation30']=early_summary()
 atomic(ROOT/'status.json',status)
 (ROOT/'STATUS.md').write_text('# V8 live status\n\n'+json.dumps(status,indent=2)+'\n')

def own_signal(j,sig):
 if same_process(j['process_identity']) and str(HERE/'worker.py') in j['process_identity']['command']:
  os.killpg(j['process_identity']['pid'],sig);return True
 return False

def reconcile(s):
 active=[]
 for j in s['active']:
  if same_process(j['process_identity']):active.append(j);continue
  if complete(j):s['completed'].append(j['id']);continue
  text=Path(j['log']).read_text(errors='replace')[-5000:] if Path(j['log']).exists() else ''
  exit=read(ROOT/'attempts'/j['id']/str(j['attempt'])/'exit.json',{})
  if exit.get('status')=='STOPPED':continue
  kind=exit.get('kind') or (failure_kind(text,0) if exit or 'Traceback' in text else 'resource_or_interrupt')
  s['failures'].append(dict(job=j['id'],kind=kind,at=time.time(),log=j['log'],error=text[-1500:]))
  attempts=s['attempts'].get(j['id'],0)+1;s['attempts'][j['id']]=attempts
  if kind=='code' or attempts>=3:s['blocked'].append(j['id'])
  else:
   s['retry_after'][j['id']]=time.time()+min(300,10*2**attempts)+random.uniform(0,3)
   if kind in ['oom','resource_or_interrupt']:s['worker_limit']=max(1,s['worker_limit']-1);s['microbatch']=1
 s['active']=active

def launch(s,c,mode):
 jid=save_job(c,mode)
 if jid in s['blocked'] or jid in {j['id'] for j in s['active']} or time.time()<s['retry_after'].get(jid,0):return False
 t=telemetry();reservation=estimate(c,mode)
 from resources import resident_profile
 profile=resident_profile(c) if mode=='attack' else None
 worker_cap=max(reservation-.2,profile['load_cap_GiB']) if profile else reservation-.2
 startup_extra=max(0,worker_cap-reservation)
 if not admit(t,s['active'],reservation,s['worker_limit'],startup_extra):return False
 # Recheck actual free memory immediately before admission.
 if not admit(telemetry(),s['active'],reservation,s['worker_limit'],startup_extra):return False
 if mode=='attack' and c['id'] not in s['admitted']:
  if len(s['admitted'])>=270:s['blocked'].append(jid);return False
  s['admitted'].append(c['id'])
 attempt=s['attempts'].get(jid,0);log=ROOT/'logs'/(jid+f'_{attempt}.log');log.parent.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(HERE/'worker.py'),mode,'--job',jid,'--attempt',str(attempt)]
 env={**os.environ,'SUPPLEMENT_MEMORY_CAP_GIB':str(min(worker_cap,t['total']-1)),'V8_RESIDENT_LIMIT_GIB':str(max(0,reservation-3)),'V8_RESERVED_GIB':str(reservation),'V8_STEADY_CAP_GIB':str(reservation-.2) if profile else ''}
 fd=int(os.environ['V8_ADMISSION_FD'])
 with log.open('a') as stream:p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,pass_fds=(fd,),start_new_session=True)
 identity=await_identity(p.pid,str(HERE/'worker.py'))
 s['active'].append(dict(id=jid,mode=mode,condition=c['id'],reservation=reservation,process_identity=identity,attempt=attempt,log=str(log),started=time.time()))
 atomic(ROOT/'scheduler.json',s);print('V8 dispatch',jid,round(reservation,2),'GiB',flush=True);return True

def run_stage2(s,deadline,count=50):
 from stage2_cli import prepare_packs,summarize
 s['phase']='STAGE2_'+str(count);state_write(s)
 try:packs=prepare_packs(count=count)
 except Exception as e:
  atomic(ROOT/'stage2/blocker.json',dict(status='BLOCKED_METHOD_OR_ARTIFACT',reason=str(e)));return False
 # Known complete CLI chain is reused; failures stop without account/model rotation.
 for limit in [20,2*count]:
  for pack in packs:
   if time.time()>=deadline-3600 or STOP:return False
   log=ROOT/'logs'/f'stage2_{count}_{pack.name}_{limit}.log'
   with log.open('a') as f:
    p=subprocess.Popen([sys.executable,str(HERE/'stage2_cli.py'),'--pack',str(pack),'--limit',str(limit)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   atomic(ROOT/'stage2/active.json',dict(process_identity=await_identity(p.pid,str(HERE/'stage2_cli.py')),method=pack.name,limit=limit,started=time.time()))
   while p.poll() is None:
    if time.time()>=deadline-3600 or STOP:
     p.terminate()
     try:p.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
     return False
    state_write(s);time.sleep(15)
   if p.returncode:
    atomic(ROOT/'stage2/blocker.json',dict(status='BLOCKED_CLI',log=str(log),exit=p.returncode,reason=log.read_text(errors='replace')[-2000:],no_paid_fallback=True));summarize(count=count);return False
 return summarize(count=count)

def main():
 global STOP
 ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--stage2-only',action='store_true');a=ap.parse_args();check_lock()
 def stop(*_):
  global STOP
  STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 window=read(ROOT/'window.json');deadline=window['deadline']
 s=read(ROOT/'scheduler.json',dict(state='RUNNING',phase='CALIBRATION',active=[],completed=[],failures=[],attempts={},retry_after={},blocked=[],admitted=[],worker_limit=1,microbatch=1))
 s['state']='RUNNING'
 # An interrupted Stage2 worker must be joined before another CLI process can start.
 old=read(ROOT/'stage2/active.json')
 if old and same_process(old['process_identity']):
  while same_process(old['process_identity']):state_write(s);time.sleep(15)
 if a.stage2_only:
  if not (ROOT/'method_lock.json').exists():raise RuntimeError('Stage2 requires frozen method')
  ok=run_stage2(s,deadline);s['state']='COMPLETE_STAGE2_ONLY' if ok else 'PARTIAL_OR_BLOCKED';state_write(s)
  from reports import generate
  generate();return 0 if ok else 2
 active_processes=[];last_report=0
 from selection import plan_next,uniform,original_pair_controls
 smokes=[make(fam,uniform(POOLS[fam][:3]),split='smoke',label='smoke',smoke=True) for fam in VALIDATORS]
 while True:
  reconcile(s)
  for j in s['active']:
   beat=read(ROOT/'attempts'/j['id']/str(j['attempt'])/'heartbeat.json',{})
   signature=digest(beat.get('position',{}))
   if signature!=j.get('position_signature'):
    j['position_signature']=signature;j['last_progress']=time.time()
   stale=time.time()-beat.get('at',j['started'])>180
   stalled=time.time()-j.get('last_progress',j['started'])>900
   if stale or stalled:
    if not j.get('term_sent'):
     own_signal(j,signal.SIGTERM);j['term_sent']=time.time()
    elif time.time()-j['term_sent']>60:
     own_signal(j,signal.SIGKILL)
     atomic(ROOT/'attempts'/j['id']/str(j['attempt'])/'exit.json',dict(status='FAILED',kind='timeout',reason='stale heartbeat or stalled progress'))
  if STOP:
   stop_stage2()
   s['state']='STOPPING'
   for j in s['active']:own_signal(j,signal.SIGTERM)
   state_write(s);return 75
  if time.time()>deadline-3600:
   stop_stage2()
   for j in s['active']:own_signal(j,signal.SIGTERM)
   s['state']='PARTIAL_DEADLINE';state_write(s);break
  calibrations=[c for c in smokes if not (ROOT/'calibration_resident_v2'/(c['id']+'.json')).exists()]
  smoke_attacks=[c for c in smokes if not attack_done(c)]
  if calibrations:phase='CALIBRATION';pending=[(c,'calibrate_resident') for c in calibrations]
  elif smoke_attacks:phase='SMOKE';pending=[(c,'attack') for c in smoke_attacks]
  elif any(not eval_done(c) for c in smokes):phase='SMOKE_NATIVE_VALIDATION';pending=[(c,'evaluate') for c in smokes if not eval_done(c)]
  elif a.smoke_only:
   if s['active']:time.sleep(3);continue
   s['state']='SMOKE_PASS';s['phase']='SMOKE';state_write(s);return 0
  else:
   s['worker_limit']=min(8,max(s['worker_limit'],1+len(s['completed'])//3)) if not any(x['kind']=='oom' for x in s['failures']) else s['worker_limit']
   phase,conditions=plan_next(s);pending=[]
   for c in conditions:
    if c['split']=='confirm':
     from stage1_methods.early_confirm import pending as confirm_pending
     pending+=confirm_pending(c);continue
    mode='ila_smoke' if phase=='ILA_DIAGNOSTIC_SMOKE' else 'ila' if phase=='ILA_SOURCE_DIAGNOSTIC' else 'evaluate' if attack_done(c) else 'attack';pending.append((c,mode))
   legacy=original_pair_controls();legacy_ids={c['id'] for c in legacy}
   background=[(c,'evaluate' if attack_done(c) else 'attack') for c in legacy if not eval_done(c)]
   if (ROOT/'method_lock.json').exists():s['commercial_state']=stage2_progress(deadline)
   paper,paper_ready=research_pending(phase)
   paper_background=[(c,'evaluate' if attack_done(c) else 'attack') for c in paper if not eval_done(c)]
   background+=paper_background
   s['paper_research']=dict(pending=len(paper_background),complete=paper_ready,blocked=read(ROOT/'stage1_methods/blocked.json'))
   # Prioritize evaluation to release adaptive search prerequisites.
   pending.sort(key=lambda x:x[1] not in ['evaluate','evaluate30']);background.sort(key=lambda x:x[1] not in ['evaluate','evaluate30'])
   pending=fair_pending(pending,background,s['active'],legacy_ids|{c['id'] for c in paper})
   s['parallel_queues']=dict(primary_stage=phase,original_pairs_completed=sum(eval_done(c) for c in legacy),original_pairs_total=len(legacy),remaining_background=len(background))
   if phase=='STAGE1_COMPLETE':
    commercial=stage2_progress(deadline);s['commercial_state']=commercial
    if commercial!='RUNNING' and not pending and not s['active']:
     s['state']='COMPLETE' if commercial=='COMPLETE' and paper_ready and not s['blocked'] else 'PARTIAL_OR_BLOCKED';state_write(s);break
    phase='STAGE2_AND_STAGE1_RESEARCH'
    state_write(s)
   if time.time()>deadline-6*3600 and not (ROOT/'method_lock.json').exists():
    if s['active']:
     # Finish in-flight microbatches; no new low-priority search admitted.
     for j in s['active']:own_signal(j,signal.SIGTERM)
     state_write(s);time.sleep(5);continue
    s['state']='PARTIAL_STAGE1_TIMEBOX';s['phase']='STAGE2_BLOCKED_NO_VALID_METHOD_LOCK';state_write(s);break
  s['phase']=phase
  for c,mode in pending:
   if len(s['active'])>=s['worker_limit']:break
   launch(s,c,mode)
  # A dead code branch stays explicit; independent pending conditions are attempted first.
  available=[(c,m) for c,m in pending if c['id']+'_'+m not in s['blocked']]
  if not available and not s['active'] and s.get('commercial_state')=='RUNNING':
   time.sleep(3);continue
  if not available and not s['active']:
   s['state']='BLOCKED_REQUIRES_INTERVENTION';state_write(s);break
  if time.time()-last_report>60:
   state_write(s)
   from reports import generate
   if time.time()-s.get('last_analysis',0)>900:
    generate();s['last_analysis']=time.time()
   if time.time()-s.get('last_hourly_summary',0)>3600:
    with (ROOT/'logs/hourly_summary.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),stage=s['phase'],completed=len(s['completed']),failures=len(s['failures']),active=len(s['active'])))+'\n')
    s['last_hourly_summary']=time.time()
   last_report=time.time()
  time.sleep(3)
 from reports import generate
 generate();return 0 if s['state']=='COMPLETE' else 2
if __name__=='__main__':sys.exit(main())
