"""Integration into the one existing GPU owner; separate CPU-only services."""
import os,sys,subprocess,signal,time
from pathlib import Path
import common as C
from paper_closeout.plan import OUT,plan

def enabled():return (OUT/'activation.json').exists()

def service(name,deadline):
    path=OUT/(name+'_lease.json');old=C.read(path)
    if old and C.same_process(old['identity']):return 'RUNNING'
    result=C.read(OUT/(name+'_driver_result.json'))
    if result:return result['status']
    if old:
        C.atomic(OUT/(name+'_driver_result.json'),dict(status='BLOCKED_UNEXPECTED_SERVICE_EXIT',lease=old));return 'BLOCKED'
    script=Path(__file__).parent/(name+'.py');log=OUT/(name+'.log');OUT.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        child=subprocess.Popen([sys.executable,str(script),'--deadline',str(deadline)],stdout=f,stderr=subprocess.STDOUT,
          env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},start_new_session=True)
    C.atomic(path,dict(identity=C.await_identity(child.pid,str(script)),CPU_only=True,started=time.time()))
    return 'RUNNING'

def pending(deadline):
    if not enabled():return [],[],True
    cc,jobs,lock=plan();commercial=service('commercial',deadline);analysis=service('analysis',deadline)
    from stage1_methods.selected_confirmation import large_done
    final_ready=bool(lock) and all(large_done(v['condition'],'evaluate_large_'+fam) for fam,values in (lock or {}).get('families',{}).items() for v in values.values())
    ready=not cc and not jobs and final_ready and commercial=='COMPLETE'
    C.atomic(OUT/'status.json',dict(at=time.time(),pending=len(cc),pending_large=len(jobs),commercial=commercial,analysis=analysis,complete=ready,final_frozen=bool(lock)))
    return cc,jobs,ready

def stop():
    for name in ['commercial','analysis']:
        old=C.read(OUT/(name+'_lease.json'))
        if old and C.same_process(old['identity']):os.kill(old['identity']['pid'],signal.SIGTERM)
