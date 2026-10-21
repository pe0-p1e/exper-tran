"""External subscription CLI only; no GPU scheduler and no scientific selection."""
import argparse,signal,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C
import scheduler as S

def main():
    a=argparse.ArgumentParser();a.add_argument('--deadline',type=float,required=True);args=a.parse_args()
    C.check_lock()
    def stop(*_):S.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    # Reuse scheduler's tested CLI loop, but never overwrite GPU owner's scheduler/status files.
    S.state_write=lambda state:C.atomic(C.ROOT/'stage2/driver_status.json',dict(at=time.time(),**state))
    old=C.read(C.ROOT/'stage2/active.json')
    while old and C.same_process(old['process_identity']):
        if S.STOP or time.time()>=args.deadline-3600:return 2
        time.sleep(5)
    try:
        from stage1_methods.early_confirm import frozen
        lock=C.read(C.ROOT/'method_lock.json');fam=lock['stage2_primary_family']
        conditions=[C.make(fam,lock['families'][fam]['primary' if method=='selected' else 'uniform'],split='confirm',label=method) for method in lock['stage2_methods']]
        ok=False
        for count in [30,50]:
            while not all(frozen(c,count) for c in conditions):
                C.atomic(C.ROOT/'stage2/driver_status.json',dict(at=time.time(),phase='WAITING_FROZEN_PNGS',count=count))
                if S.STOP or time.time()>=args.deadline-3600:return 2
                time.sleep(15)
            ok=S.run_stage2(dict(phase='STAGE2_'+str(count),active=[],failures=[]),args.deadline,count=count)
            if not ok:break
        C.atomic(C.ROOT/'stage2/parallel_driver_result.json',dict(status='COMPLETE' if ok else 'BLOCKED_OR_PARTIAL',at=time.time()))
        return 0 if ok else 2
    except Exception as e:
        import traceback
        C.atomic(C.ROOT/'stage2/parallel_driver_result.json',dict(status='BLOCKED_OR_PARTIAL',error=str(e),trace=traceback.format_exc()))
        return 2
if __name__=='__main__':sys.exit(main())
