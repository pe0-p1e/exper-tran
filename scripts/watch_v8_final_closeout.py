#!/usr/bin/env python3
"""Finalize the existing CPU commercial campaign, without launching GPU work."""
import fcntl,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from paper_closeout.plan import OUT

def refresh():
    subprocess.run([sys.executable,str(BASE/'scripts/finalize_v8_statistics.py')],check=True,
                   env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})

def finish(complete):
    # Stop only the verified owned analysis process; let it flush its own writes.
    lease=C.read(OUT/'analysis_lease.json',{}).get('identity')
    if lease and C.same_process(lease):
        os.kill(lease['pid'],signal.SIGTERM)
        end=time.time()+600
        while C.same_process(lease) and time.time()<end:time.sleep(3)
        if C.same_process(lease):raise RuntimeError('Analysis did not drain; preserve checkpoint, do not kill')
    refresh()
    from paper_closeout.analysis import calculate,projections
    calculate();projections()
    if (OUT/'reports/commercial_cancellation.json').exists():
        p=OUT/'reports/FINAL_REPORT.md';p.write_text(p.read_text()+'\n## User cancellation\nRemaining GPT5.6 requests were cancelled by the user because some requests waited around five minutes. Completed partial results are preserved; no further GPT5.6 requests are scheduled. Other completed results are unchanged.\n')
    validation=C.read(OUT/'reports/independent_validation.json')
    if complete and (validation['errors'] or validation['incomplete_conditions']):raise RuntimeError('Final audit has missing or invalid conditions')
    status='COMPLETE' if complete else ('FINISHED_WITH_USER_CANCELLATIONS' if (OUT/'reports/commercial_cancellation.json').exists() else 'PARTIAL_BLOCKED')
    C.atomic(OUT/'status.json',dict(at=time.time(),state=status,complete=complete,scope='User-authorized paper closeout only; historical stage1 states preserved',
        pending=validation['incomplete_conditions']-validation.get('cancelled_conditions',0),cancelled=validation.get('cancelled_conditions',0),analysis='COMPLETE',final_frozen=True,commercial=C.read(OUT/'commercial_driver_result.json')))
    # Full experiment records remain on root disk; publish just this compact receipt.
    root_mount=subprocess.check_output(['findmnt','-n','-o','SOURCE','--target',str(C.ROOT)],text=True).strip()
    if root_mount!='/dev/vda1':raise RuntimeError('Expected persistent root device; archive requires storage review: '+root_mount)
    links=[str(p) for p in C.ROOT.rglob('*') if p.is_symlink()]
    if links:raise RuntimeError('Resolve output symlinks before full persistent archive: '+str(links[:5]))
    backup=Path('/home/ubuntu/experiment-checkpoints');backup.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    C.atomic(OUT/'reports/final_checkpoint.json',dict(status=status,persistent_root=str(C.ROOT),device=root_mount,source_commit=commit,
        full_PNGs_raw_CLI_failed_attempts_preserved=True,compact_GitHub_only=True,at=time.time()))
    from paper_closeout.analysis import publish
    publish()
    code=backup/('v8-source-'+commit[:12]+'.tar.gz')
    if not code.exists():
        with code.open('wb') as f:subprocess.run(['git','archive','--format=tar.gz',commit],cwd=BASE,stdout=f,check=True)
    target=backup/('v8-closeout-'+status.lower()+'-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.tar.gz')
    temp=target.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz',compresslevel=1) as archive:archive.add(C.ROOT,arcname=C.ROOT.name,recursive=True)
    temp.replace(target)
    C.atomic(backup/(target.name+'.json'),dict(status=status,archive=str(target),sha256=C.sha(target),bytes=target.stat().st_size,
        code_archive=str(code),source_commit=commit,full_results_persistent=True))
    C.atomic(OUT/'finalizer_result.json',dict(status=status,archive=str(target),source_archive=str(code),at=time.time()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        C.atomic(OUT/'finalizer_lease.json',dict(identity=C.proc_identity(os.getpid()),CPU_only=True))
        while True:
            refresh()
            result=C.read(OUT/'commercial_driver_result.json',{})
            if result.get('status')=='COMPLETE':finish(True);return 0
            # No further request can run after all remaining conditions reach
            # bounded timeout/manual-review blocks. Do not spin forever or fake N.
            summary=C.read(OUT/'commercial150/summary.json',{});remaining=[r for r in summary.get('methods',[]) if r['completed_pairs']!=r['N']]
            blocked=[]
            for r in remaining:
                key=C.digest(['commercial150',r['model'],r['method']])[:18]
                b=C.read(OUT/'commercial150/blocked'/key,{})
                if b.get('recovery_error'):blocked.append(b)
            active=C.read(OUT/'commercial_active.json',{}).get('identity')
            if result.get('status','').startswith('BLOCKED') or remaining and len(blocked)==len(remaining) and not (active and C.same_process(active)):
                C.atomic(OUT/'commercial_driver_result.json',dict(status='BLOCKED_REQUIRES_INTERVENTION',blocks=blocked,at=time.time()))
                finish(False);return 2
            C.atomic(OUT/'finalizer_heartbeat.json',dict(at=time.time(),remaining_conditions=len(remaining),state='WAITING_FOR_EXISTING_COMMERCIAL_DRIVER'))
            time.sleep(120)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        import traceback
        C.atomic(OUT/'finalizer_error.json',dict(error=str(exc),traceback=traceback.format_exc(),at=time.time()));raise
