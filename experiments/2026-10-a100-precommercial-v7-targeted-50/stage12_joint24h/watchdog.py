"""Disconnect-safe local supervisor; original deadline never resets on resume."""
from common import *
import argparse,subprocess,signal

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--deadline-hours',type=float,default=24);ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--stage2-only',action='store_true');a=ap.parse_args()
 from preflight import lock
 if (ROOT/'study_lock.json').exists():check_lock()
 else:lock()
 with (ROOT/'supervisor.lock').open('a') as owner:
  try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('V8 supervisor already running')
  old=read(ROOT/'supervisor.json')
  atomic(ROOT/'supervisor.json',dict(identity=proc_identity(os.getpid()),started=time.time()))
  shared=DATA/'proxy_composition_18h/scheduler.lock';shared.parent.mkdir(parents=True,exist_ok=True)
  with shared.open('a') as admission:
   # Wait for a foreign owner; never terminate it or start a second GPU controller.
   while True:
    try:fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:
     atomic(ROOT/'waiting_for_owner.json',dict(at=time.time(),state='QUEUED_SHARED_GPU_LOCK'));time.sleep(15)
   if not (ROOT/'window.json').exists():
    now=time.time();atomic(ROOT/'window.json',dict(start=now,deadline=now+a.deadline_hours*3600,hours=a.deadline_hours))
   env={**os.environ,'V8_ADMISSION_FD':str(admission.fileno())}
   for restart in range(4):
    if (ROOT/'stop.request').exists():return 75
    cmd=[sys.executable,str(HERE/'scheduler.py')]+(['--smoke-only'] if a.smoke_only else [])+(['--stage2-only'] if a.stage2_only else [])
    p=subprocess.Popen(cmd,env=env,pass_fds=(admission.fileno(),),start_new_session=True)
    atomic(ROOT/'controller.json',dict(identity=await_identity(p.pid,str(HERE/'scheduler.py')),restart=restart))
    def stop(*_):
     (ROOT/'stop.request').touch();p.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while p.poll() is None:
     atomic(ROOT/'supervisor_heartbeat.json',dict(at=time.time(),controller=proc_identity(p.pid),restart=restart));time.sleep(10)
    if p.returncode in [0,2,75]:return p.returncode
    atomic(ROOT/'supervisor_restart.json',dict(at=time.time(),exit=p.returncode,restart=restart))
    if time.time()>effective_deadline()-3600:break
    time.sleep(min(60,5*2**restart))
   return 2
if __name__=='__main__':sys.exit(main())
