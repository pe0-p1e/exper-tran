from common import *
import argparse,subprocess

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 for arg in ['preflight','prepare-only','smoke','start','resume','status','stop','report','stage2-only']:g.add_argument('--'+arg,action='store_true')
 p.add_argument('--deadline-hours',type=float,default=24);a=p.parse_args()
 ROOT.mkdir(parents=True,exist_ok=True)
 if a.status:print(json.dumps(read(ROOT/'status.json',dict(state='NOT_STARTED')),indent=2));return
 if a.stop:
  (ROOT/'stop.request').touch();c=read(ROOT/'controller.json')
  if c and same_process(c['identity']):os.kill(c['identity']['pid'],15)
  print('Graceful stop requested; checkpoints retained');return
 if a.report:
  from reports import generate
  generate();print(ROOT/'reports/PAPER_HANDOFF.md');return
 if a.preflight or a.prepare_only:
  from preflight import prepare
  print(json.dumps(prepare(),indent=2));return
 if a.stage2_only:
  check_lock()
  if not (ROOT/'method_lock.json').exists():raise RuntimeError('Stage2 forbidden before method freeze')
  # Continue via the same supervisor and shared admission lock.
 old=read(ROOT/'supervisor.json')
 if old and same_process(old['identity']):print('Supervisor already running; use --status');return
 if a.deadline_hours<=6:raise ValueError('Budget must leave at least6h for Stage2/report')
 (ROOT/'stop.request').unlink(missing_ok=True)
 name='v8-stage12'
 cmd=['bash',str(EXP/'run_stage12_joint24h.sh'),'--internal-supervisor','--deadline-hours',str(a.deadline_hours)]
 if a.smoke:cmd+=['--smoke-only']
 if a.stage2_only:cmd+=['--stage2-only']
 # subprocess args remain shell-quoted by shlex; tmux provides independent session lifetime.
 import shlex
 line=shlex.join(cmd)+' >> '+shlex.quote(str(ROOT/'logs/supervisor.log'))+' 2>&1'
 (ROOT/'logs').mkdir(parents=True,exist_ok=True)
 r=subprocess.run(['tmux','new-session','-d','-s',name,line],cwd=REPO)
 if r.returncode:raise RuntimeError('tmux launch failed; inspect existing session, do not start duplicate owners')
 print('Started detached supervisor:',name,'output:',ROOT)
if __name__=='__main__':main()
