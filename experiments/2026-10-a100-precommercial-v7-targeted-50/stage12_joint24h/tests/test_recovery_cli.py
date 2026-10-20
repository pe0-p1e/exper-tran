import sys,unittest,tempfile,json,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as C, scheduler as S, stage2_cli as CLI
class RecoveryTests(unittest.TestCase):
 def test_simulated_oom_reduces_concurrency_and_retries(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);log=root/'worker.log';log.write_text('CUDA out of memory')
   j=dict(id='fake',mode='attack',condition='fake',reservation=20,process_identity=dict(pid=99999999,start='0',command='fake'),attempt=0,log=str(log))
   s=dict(active=[j],completed=[],failures=[],attempts={},blocked=[],retry_after={},worker_limit=4,microbatch=2)
   with patch.object(S,'ROOT',root),patch.object(S,'complete',return_value=False):S.reconcile(s)
   self.assertEqual(s['worker_limit'],3);self.assertEqual(s['microbatch'],1);self.assertEqual(s['attempts']['fake'],1);self.assertEqual(len(s['active']),0);self.assertFalse(s['blocked'])
 def test_bug_is_blocked_not_retried(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);log=root/'worker.log';log.write_text('Traceback: shape mismatch')
   j=dict(id='bad',mode='attack',condition='bad',reservation=20,process_identity=dict(pid=99999999,start='0',command='fake'),attempt=0,log=str(log))
   s=dict(active=[j],completed=[],failures=[],attempts={},blocked=[],retry_after={},worker_limit=2)
   with patch.object(S,'ROOT',root),patch.object(S,'complete',return_value=False):S.reconcile(s)
   self.assertEqual(s['blocked'],['bad']);self.assertFalse(s['retry_after'])
 def test_completed_orphan_not_repeated(self):
  s=dict(active=[dict(id='done',process_identity=dict(pid=99999999,start='0',command='fake'))],completed=[])
  with patch.object(S,'complete',return_value=True):S.reconcile(s)
  self.assertEqual(s['completed'],['done']);self.assertEqual(s['active'],[])
 def test_fake_cli_50_and_idempotency(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);pack=root/'pack';cases=[];req=[];order=[]
   for i in range(50):
    ids={}
    for kind in ['clean','adversarial']:
     rid=f'r_{i}_{kind}';p=pack/'blinded'/rid;p.mkdir(parents=True);(p/'image.png').write_bytes(f'image{i}{kind}'.encode());(p/'prompt.txt').write_text('Return one digit 0–9.')
     ids[kind]=rid;order.append(rid);req.append(dict(request_id=rid,image_sha256=C.sha(p/'image.png'),prompt_sha256=C.sha(p/'prompt.txt')))
    cases.append(dict(image_id=str(i),transition='T01',source_label=1,target_label=3,requests=ids))
   C.atomic(pack/'private/manifest.json',dict(cases=cases,requests=req));C.atomic(pack/'request_order.json',order)
   calls=[]
   def fake_run(cmd,**kw):
    if cmd[:3]==['codex','login','status']:return subprocess.CompletedProcess(cmd,0,stdout='Logged in using ChatGPT',stderr='')
    calls.append(cmd);answer=Path(cmd[cmd.index('--output-last-message')+1]);answer.write_text('2')
    events=[dict(type='thread.started',thread_id='fake_'+str(len(calls))),dict(type='item.completed',item=dict(type='agent_message',text='2')),dict(type='turn.completed')]
    return subprocess.CompletedProcess(cmd,0,stdout='\n'.join(json.dumps(e) for e in events),stderr='')
   # Uses the existing known-good CLI parsing/isolation implementation; no real CLI invocation.
   import run_codex_cli_pilot as old
   add=old.argparse.ArgumentParser.add_argument;argv=sys.argv[:]
   try:
    with patch.object(CLI,'ROOT',root),patch('subprocess.run',side_effect=fake_run),patch('subprocess.check_output',return_value='codex-test'):
     CLI.run_legacy(pack,20);self.assertEqual(len(calls),20)
     old.argparse.ArgumentParser.add_argument=add
     CLI.run_legacy(pack,20);self.assertEqual(len(calls),20)
    d=C.read(root/'stage2/responses/pilot_summary.json');self.assertEqual(d['completed_pairs'],10);self.assertEqual(d['denominator'],50);self.assertIsNone(d['ASR'])
    for cmd in calls:self.assertIn('--ephemeral',cmd);self.assertIn('--image',cmd);self.assertIn('model_reasoning_effort="low"',cmd)
   finally:old.argparse.ArgumentParser.add_argument=add;sys.argv=argv
if __name__=='__main__':unittest.main()

class IdentityRaceTests(unittest.TestCase):
 def test_stage2_identity_matches_actual_cli_child(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'logs').mkdir();pack=root/'pack'
   fake=SimpleNamespace(pid=123,poll=lambda:0,returncode=0)
   identity=dict(pid=123,start='1',command='python stage2_cli.py',state='R')
   with patch.object(S,'ROOT',root),patch.object(S,'STOP',False),patch.object(S,'state_write'),patch.object(CLI,'prepare_packs',return_value=[pack]),patch.object(CLI,'summarize',return_value=True),patch.object(S.subprocess,'Popen',return_value=fake),patch.object(S,'await_identity',return_value=identity) as check:
    self.assertTrue(S.run_stage2({},C.time.time()+10000))
    self.assertEqual(check.call_count,2)
    for call in check.call_args_list:self.assertEqual(call.args,(123,str(S.HERE/'stage2_cli.py')))
 def test_waits_for_child_exec_identity(self):
  identities=[dict(pid=1,state='R',command='',start='1'),dict(pid=1,state='R',command='python worker.py --job x',start='1')]
  with patch.object(C,'proc_identity',side_effect=identities):
   self.assertIn('worker.py',C.await_identity(1,'worker.py')['command'])
