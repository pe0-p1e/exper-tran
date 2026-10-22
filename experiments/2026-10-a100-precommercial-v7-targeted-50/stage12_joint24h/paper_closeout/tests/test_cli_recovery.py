import sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import common as C
from paper_closeout import cli_recovery as R
class RecoveryTests(unittest.TestCase):
 def test_timeout_retained_once_and_score_not_retried(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);f=root/'responses/model/r';C.atomic(f/'transport_error.json',{'kind':'timeout'})
   self.assertEqual(R.recover_timeouts(root,'model',['r']),1)
   self.assertEqual(R.recover_timeouts(root,'model',['r']),0)
   self.assertTrue((f/'failed_attempts/attempt_01/transport_error.json').exists())
   C.atomic(f/'result.json',{'hits':1});C.atomic(f/'transport_error.json',{'kind':'timeout'})
   self.assertEqual(R.recover_timeouts(root,'model',['r']),0)
 def test_answer_or_completed_turn_requires_manual_review(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);f=root/'responses/model/r';C.atomic(f/'transport_error.json',{'kind':'timeout'})
   (f/'stdout.jsonl').write_text('{"type":"turn.completed"}\n')
   with self.assertRaises(RuntimeError):R.recover_timeouts(root,'model',['r'])
 def test_timeout_retries_bounded(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);f=root/'responses/model/r'
   for _ in range(3):
    C.atomic(f/'transport_error.json',{'kind':'timeout'});R.recover_timeouts(root,'model',['r'])
   C.atomic(f/'transport_error.json',{'kind':'timeout'})
   with self.assertRaises(RuntimeError):R.recover_timeouts(root,'model',['r'])
 def test_auth_and_unknown_failures_never_retried(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);f=root/'responses/model/r';C.atomic(f/'transport_error.json',{'kind':'timeout'})
   (f/'stderr.txt').write_text('429 quota')
   with self.assertRaises(RuntimeError):R.recover_timeouts(root,'model',['r'])
 def test_timeout_capture_keeps_command_and_partial_trace(self):
  with tempfile.TemporaryDirectory() as tmp:
   cmd=['codex','exec','--output-last-message',str(Path(tmp)/'answer.txt')]
   fake=unittest.mock.Mock(side_effect=subprocess.TimeoutExpired(cmd,600,output=b'partial',stderr=b'network'))
   with patch.object(subprocess,'run',fake):
    R.install_timeout_capture()
    with self.assertRaises(subprocess.TimeoutExpired):subprocess.run(cmd,input='prompt',timeout=180)
   self.assertEqual(fake.call_args.kwargs['timeout'],600)
   self.assertEqual((Path(tmp)/'timeout_stdout.jsonl').read_text(),'partial')
if __name__=='__main__':unittest.main()
