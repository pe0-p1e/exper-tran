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

class SharedRecoveryTests(unittest.TestCase):
 def test_shared_clean_recovered_by_other_method_clears_old_timeout(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);rid='r_ab12';f=root/'responses/model'/rid
   C.atomic(f/'result.json',dict(request_id=rid,identity={'model':'model'}))
   log=root/'old.log';log.write_text(f"/responses/model/{rid}/answer.txt\nTraceback (most recent call last):\nRuntimeError: CLI request timed out; do not count as a completed observation\n")
   self.assertEqual(R.recover_blocked(root,'model',[rid],log)['shared_timeout_already_resolved'],rid)
 def test_new_unknown_error_is_not_cleared_by_old_shared_success(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);rid='r_ab12';C.atomic(root/'responses/model'/rid/'result.json',dict(request_id=rid,identity={'model':'model'}))
   log=root/'old.log';log.write_text(f"/responses/model/{rid}/answer.txt\nTraceback (most recent call last):\nKeyError: changed key\n")
   with self.assertRaises(RuntimeError):R.recover_blocked(root,'model',[rid],log)
 def test_quota_blocks_all_methods_not_just_one_condition(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);f=root/'responses/model/r_ab12';C.atomic(f/'transport_error.json',{'kind':'nonzero_exit'});(f/'stderr.txt').write_text('HTTP 429 rate limit')
   with self.assertRaises(R.QuotaAuthBlock):R.recover_blocked(root,'model',['r_ab12'],root/'none.log')
 def test_thread_id_containing_429_is_not_rate_limit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);f=root/'responses/model/r_ab12';C.atomic(f/'transport_error.json',{'kind':'timeout'});(f/'timeout_stdout.jsonl').write_text('{"thread_id":"a429e"}')
   self.assertIsNone(R.fatal_transport(root,'model',['r_ab12']))

class InterruptedRecoveryTests(unittest.TestCase):
 def test_archived_retry_resumes_without_spending_another_attempt(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);rid='r_ab12';f=root/'responses/model'/rid;C.atomic(f/'transport_error.json',{'kind':'timeout'})
   log=root/'old.log';log.write_text(f"/responses/model/{rid}/answer.txt\nTraceback (most recent call last):\nRuntimeError: CLI request timed out; do not count as a completed observation\n")
   R.recover_timeouts(root,'model',[rid])
   self.assertEqual(R.recover_blocked(root,'model',[rid],log)['archived_retry_ready'],rid)
   self.assertEqual(C.read(f/'retry_history.json')['transport_retries'],1)
if __name__=='__main__':unittest.main()
