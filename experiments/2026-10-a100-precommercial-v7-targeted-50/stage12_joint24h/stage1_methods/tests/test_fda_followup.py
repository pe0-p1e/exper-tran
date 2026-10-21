import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_methods import fda_confirmation as F,fda_commercial as M

class FDAFollowupTests(unittest.TestCase):
 def test_fda_freeze_is_never_reselected_from_confirmation(self):
  with tempfile.TemporaryDirectory() as d,patch.object(F,'OUT',Path(d)):
   value=dict(version='frozen',families={});F.C.atomic(Path(d)/'method_lock.json',value)
   self.assertEqual(F.prepare(),value)
 def test_cache_identity_requires_exact_model_image_prompt(self):
  req=dict(request_id='r1',image_sha256='image',prompt_sha256='prompt')
  result=dict(request_id='r1',identity=dict(model='gpt-6.1-sol',image_sha256='image',prompt_sha256='prompt'),parsed_label=None)
  self.assertTrue(M.cache_matches(result,req))
  for key in ['model','image_sha256','prompt_sha256']:
   self.assertFalse(M.cache_matches({**result,'identity':{**result['identity'],key:'different'}},req))
 def test_commercial_snapshot_frozen_before_requests(self):
  with tempfile.TemporaryDirectory() as d,patch.object(M,'OUT',Path(d)):
   value=dict(methods={'uniform':{},'selected':{},'FDA':{}},no_target_retuning=True)
   M.C.atomic(Path(d)/'method_lock.json',value);self.assertEqual(M.prepare(),value)
 def test_fda_membership_rejects_unfrozen_condition(self):
  with patch.object(F.C,'read',return_value=None):self.assertFalse(F.contains(dict(id='unknown')))
