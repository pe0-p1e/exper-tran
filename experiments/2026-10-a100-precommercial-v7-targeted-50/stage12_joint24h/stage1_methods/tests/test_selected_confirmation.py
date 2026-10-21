import unittest,tempfile,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_methods import selected_confirmation as F
import resources as R

class SelectedConfirmationTests(unittest.TestCase):
 def row(self,hits,taps,condition):
  return dict(split='explore',strategy='selected',complete=True,hits=hits,N=30,layers={'model':list(range(taps))},condition=condition)
 def test_development_only_highest_hits_then_fewer_taps(self):
  rows=[self.row(28,3,'a'),self.row(27,2,'b'),self.row(28,2,'c')]
  self.assertEqual(F.choose(rows)['condition'],'c')
  with self.assertRaises(RuntimeError):F.choose([{**rows[0],'split':'confirm'}])
 def test_incomplete_is_not_zero_or_selection_evidence(self):
  with self.assertRaises(RuntimeError):F.choose([{**self.row(None,2,'x'),'complete':False}])
 def test_existing_freeze_is_reused_without_reselection(self):
  with tempfile.TemporaryDirectory() as d,patch.object(F,'OUT',Path(d)):
   value=dict(version='unchanged',families={});F.C.atomic(Path(d)/'method_lock.json',value)
   self.assertEqual(F.prepare(),value)
 def test_large_family_exclusion_cannot_be_overridden(self):
  c=F.C.make('internvl',{F.C.Q:[1]},split='confirm')
  lock=dict(families={'internvl':dict(selected=c,uniform=c)},cohort_sha256='hash')
  with patch.object(F.C,'read',return_value=lock),patch.object(F.C,'sha',return_value='hash'),patch.object(F.C,'attack_done',return_value=True):
   with self.assertRaises(ValueError):F.validate(c,'evaluate_large_qwen')
 def test_large_validation_rejects_unfrozen_pngs(self):
  c=F.C.make('qwen',{F.C.I:[1]},split='confirm');lock=dict(families={'qwen':dict(selected=c,uniform=c)},cohort_sha256='hash')
  with patch.object(F.C,'read',return_value=lock),patch.object(F.C,'sha',return_value='hash'),patch.object(F.C,'attack_done',return_value=False):
   with self.assertRaises(RuntimeError):F.validate(c,'evaluate_large_qwen')
 def test_large_result_does_not_overwrite_development_result(self):
  c=dict(id='x');self.assertNotEqual(F.large_path(c,'evaluate_large_qwen'),F.C.eval_path(c))
 def test_checkpoint_memory_uses_bf16_parameter_size_upper_bound(self):
  with patch.object(R,'read',return_value=dict(metadata=dict(total_size=10*2**30,total_parameters=10*2**30))),patch.object(R,'REG',{'large':dict(repo_id='owner/repo',revision='pinned')}):
   self.assertEqual(R.checkpoint_gib('large'),20)

