import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import common as C
from paper_closeout import plan as P,commercial as M,measure as V

class CloseoutTests(unittest.TestCase):
 def test_zero_size_smoke_direction_is_na(self):
  from paper_closeout.analysis import rate
  self.assertIsNone(rate(0,0,0));self.assertIsNone(rate(2,30,3));self.assertEqual(rate(2,30,30),2/30)
 def test_only_three_non_astra_low_models(self):
  self.assertEqual(P.MODELS,['gpt-6.1-sol','gpt-6-sol','gpt-5.6-sol']);self.assertTrue(all('astra' not in x for x in P.MODELS))
 def test_final_cannot_freeze_before_gpt50_selection(self):
  with tempfile.TemporaryDirectory() as d,patch.object(P,'OUT',Path(d)),patch.object(P.C,'read',return_value=None):
   with patch.object(P.C,'read',side_effect=lambda p,*a: {'complete':False} if 'results.json' in str(p) else None):self.assertIsNone(P.freeze_final({}))
 def test_exact_family_ban_remains_hard(self):
  with self.assertRaises(ValueError):C.make('qwen',{C.Q:[0]},split='final150')
 def test_ratio_config_changes_identity(self):
  c=C.make('qwen',{C.V:[5,11,17,23]})
  with tempfile.TemporaryDirectory() as d,patch.object(C,'ROOT',Path(d)):
   a=P.modify(c,pull=1.,push=0.);b=P.modify(c,pull=0.,push=1.)
   self.assertNotEqual(a['id'],b['id']);self.assertEqual(a['layers'],b['layers']);self.assertEqual(a['seed'],b['seed'])
 def test_existing_matching_condition_is_reused(self):
  with tempfile.TemporaryDirectory() as d,patch.object(C,'ROOT',Path(d)):
   c=C.make('qwen',{C.V:[11,23]},split='explore');C.atomic(Path(d)/'conditions'/f"{c['id']}.json",c)
   self.assertEqual(P.modify(c,pull=.5,push=.5)['id'],c['id'])
 def test_large_completion_uses_cohort_N(self):
  from stage1_methods import selected_confirmation as S
  c=C.make('qwen',{C.V:[11]},split='final150');data=dict(complete=True,N=150,model=C.LARGE_EVAL_MODES['evaluate_large_qwen'],science=C.digest(C.scientific(c)))
  with patch.object(S.C,'read',return_value=data),patch.object(S.C,'cohort',return_value={'images':[{}]*150}):self.assertTrue(S.large_done(c,'evaluate_large_qwen'))
 def test_class_statistics_are_finite(self):
  import numpy as np
  d=V.stats(np.eye(5));self.assertGreater(d['dispersion'],0);self.assertGreater(d['effective_rank'],0);self.assertAlmostEqual(V.stats(np.ones((5,4)))['covariance_trace'],0)
 def test_normalization_handles_zero(self):
  import numpy as np
  self.assertTrue(np.isfinite(V.unit(np.zeros((3,4)))).all())
 def test_empty_commercial_summary_is_partial_not_zero_asr(self):
  with tempfile.TemporaryDirectory() as d,patch.object(M,'OUT',Path(d)):
   x=M.summarize('commercial150');self.assertFalse(x['complete']);self.assertEqual(x['methods'],[])
 def test_weighted_objective_linear_and_differentiable(self):
  import torch
  import torch.nn.functional as F
  from pp4_objective import objective
  torch.manual_seed(42);bank=F.normalize(torch.randn(10,48,4),dim=-1)
  class Proxy:
   def __init__(self):self.bank={0:(bank,None)}
   def features(self,x):return {0:(F.normalize(x,dim=-1),None,None)}
  x=torch.randn(1,4,requires_grad=True);values=[]
  for p in [1.,0.,.5]:
   value,_=objective(Proxy(),x,1,2,dict(loss='pull_push',layers=[0],pull_weight=p,push_weight=1-p,patch_weight=0,projector_weight=0));values.append(value)
  self.assertTrue(torch.allclose(values[2],(values[0]+values[1])/2));self.assertTrue(torch.isfinite(torch.autograd.grad(values[2].sum(),x)[0]).all())

if __name__=='__main__':unittest.main()
