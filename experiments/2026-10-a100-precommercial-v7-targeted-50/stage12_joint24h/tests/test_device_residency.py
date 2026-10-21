import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import device_residency as D
import resources as R
import torch

class Spy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.ones(3,3,dtype=torch.bfloat16));self.calls=[]
 def to(self,device):
  self.calls.append(str(device))
  if str(device)!='cuda':super().to(device)
  return self
class ResidencyTests(unittest.TestCase):
 def proxy(self,family):
  full=torch.nn.Module();inner=torch.nn.Module();vision=Spy();language=Spy();full.add_module('model',inner);inner.add_module('language',language)
  if family=='qwen':inner.add_module('visual',vision)
  elif family in ['internvl','gemma4']:inner.add_module('vision_tower',vision)
  elif family in ['clip','siglip2']:full.add_module('vision_model',vision)
  else:full=vision
  return SimpleNamespace(model=full,spec=dict(family=family),offload_seconds=0),vision,language
 def test_move_only_vision_for_every_family(self):
  for fam in ['qwen','internvl','gemma4','clip','siglip2','dinov2']:
   m,v,l=self.proxy(fam);D.install(m);m.to('cuda')
   self.assertEqual(v.calls[-1],'cuda');self.assertNotIn('cuda',l.calls)
   self.assertTrue(all(p.dtype==torch.bfloat16 for p in m.model.parameters()))
 def test_parameters_and_gradient_unchanged(self):
  m,v,_=self.proxy('qwen');x=torch.ones(1,3,requires_grad=True)
  f=lambda:(x.to(torch.bfloat16)@v.weight).float().sum()
  a=torch.autograd.grad(f(),x)[0];ids=[id(p) for p in m.model.parameters()]
  D.install(m);m.to('cpu');b=torch.autograd.grad(f(),x)[0]
  self.assertEqual(ids,[id(p) for p in m.model.parameters()]);self.assertTrue(torch.equal(a,b))
 def test_resident_budget_and_real_free_guard(self):
  self.assertTrue(D.may_reside(3,8,20));self.assertFalse(D.may_reside(6,8,20));self.assertFalse(D.may_reside(3,8,10))
  self.assertTrue(D.may_reside(.57,2.56,20,activation_margin_gib=1))
 def test_eight_worker_limit_and_headroom(self):
  active=[dict(reservation=4,process_identity=dict(pid=i)) for i in range(7)]
  t=dict(total=80,used=28,free=52,processes=[dict(pid=i,gib=4) for i in range(7)],host_available=40)
  self.assertTrue(R.admit(t,active,4,8));self.assertFalse(R.admit(t,active,4,7))
  self.assertFalse(R.admit(t,active+[dict(reservation=4,process_identity=dict(pid=8))],4,8))
  self.assertFalse(R.admit({**t,'free':10},active,4,8))

class CalibratedAdmissionTests(unittest.TestCase):
 def test_loading_margin_respects_aggregate_and_actual_free(self):
  t=dict(total=80,used=50,free=30,processes=[dict(pid=i,gib=7) for i in range(7)],host_available=50)
  active=[dict(reservation=7,process_identity=dict(pid=i)) for i in range(7)]
  self.assertTrue(R.admit(t,active,7,8,startup_extra=5))
  self.assertFalse(R.admit({**t,'free':19},active,7,8,startup_extra=5))
  heavy=[dict(reservation=9,process_identity=dict(pid=i)) for i in range(7)]
  self.assertFalse(R.admit(t,heavy,7,8,startup_extra=5))
 def test_memory_profile_requires_resident_evidence_and_matching_models(self):
  import tempfile
  from unittest.mock import patch
  import common as C
  with tempfile.TemporaryDirectory() as d,patch.object(R,'ROOT',Path(d)):
   root=Path(d);C.atomic(root/'calibration_resident_v2/x.json',dict(status='PASS',measurements=[dict(reserved_gib=4)]))
   C.atomic(root/'residency_plans/x.json',dict(resident=True))
   C.atomic(root/'jobs/x_calibrate_resident.json',dict(condition=dict(parameter_dtype='bfloat16',CLS=0,layers={'a':[0,1,2,3]})))
   self.assertAlmostEqual(R.resident_profile(dict(layers={'a':[1,2]}))['steady_GiB'],6.7)
   self.assertIsNone(R.resident_profile(dict(layers={'a':[0,1,2,3,4]})))
   self.assertIsNone(R.resident_profile(dict(layers={'b':[0]})))
   C.atomic(root/'residency_plans/x.json',dict(resident=False))
   self.assertIsNone(R.resident_profile(dict(layers={'a':[0]})))
