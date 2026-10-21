import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as C, ila_selection as ILA, combo_selection as M, selection as S

class IntegratedSelectionTests(unittest.TestCase):
 def test_ILA_projection_has_original_direction_at_clean(self):
  import torch
  clean=torch.tensor([2.,3.]);baseline=torch.tensor([4.,2.]);x=clean.clone().requires_grad_(True)
  grad=torch.autograd.grad(ILA.projection_loss(x,clean,baseline),x)[0]
  self.assertTrue(torch.equal(-grad,baseline-clean))
  self.assertEqual(ILA.displacement(baseline.numpy(),clean.numpy(),baseline.numpy()),1)
  self.assertIsNone(ILA.displacement(clean.numpy(),clean.numpy(),clean.numpy()))
 def test_latest_local_peak_not_global_max_or_deepest(self):
  matrix=np.ones((6,6));matrix[1,1]=9;matrix[3,3]=2;matrix[5,5]=10
  chosen,peaks=ILA.latest_peak(matrix)
  self.assertEqual(chosen,3);self.assertEqual(peaks,[1,3])
  self.assertIsNone(ILA.latest_peak(np.ones((6,6)))[0])
 def test_invalid_baseline_displacement_is_explicit(self):
  matrix=np.full((4,4),np.nan);self.assertIsNone(ILA.latest_peak(matrix)[0])
 def test_raw_spatial_feature_excludes_masked_tokens(self):
  import torch
  t=torch.arange(12).reshape(1,3,4);f={0:(torch.zeros(1,4),t,torch.tensor([[True,False,True]]))}
  self.assertEqual(ILA.raw(f,0).tolist(),[0,1,2,3,8,9,10,11])
 def test_ILA_resume_checks_png_hash(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);png=p/'adv.png';png.write_bytes(b'original');state=p/'state.json'
   C.atomic(state,dict(complete=True,diagnostic=C.digest(ILA.DIAGNOSTIC),png_hashes={str(png):C.sha(png)}))
   self.assertTrue(ILA.valid(state));png.write_bytes(b'corrupt');self.assertFalse(ILA.valid(state))
 def test_reuse_retains_existing_model_order_and_condition(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);old=C.make('qwen',{C.I:[17],C.V:[11]},label='old');C.atomic(root/'conditions'/f"{old['id']}.json",old)
   with patch.object(M,'ROOT',root):new=M.reuse_make('qwen',{C.V:[11],C.I:[17]},'new')
   self.assertEqual(new['id'],old['id']);self.assertEqual(new['model_order'],old['model_order'])
 def test_pareto_beam_keeps_strength_and_cost_extremes(self):
  cs=[dict(id=i) for i in ['strong','cheap','dominated']]
  evidence={'strong':dict(minimum=1,hits=9,backwards=1000,seconds=100,peak=8),
            'cheap':dict(minimum=1,hits=8,backwards=500,seconds=50,peak=4),
            'dominated':dict(minimum=0,hits=7,backwards=1100,seconds=150,peak=9)}
  with patch.object(M,'eval_done',return_value=True),patch.object(S,'score',side_effect=lambda c:evidence[c['id']]):
   chosen=M.pareto_beam(cs,2)
  self.assertEqual({c['id'] for c in chosen},{'strong','cheap'})
 def test_geometry_prior_penalizes_conflict_with_same_novelty(self):
  a,b=(C.V,11),(C.I,17)
  g=dict(units=[dict(model=a[0],layer=a[1],quality=1,seconds=1),dict(model=b[0],layer=b[1],quality=1,seconds=1)],
         pairs=[dict(left=list(a),right=list(b),CKA=.2,gradient_abs_cos=.3,negative_dot_fraction=0)])
  safe=M.prior(g,b,[a]);g['pairs'][0]['negative_dot_fraction']=1
  self.assertGreater(safe,M.prior(g,b,[a]));self.assertEqual(M.prior(g,b,[a]),0)
 def test_pool_family_exclusion_and_no_peak_fallback(self):
  g={'units':[dict(model=n,layer=l,quality=1+l/100,seconds=1) for n in {C.V,C.Q,C.I,C.G} for l in [0,1]]}
  with patch.object(ILA,'selected',return_value=dict(latest_peak=None)):
   for fam in C.VALIDATORS:
    pool=M.eligible_pool(fam,g);self.assertTrue(C.legal({n:[l] for n,l in pool},C.VALIDATORS[fam]))
 def test_small_method_search_grows_real_conditions_and_resumes(self):
  units=[(C.V,11),(C.I,17),(C.G,15)]
  completed=set();g=dict(units=[dict(model=n,layer=l,quality=1,seconds=1) for n,l in units],
   pairs=[dict(left=list(a),right=list(b),CKA=.4,gradient_abs_cos=.3,negative_dot_fraction=0) for ix,a in enumerate(units) for b in units[ix+1:]])
  def score(c):return dict(hits=5+len(S.atoms(c)),N=10,minimum=0,backwards=500*len(c['layers']),seconds=10*len(c['layers']),peak=len(c['layers']))
  with tempfile.TemporaryDirectory() as td,patch.object(M,'ROOT',Path(td)),patch.object(M,'eligible_pool',return_value=units),patch.object(ILA,'selected',side_effect=lambda n:dict(latest_peak=next(l for m,l in units if m==n))),patch.object(M,'eval_done',side_effect=lambda c:c['id'] in completed),patch.object(S,'eval_done',side_effect=lambda c:c['id'] in completed),patch.object(S,'score',side_effect=score):
   for _ in range(6):
    pending,allc,trace=M.method_screen('qwen',g)
    for c in pending:self.assertTrue(C.legal(c['layers'],c['validator']));completed.add(c['id'])
    if not pending:break
   self.assertFalse(pending);self.assertTrue(any(len(S.atoms(c))>1 for c in allc));self.assertLessEqual(len(allc),13)

if __name__=='__main__':unittest.main()

class ConcurrentSelectionTests(unittest.TestCase):
 def test_large_target_priority_defers_search_and_ignores_blocked_cells(self):
  import scheduler as S
  q=[(dict(id='search'),'attack'),(dict(id='qwen'),'evaluate_large_qwen'),(dict(id='intern'),'evaluate_large_internvl')]
  jobs,priority=S.prioritize_large(q)
  self.assertTrue(priority);self.assertEqual([c['id'] for c,m in jobs],['intern','qwen'])
  jobs,priority=S.prioritize_large(q,['qwen_evaluate_large_qwen','intern_evaluate_large_internvl'])
  self.assertFalse(priority);self.assertEqual(jobs,q)
 def test_shared_queue_fairness_and_no_duplicates(self):
  import scheduler as S
  primary=[(dict(id='new1'),'ila'),(dict(id='new2'),'ila')]
  background=[(dict(id='old1'),'attack'),(dict(id='old2'),'attack')]
  jobs=[dict(condition='new0')]
  q=S.fair_pending(primary,background,jobs,{'old1','old2'})
  self.assertEqual([c['id'] for c,_ in q],['old1','new1','old2','new2'])
  q=S.fair_pending(primary,primary,[],set())
  self.assertEqual(len(q),2)
 def test_six_workers_still_require_real_memory_headroom(self):
  import resources as R
  t=dict(total=80,used=0,free=80,processes=[],host_available=100)
  active=[dict(reservation=8,process_identity=dict(pid=i)) for i in range(5)]
  self.assertTrue(R.admit(t,active,8,6));self.assertFalse(R.admit(t,active,8,5))
  self.assertFalse(R.admit({**t,'free':10},active,8,6))
  self.assertFalse(R.admit(t,active+[dict(reservation=8,process_identity=dict(pid=6))],8,6))
 def test_shortlist_tests_layer_and_model_expansion(self):
  resident,new=(C.V,11),(C.I,17);current=[(C.V,5)]
  with patch.object(M,'prior',side_effect=lambda g,u,s:100 if u[0]==C.V else 1):
   got=M.shortlist({},[resident,new,(C.V,17)],current)
  self.assertIn(new,got);self.assertEqual(len(got),2)
