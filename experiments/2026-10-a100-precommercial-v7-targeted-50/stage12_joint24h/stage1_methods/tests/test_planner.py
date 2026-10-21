import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG.parent))
from stage1_methods import planner as P

class MethodTests(unittest.TestCase):
 def test_family_exclusion_at_condition_boundary(self):
  with self.assertRaises(ValueError):P.condition('qwen',[('Qwen3.5-2B',0)],'bad')
 def test_core_proxy_matrix_fixes_layers(self):
  ns=P.C.POOLS['qwen'][:3];layers={n:[0,2] for n in ns}
  with patch.object(P.J,'reuse_make',side_effect=lambda fam,ls,label,split:P.C.make(fam,ls,label=label,split=split)):
   cs=P.proxy_fixed('qwen',layers)
  self.assertEqual(len(cs),7)
  self.assertEqual([sum(len(c['layers'])==k for c in cs) for k in [1,2,3]],[3,3,1])
  for c in cs:
   for n,ls in c['layers'].items():self.assertEqual(ls,layers[n])
 def test_alignment_uses_pixel_gradients_and_actual_deepest(self):
  data={'a':([0,2],np.array([[[1.,0],[0.,1]]])), 'b':([0,1],np.array([[[-1.,0],[0.,1]]]))}
  ranked=P.alignment_scores(data,{'a':3,'b':2})
  self.assertEqual(ranked[0][0],1.)
  with self.assertRaises(ValueError):P.alignment_scores(data,{'a':4,'b':2})
 def test_alignment_rejects_cancellation(self):
  with self.assertRaises(ValueError):P.alignment_scores({'a':([0],np.array([[[1.,0]]])),'b':([0],np.array([[[-1.,0]]]))},{'a':1,'b':1})
 def test_alignment_rejects_dimension_mismatch(self):
  with self.assertRaises(ValueError):P.alignment_scores({'a':([0],np.ones((1,1,2))),'b':([0],np.ones((1,1,3)))},{'a':1,'b':1})
 def test_spaced_ranks_are_explicit_not_top3(self):
  scores=[(1-i/10,('m',i)) for i in range(10)]
  self.assertEqual(P.spaced_ranks(scores),[('m',0),('m',4),('m',8)])
 def test_neighborhood_reopens_weak_layers_and_proxy_identity(self):
  cur=[('a',1),('b',2)];pool=cur+[('a',0),('c',3)]
  ns=P.neighborhood(cur,pool)
  self.assertIn((('a',0),('b',2)),ns) # replace strong with unranked weak
  self.assertIn((('b',2),),ns) # remove
  self.assertIn((('a',1),('b',2),('c',3)),ns) # add new model
 def test_neighborhood_nonempty_and_bounded(self):
  self.assertNotIn((),P.neighborhood([('a',1)],[('a',1)]))
  self.assertTrue(all(len(x)<=4 for x in P.neighborhood([('a',1),('b',2)],[('a',1),('b',2),('c',3),('d',4)])))
 def test_FDA_uses_all_candidates_not_singleton_topk(self):
  calls=[]
  def reg(cs,method):calls.append((cs,method));return cs
  with patch.object(P,'condition',side_effect=lambda f,us,l:dict(id=str(us),layers={n:[x for nn,x in us if nn==n] for n,_ in us})),patch.object(P,'register',side_effect=reg):
   pending,_,_=P.fda_model('qwen',P.C.V)
  self.assertEqual(len(pending),6)
 def test_owner_patch_allows_primary_confirmation_and_parallel_research(self):
  source=(P.C.HERE/'scheduler.py').read_text();new=P.integration_text(source)
  self.assertIn('paper,paper_ready=research_pending(phase)',new)
  self.assertIn('commercial=stage2_progress(deadline)',new)
  self.assertNotIn('literature_plan(g)',new)
  compile(new,'prepared_owner','exec')
 def test_owner_patch_fails_closed_if_api_changed(self):
  with self.assertRaises(RuntimeError):P.integration_text('unexpected source')
 def test_owner_patch_is_idempotent(self):
  source=(P.C.HERE/'scheduler.py').read_text();new=P.integration_text(source)
  self.assertEqual(P.integration_text(new),new)
 def test_report_never_counts_partial_as_complete(self):
  with tempfile.TemporaryDirectory() as d,patch.object(P,'OUT',Path(d)):
   P.C.atomic(Path(d)/'registry.json',{'x':dict(condition=dict(id='x',family='qwen',layers={P.C.V:[0]}),methods=['test'])})
   with patch.object(P,'completed',return_value=False):r=P.report()
   self.assertEqual(r['complete'],0)
   self.assertIsNone(P.C.read(Path(d)/'results.json')['rows'][0]['score'])
 def test_preparation_does_not_mutate_active_owner(self):
  before=P.C.sha(P.C.HERE/'scheduler.py')
  with tempfile.TemporaryDirectory() as d,patch.object(P,'OUT',Path(d)):
   r=P.prepare();self.assertEqual(r['status'],'PREPARED_NOT_ACTIVATED')
  self.assertEqual(before,P.C.sha(P.C.HERE/'scheduler.py'))
if __name__=='__main__':unittest.main()

class IntegrationTests(unittest.TestCase):
 def test_full_adaptive_queue_resume_and_no_target_feedback(self):
  """Simulated existing worker receipts drive ALL mandatory phases, no inference."""
  import contextlib,itertools
  with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
   root=Path(d);out=root/'stage1_methods';finished=set();phases=[]
   stack.enter_context(patch.object(P.C,'ROOT',root));stack.enter_context(patch.object(P,'OUT',out))
   stack.enter_context(patch.object(P.J,'ROOT',root))
   stack.enter_context(patch.object(P.C,'attack_done',side_effect=lambda c:c['id'] in finished))
   stack.enter_context(patch.object(P.C,'eval_done',side_effect=lambda c:c['id'] in finished))
   stack.enter_context(patch.object(P.J,'eval_done',side_effect=lambda c:c['id'] in finished))
   stack.enter_context(patch.object(P.C,'native',side_effect=lambda n,fracs:[0,3,7,P.C.REG[n]['expected_vision_depth']-1]))
   stack.enter_context(patch.object(P.S,'native',side_effect=P.C.native))
   for rel in ['study_lock.json','manifests/screen.json']:P.C.atomic(root/rel,{'test':True})
   def metric(c):
    u=P.units(c);hits=min(9,len(u)+sum(l==3 for _,l in u)+2)
    return dict(hits=hits,N=10,minimum=hits//5,backwards=len(c['layers'])*500,seconds=len(u)*10,peak=len(c['layers'])*2)
   stack.enter_context(patch.object(P.S,'score',side_effect=metric))
   stack.enter_context(patch.object(P.S,'rank',side_effect=lambda c:(metric(c)['minimum'],metric(c)['hits'],-metric(c)['backwards'],-metric(c)['seconds'],c['id'])))
   def gm(n):
    ls=[0,3,7,P.C.REG[n]['expected_vision_depth']-1]
    a=np.array([[[1.,.1],[.8,.2],[.6,.4],[.5,.5]]]*10)
    return ls,a,[]
   stack.enter_context(patch.object(P,'gradient_matrix',side_effect=gm))
   names=sorted({n for pool in P.C.POOLS.values() for n in pool[:3]})
   us=[(n,l) for n in names for l in gm(n)[0]]
   g=dict(units=[dict(model=n,layer=l,quality=1.,seconds=1.) for n,l in us],
     pairs=[dict(left=list(a),right=list(b),CKA=.5,gradient_abs_cos=.3,negative_dot_fraction=.1) for a,b in itertools.combinations(us,2)])
   # Existing singleton controls are simulated completed evidence, matching real starting conditions.
   for fam in P.C.VALIDATORS:
    for c in P.S.layer_conditions(fam):
     finished.add(c['id']);P.C.atomic(root/'conditions'/(c['id']+'.json'),c)
   for iteration in range(30):
    phase,cc=P.plan_next(g);phases.append(phase)
    if phase=='READY':break
    self.assertTrue(cc)
    # Interrupt/replan before worker completion: exact IDs must remain stable.
    repeated_phase,repeated=P.plan_next(g)
    self.assertEqual([c['id'] for c in cc],[c['id'] for c in repeated])
    for c in cc:
     self.assertEqual(c['validator'],P.C.VALIDATORS[c['family']]);P.C.legal(c['layers'],c['validator'])
     self.assertEqual(c['steps'],50);self.assertEqual(c['parameter_dtype'],'bfloat16')
     self.assertEqual(c['pull'],.5);self.assertEqual(c['push'],.5)
     finished.add(c['id'])
   self.assertEqual(phases[-1],'READY')
   for fam in P.C.VALIDATORS:
    result=P.C.read(out/(fam+'_method_results.json'));self.assertTrue(result['complete'])
    self.assertTrue(P.nominees(fam))
   reg=P.C.read(out/'registry.json');methods={m for e in reg.values() for m in e['methods']}
   self.assertTrue({'FDA_conditional_2','FDA_conditional_4','proxy_fixed_layers','PRAF_and_independent','weak_pair_control','joint_greedy_beam'}<=methods)
   self.assertFalse((root/'method_lock.json').exists())

class ParallelTests(unittest.TestCase):
 def test_research_not_started_during_original_pilot(self):
  from stage1_methods import parallel_owner as A
  with tempfile.TemporaryDirectory() as d,patch.object(A.C,'ROOT',Path(d)):
   with patch.object(A.P,'plan_next',side_effect=AssertionError('must not plan yet')):
    self.assertEqual(A.research_pending('INTEGRATED_METHOD_PILOT'),([],False))
 def test_ambiguous_commercial_driver_exit_does_not_retry(self):
  from stage1_methods import parallel_owner as A
  with tempfile.TemporaryDirectory() as d,patch.object(A.C,'ROOT',Path(d)):
   A.C.atomic(Path(d)/'stage2/parallel_driver.json',dict(identity={'pid':0}))
   with patch.object(A.C,'same_process',return_value=False),patch.object(A.subprocess,'Popen',side_effect=AssertionError('no retry')):
    self.assertEqual(A.stage2_progress(0),'BLOCKED')
 def test_controller_restart_adopts_commercial_child(self):
  from stage1_methods import parallel_owner as A
  with tempfile.TemporaryDirectory() as d,patch.object(A.C,'ROOT',Path(d)):
   A.C.atomic(Path(d)/'stage2/parallel_driver.json',dict(identity={'pid':0}))
   with patch.object(A.C,'same_process',return_value=True),patch.object(A.subprocess,'Popen',side_effect=AssertionError('no duplicate')):
    self.assertEqual(A.stage2_progress(0),'RUNNING')
 def test_research_failure_is_quarantined(self):
  from stage1_methods import parallel_owner as A
  with tempfile.TemporaryDirectory() as d,patch.object(A.C,'ROOT',Path(d)),patch.object(A.P,'OUT',Path(d)/'stage1_methods'):
   with patch.object(P.J,'source_geometry',side_effect=RuntimeError('test resource/cache error')):
    self.assertEqual(A.research_pending('INTEGRATED_EXPLORE30'),([],False))
   self.assertTrue(A.C.read(Path(d)/'stage1_methods/blocked.json')['primary_unaffected'])

class DriverIsolationTests(unittest.TestCase):
 def test_fake_commercial_driver_does_not_overwrite_GPU_owner_state(self):
  from stage1_methods import stage2_driver as D
  with tempfile.TemporaryDirectory() as d,patch.object(D.C,'ROOT',Path(d)),patch.object(D.C,'check_lock',return_value={}):
   root=Path(d);D.C.atomic(root/'scheduler.json',dict(owner='GPU',phase='paper'))
   def fake(s,deadline):D.S.state_write(dict(phase='STAGE2',active=[]));return True
   with patch.object(D.sys,'argv',['driver','--deadline','9999999999']),patch.object(D.S,'run_stage2',side_effect=fake),patch.object(D.S,'state_write'),patch.object(D.signal,'signal'):
    self.assertEqual(D.main(),0)
   self.assertEqual(D.C.read(root/'scheduler.json'),dict(owner='GPU',phase='paper'))
   self.assertEqual(D.C.read(root/'stage2/parallel_driver_result.json')['status'],'COMPLETE')
