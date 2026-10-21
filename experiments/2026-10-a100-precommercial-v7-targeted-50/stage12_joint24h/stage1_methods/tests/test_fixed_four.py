import sys,tempfile,contextlib,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_methods import fixed_four as F

class FixedFourTests(unittest.TestCase):
 def test_exact_native_positions(self):
  for n in [F.C.V,F.C.Q,F.C.I]:
   self.assertEqual(F.C.native(n,F.SCHEMES['shifted_quartiles']),[2,8,14,20])
   self.assertEqual(F.C.native(n,F.SCHEMES['interior_swap']),[5,11,14,23])
  self.assertEqual(F.C.native(F.C.G,F.SCHEMES['shifted_quartiles']),[1,5,9,13])
  self.assertEqual(F.C.native(F.C.G,F.SCHEMES['interior_swap']),[3,7,9,15])
 def test_only_layer_positions_change(self):
  with tempfile.TemporaryDirectory() as d,patch.object(F.C,'ROOT',Path(d)),patch.object(F.J,'ROOT',Path(d)):
   for fam in F.C.VALIDATORS:
    cs=F.controls(fam);base=F.C.scientific(cs['uniform_quartiles'])
    for c in cs.values():
     F.C.legal(c['layers'],F.C.VALIDATORS[fam]);self.assertEqual(len(c['layers']),3)
     self.assertTrue(all(len(ls)==4 for ls in c['layers'].values()))
     self.assertEqual({k:v for k,v in base.items() if k!='layers'},{k:v for k,v in F.C.scientific(c).items() if k!='layers'})
 def test_interior_control_changes_exactly_one_layer_per_proxy(self):
  with tempfile.TemporaryDirectory() as d,patch.object(F.C,'ROOT',Path(d)),patch.object(F.J,'ROOT',Path(d)):
   cs=F.controls('qwen')
   for n,ls in cs['uniform_quartiles']['layers'].items():
    self.assertEqual(len(set(ls)^set(cs['interior_swap']['layers'][n])),2)
 def test_exact_cache_does_not_reuse_model_permutation(self):
  with tempfile.TemporaryDirectory() as d,patch.object(F.C,'ROOT',Path(d)):
   a=F.C.make('qwen',{F.C.V:[0],F.C.I:[0]});F.C.atomic(Path(d)/'conditions'/(a['id']+'.json'),a)
   b=F.exact('qwen',{F.C.I:[0],F.C.V:[0]},'test','screen');self.assertNotEqual(a['id'],b['id'])
 def test_first_plan_has_six_new_position_conditions_no_confirm(self):
  with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as s:
   root=Path(d);s.enter_context(patch.object(F.C,'ROOT',root));s.enter_context(patch.object(F.J,'ROOT',root));s.enter_context(patch.object(F,'OUT',root/'fixed'))
   for split in ['screen','explore']:F.C.atomic(root/'manifests'/(split+'.json'),dict(images=[]))
   def complete(c):return c['label']=='fixed4_uniform'
   s.enter_context(patch.object(F,'completed',side_effect=complete))
   s.enter_context(patch.object(F.S,'score',return_value=dict(hits=8,N=10,directions=[2,2,2,1,1],seconds=1,backwards=1500,peak=10)))
   s.enter_context(patch.object(F,'pair_outcomes',return_value=None))
   pending,done=F.plan();self.assertFalse(done);self.assertEqual(len(pending),6)
   self.assertTrue(all(c['split']=='screen' for c in pending));self.assertFalse((root/'method_lock.json').exists())
   rows=F.C.read(root/'reports/four_layer_positions.json')['rows']
   self.assertTrue(all(r['hits'] is None for r in rows if not r['complete']))
 def test_pair_gained_and_lost_not_only_net(self):
  a=dict(id='a');b=dict(id='b')
  records={'a':[dict(image_id='1',source=1,target=2,transition='T01',clean_label=1,target_hit=True),dict(image_id='2',source=1,target=2,transition='T01',clean_label=1,target_hit=False)],
   'b':[dict(image_id='1',source=1,target=2,transition='T01',clean_label=1,target_hit=False),dict(image_id='2',source=1,target=2,transition='T01',clean_label=1,target_hit=True)]}
  with patch.object(F,'completed',return_value=True),patch.object(F.C,'eval_path',side_effect=lambda c:c['id']),patch.object(F.C,'read',side_effect=lambda p:dict(images=records[p])):
   self.assertEqual(F.pair_outcomes(a,b),dict(gained=1,lost=1,net=0))
