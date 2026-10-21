import sys,tempfile,unittest,contextlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_methods import balanced_layers as B

class ThreeProxyTests(unittest.TestCase):
 def test_completed_promotion_survives_json_roundtrip(self):
  with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
   root=Path(d)
   for obj,attr,value in [(B.C,'ROOT',root),(B.J,'ROOT',root),(B,'OUT',root/'three')]:stack.enter_context(patch.object(obj,attr,value))
   for split in ['screen','explore']:B.C.atomic(root/'manifests'/(split+'.json'),dict(images=[]))
   def choice(fam,g):
    return dict(selected={f'{method}{k}':{n:list(range(k)) for n in B.C.POOLS[fam][:3]} for method in ['independent_top','geometry_selected'] for k in [2,3]})
   stack.enter_context(patch.object(B,'select',side_effect=choice))
   stack.enter_context(patch.object(B.C,'attack_done',return_value=True));stack.enter_context(patch.object(B.C,'eval_done',return_value=True))
   stack.enter_context(patch.object(B.S,'rank',return_value=(1,)))
   stack.enter_context(patch.object(B.S,'score',return_value=dict(hits=8,N=10)))
   stack.enter_context(patch.object(B,'pair_outcomes',return_value=None))
   B.plan({});before=B.C.sha(root/'three/qwen_promotion.json')
   B.plan({});self.assertEqual(before,B.C.sha(root/'three/qwen_promotion.json'))
   self.assertEqual(set(B.C.read(root/'three/qwen_promotion.json')['winners']),{'2','3'})
 def test_source_selection_keeps_each_model_and_can_pick_weak_singleton(self):
  order=['a','b','c'];pool=[(n,l) for n in order for l in range(4)];seeds=[(n,0) for n in order]
  with patch.object(B.J,'prior',side_effect=lambda g,u,cur:100 if u[1]==3 else 10-u[1]):
   result=B.source_conditional(pool,seeds,{},order)
  for k in [2,3]:
   self.assertEqual(list(result[k]),order)
   self.assertTrue(all(len(ls)==k for ls in result[k].values()))
   self.assertTrue(all(3 in ls for ls in result[k].values()))
 def test_insufficient_layers_fails_closed(self):
  with self.assertRaises(ValueError):B.source_conditional([('a',0)],[('a',0)],{},['a'])
 def test_conditions_change_only_layers_keep_three_models(self):
  with tempfile.TemporaryDirectory() as d,patch.object(B.C,'ROOT',Path(d)),patch.object(B.J,'ROOT',Path(d)):
   for fam in B.C.VALIDATORS:
    ns=B.C.POOLS[fam][:3]
    choice=dict(selected={f'{method}{k}':{n:list(range(k)) for n in ns} for method in ['independent_top','geometry_selected'] for k in [2,3]})
    base,cc=B.make_conditions(fam,choice,'screen')
    for label,c in cc.items():
     self.assertEqual(len(c['layers']),3);self.assertTrue(all(len(ls)==int(label[-1]) for ls in c['layers'].values()))
     self.assertEqual({k:v for k,v in B.C.scientific(c).items() if k!='layers'},{k:v for k,v in B.C.scientific(base).items() if k!='layers'})
     B.C.legal(c['layers'],c['validator'])
 def test_wrong_cardinality_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(B.C,'ROOT',Path(d)),patch.object(B.J,'ROOT',Path(d)):
   choice=dict(selected={'independent_top2':{n:[0] for n in B.C.POOLS['qwen'][:3]}})
   with self.assertRaises(RuntimeError):B.make_conditions('qwen',choice,'screen')
 def test_identical_methods_share_condition_identity(self):
  with tempfile.TemporaryDirectory() as d,patch.object(B.C,'ROOT',Path(d)),patch.object(B.J,'ROOT',Path(d)):
   ns=B.C.POOLS['qwen'][:3];ls={n:[1,3] for n in ns}
   _,cc=B.make_conditions('qwen',dict(selected={'independent_top2':ls,'geometry_selected2':ls}),'screen')
   self.assertEqual(cc['independent_top2']['id'],cc['geometry_selected2']['id'])
