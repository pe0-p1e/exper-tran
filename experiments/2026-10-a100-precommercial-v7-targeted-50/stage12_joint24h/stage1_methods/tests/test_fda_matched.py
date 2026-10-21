import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_methods import fda_matched as F

class MatchedTests(unittest.TestCase):
 def test_rejects_proxy_change(self):
  with self.assertRaises(ValueError):F.matched('qwen',['a','b'],{'a':[0,1]},'test')
 def test_rejects_unequal_layer_count(self):
  with self.assertRaises(ValueError):F.matched('qwen',['a','b'],{'a':[0,1],'b':[0,1,2,3]},'test')
 def test_keeps_order_and_only_changes_layers(self):
  with patch.object(F,'exact',side_effect=lambda fam,layers,label,split:(fam,layers,split)):
   got=F.matched('qwen',['b','a'],{'a':[0,1],'b':[1,2]},'test')
   self.assertEqual(list(got[1]),['b','a']);self.assertEqual(got[2],'explore')
 def test_rejects_three_layer_condition(self):
  with self.assertRaises(ValueError):F.matched('qwen',['a'],{'a':[0,1,2]},'test')
 def test_growth_only_reads_development_conditions(self):
  made=[]
  def condition(fam,units,label):
   made.append((fam,units,label));return dict(id=str(units),layers={'model':[v for _,v in units]})
  with patch.object(F.C,'native',return_value=[0,1,2,3,4,5]),patch.object(F.P,'condition',side_effect=condition),patch.object(F.P,'completed',return_value=True),patch.object(F.P,'best',side_effect=lambda cs:[cs[0]]),patch.object(F.P,'units',side_effect=lambda c:[('model',v) for v in c['layers']['model']]):
   missing,chosen,trace=F.grow('qwen','model')
   self.assertFalse(missing);self.assertEqual(len(chosen['layers']['model']),4)
   self.assertEqual({x['size'] for x in trace},{2,3,4})
