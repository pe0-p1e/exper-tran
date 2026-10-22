import importlib.util,sys,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/finalize_v8_statistics.py'
spec=importlib.util.spec_from_file_location('v8stats',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PairedStatisticsTests(unittest.TestCase):
 def test_equal_masks_zero_delta_and_p_one(self):
  a={str(i):dict(hit=i%2,transition=m.C.TIDS[i%5]) for i in range(50)}
  s=m.paired(a,a);self.assertEqual(s['McNemar_exact_p'],1);self.assertEqual(s['difference_pp'],0);self.assertEqual(s['CI95_low_pp'],0)
 def test_same_totals_can_have_different_success_masks(self):
  a={str(i):dict(hit=i%2,transition=m.C.TIDS[i%5]) for i in range(50)}
  b={k:dict(v,hit=1-v['hit']) for k,v in a.items()};s=m.paired(a,b)
  self.assertEqual(s['a_only'],25);self.assertEqual(s['b_only'],25);self.assertEqual(s['difference_pp'],0)
 def test_mismatched_cohorts_not_silently_intersected(self):
  with self.assertRaises(ValueError):m.paired({'a':{}},{'b':{}})
 def test_wilson_edges(self):
  lo,hi=m.wilson(0,150);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)
  lo,hi=m.wilson(150,150);self.assertLess(lo,1);self.assertAlmostEqual(hi,1)
if __name__=='__main__':unittest.main()
