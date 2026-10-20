import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[4]/'scripts/analyze_v8_pairwise.py'
spec = importlib.util.spec_from_file_location('pairwise_analysis', script)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def outcome(i, hit, clean=1):
    return dict(image_id=str(i), source=1, target=2, transition='T01', clean_label=clean,
                adv_label=2 if hit else 1, target_hit=bool(hit))


class PairwiseTests(unittest.TestCase):
    def test_actual_joint_outcomes_and_interaction(self):
        joint = {str(i): outcome(i, h) for i,h in enumerate([1,1,1,0])}
        left = {str(i): outcome(i, h) for i,h in enumerate([1,0,0,0])}
        right = {str(i): outcome(i, h) for i,h in enumerate([0,1,0,0])}
        s = m.summarize(m.comparison(joint,left,right))
        self.assertEqual(s['I'], .25)
        self.assertEqual(s['joint_only_hits'],1)
        self.assertEqual(s['gains_vs_best'],2)
        self.assertEqual(s['N'],4)

    def test_clean_target_baseline_and_ceiling(self):
        j={'x':outcome('x',1,clean=2)}
        self.assertEqual(m.summarize(m.comparison(j,j,j))['I'],0)
        j={'x':outcome('x',1)}
        self.assertEqual(m.summarize(m.comparison(j,j,j))['I'],-1)

    def test_mismatched_clean_rejected(self):
        with self.assertRaises(ValueError):
            m.comparison({'x':outcome('x',1)}, {'x':outcome('x',0,clean=2)}, {'x':outcome('x',0)})

    def test_partial_duplicates_and_false_scores_rejected(self):
        r=outcome('x',1); manifest={'images':[r]}
        for e in [dict(complete=False,N=1,images=[r]),dict(complete=True,N=1,images=[r,r]),
                  dict(complete=True,N=1,images=[dict(r,target_hit=False)])]:
            with self.assertRaises(ValueError):m.checked_rows(e,manifest)
        self.assertEqual(m.checked_rows(dict(complete=True,N=1,images=[r]),manifest), {'x':r})

if __name__ == '__main__':unittest.main()
