import sys,tempfile,unittest,contextlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import common as C
import stage2_cli as CLI
from stage1_methods import early_confirm as E

class EarlyTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
  self.stack.enter_context(patch.object(C,'ROOT',self.root));self.stack.enter_context(patch.object(CLI,'ROOT',self.root))
  self.stack.enter_context(patch.object(CLI,'DATA',self.root/'data'))
  prompt=self.root/'data/commercial_pilot10/blinded/example/prompt.txt';prompt.parent.mkdir(parents=True);prompt.write_text('Canonical blind classification')
  self.rows=[dict(image_id=f'{t}_{i}',transition=t,source=1,target=2) for t in C.TIDS for i in range(10)]
  C.atomic(self.root/'manifests/confirm.json',dict(images=self.rows))
  self.cs=[C.make('qwen',{C.V:[l]},split='confirm',label=method) for l,method in [(0,'selected'),(1,'uniform')]]
  C.atomic(self.root/'method_lock.json',dict(stage2_primary_family='qwen',stage2_methods=['selected','uniform'],families={'qwen':dict(primary=self.cs[0]['layers'],uniform=self.cs[1]['layers'])}))
 def receipts(self,count):
  for c in self.cs:
   for r in E.cases(count):
    folder=C.image_receipt(c,r['image_id']).parent;folder.mkdir(parents=True,exist_ok=True)
    clean=self.root/(r['image_id']+'.clean.png');clean.write_bytes(r['image_id'].encode())
    adv=folder/'adv.png';adv.write_bytes((c['id']+r['image_id']).encode())
    C.atomic(folder/'state.json',dict(status='COMPLETE',science=C.digest(C.scientific(c)),clean=str(clean),adv=str(adv),
      png_hashes={str(p):C.sha(p) for p in [clean,adv]},linf_png=.01,target_access=False,**r))
 def test_balanced_nested10_30_50(self):
  ordered=E.ordered_cases(self.rows)
  for n,k in [(10,2),(30,6),(50,10)]:
   self.assertEqual({t:sum(r['transition']==t for r in ordered[:n]) for t in C.TIDS},{t:k for t in C.TIDS})
  self.assertEqual({r['image_id'] for r in ordered},{r['image_id'] for r in self.rows})
 def test_missing_or_changed_png_blocks_early_eval(self):
  self.assertFalse(E.frozen(self.cs[0],30));self.receipts(30);self.assertTrue(E.frozen(self.cs[0],30))
  p=C.image_receipt(self.cs[0],E.cases(30)[0]['image_id']);v=C.read(p);Path(v['adv']).write_bytes(b'changed')
  self.assertFalse(E.frozen(self.cs[0],30))
 def test_no_method_lock_no_early_target(self):
  self.receipts(30);(self.root/'method_lock.json').unlink();self.assertFalse(E.frozen(self.cs[0],30))
 def test30_packet_extends50_with_same_request_ids(self):
  self.receipts(30);packs30=CLI.prepare_packs(count=30)
  with self.assertRaises(RuntimeError):CLI.prepare_packs(count=50)
  self.receipts(50);packs50=CLI.prepare_packs(count=50)
  for a,b in zip(packs30,packs50):
   small=C.read(a/'request_order.json');big=C.read(b/'request_order.json')
   self.assertEqual(small,big[:60]);self.assertEqual(len(set(big)),100)
  self.assertEqual(C.read(self.root/'stage2/pack_lock30.json')['prospective_unique_requests'],90)
  self.assertEqual(C.read(self.root/'stage2/pack_lock.json')['prospective_unique_requests'],150)
 def test30_scores_not_counted_as_complete50(self):
  self.receipts(50);small=CLI.prepare_packs(count=30);CLI.prepare_packs(count=50)
  for p in small:
   for r in C.read(p/'request_order.json'):
    C.atomic(self.root/'stage2/responses'/r/'result.json',dict(parsed_label=2,parser_status='ok'))
  self.assertTrue(CLI.summarize(count=30));self.assertFalse(CLI.summarize(count=50))
  self.assertIsNone(C.read(self.root/'stage2/summary.json')['methods'][0]['ASR'])
  self.assertEqual(C.read(self.root/'stage2/summary30.json')['methods'][0]['N'],30)
 def test_early_eval_does_not_mark_final_eval_complete(self):
  c=self.cs[0];C.atomic(E.path(c),dict(complete=True,science=C.digest(C.scientific(c)),N=30,images=E.cases(30)))
  self.assertTrue(E.done(c));self.assertFalse(C.eval_done(c))
 def test_attack_and_early_eval_can_be_pending_together(self):
  self.receipts(30);jobs=E.pending(self.cs[0]);self.assertEqual([m for c,m in jobs],['evaluate30','attack'])
 def test_all50_waits_for_early_eval_then_reuses_image_cache(self):
  self.receipts(50);c=self.cs[0];self.assertEqual([m for _,m in E.pending(c)],['evaluate30'])
  C.atomic(E.path(c),dict(complete=True,science=C.digest(C.scientific(c)),N=30,images=E.cases(30)))
  self.assertEqual([m for _,m in E.pending(c)],['evaluate'])
