import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('proxy_followup_prepare',HERE/'prepare.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
C=P.C

class FollowupPreparationTests(unittest.TestCase):
 def fixture(self,root):
  C.atomic(root/'method_lock.json',dict(test_fixture=True))
  lock=P.layer_template();lock['complete']=True;lock['source_method_lock_sha256']=C.sha(root/'method_lock.json')
  for fam,d in lock['families'].items():
   for n,v in d['models'].items():
    c=C.make(fam,{n:C.native(n,[.25,.5,.75,1])});C.atomic(root/'conditions'/(c['id']+'.json'),c)
    v['layers']=c['layers'][n];v['layer_evidence_conditions']=[c['id']]
  return lock
 def test_draft_is_not_executable(self):
  with self.assertRaises(ValueError):P.validate_layer_lock(P.layer_template())
 def test_compiler_has_21_legal_fixed_layer_subsets(self):
  with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)),patch.object(C,'eval_done',return_value=True),patch.object(C,'attack_done',return_value=False):
   lock=self.fixture(Path(td));tasks=P.compile_tasks(lock)
   self.assertEqual(len(tasks),21)
   for t in tasks:self.assertTrue(C.legal(t['condition']['layers'],P.LARGE[t['family']]))
   self.assertEqual(sum(t['K']==1 for t in tasks),9);self.assertEqual(sum(t['K']==2 for t in tasks),9);self.assertEqual(sum(t['K']==3 for t in tasks),3)
 def test_same_family_proxy_injection_blocked(self):
  with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)),patch.object(C,'eval_done',return_value=True):
   lock=self.fixture(Path(td));lock['families']['qwen']['models'][C.Q]=dict(layers=[1])
   with self.assertRaises(ValueError):P.validate_layer_lock(lock)
 def test_native_layer_revision_and_provenance_not_guessed(self):
  for mutation in ['layers','revision','source']:
   with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)),patch.object(C,'eval_done',return_value=True):
    lock=self.fixture(Path(td));v=lock['families']['qwen']['models'][C.I]
    if mutation=='layers':v['layers']=[999]
    elif mutation=='revision':v['revision']='wrong'
    else:lock['source_method_lock_sha256']='wrong'
    with self.assertRaises(ValueError):P.validate_layer_lock(lock)
 def test_cache_reuse_preserves_order_and_requires_valid_attacks(self):
  with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)),patch.object(C,'attack_done',return_value=True),patch.object(C,'eval_done',return_value=True):
   c=C.make('qwen',{C.I:[17],C.V:[11]});C.atomic(Path(td)/'conditions'/(c['id']+'.json'),c)
   d=P.reusable('qwen',{C.V:[11],C.I:[17]},'screen')
   self.assertTrue(d['reuse_attack']);self.assertEqual(d['condition']['id'],c['id'])
   with patch.object(C,'attack_done',return_value=False):self.assertFalse(P.reusable('qwen',c['layers'],'screen')['reuse_attack'])
 def test_freeze_rejects_layer_change_and_more_than_two_methods(self):
  with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)),patch.object(P,'OUT',Path(td)/'followup'),patch.object(C,'eval_done',return_value=True):
   root=Path(td);lock=self.fixture(root);path=root/'layer_lock.json';C.atomic(path,lock)
   methods={f:[C.make(f,{C.POOLS[f][0]:lock['families'][f]['models'][C.POOLS[f][0]]['layers']},split='confirm')] for f in P.LARGE}
   frozen=P.freeze_methods(path,methods);self.assertTrue(frozen.exists())
   methods['qwen'][0]=C.make('qwen',{C.V:[0]},split='confirm')
   with self.assertRaises(ValueError):P.freeze_methods(path,methods)
 def test_big_panel_is_not_small_validator_panel(self):
  self.assertEqual(set(P.LARGE.values()),{'Qwen3.5-27B','InternVL3.5-14B-HF','Gemma 4 31B-it'})
  self.assertTrue(set(P.LARGE.values()).isdisjoint(C.VALIDATORS.values()))
 def test_small_model_or_unfrozen_packet_cannot_be_substituted(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'lock.json';C.atomic(p,dict(protocol='wrong'))
   with self.assertRaises(ValueError):P.frozen_large_packet(C.make('qwen',{C.I:[17]},split='confirm'),p,C.sha(p))

if __name__=='__main__':unittest.main()
