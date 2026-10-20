import sys,unittest,tempfile,json,subprocess,time,os,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as C
import resources as R
class CoreTests(unittest.TestCase):
 def test_family_rules(self):
  for fam,validator in C.VALIDATORS.items():
   for n in C.POOLS[fam]:self.assertTrue(C.legal({n:[0]},validator))
   with self.assertRaises(ValueError):C.legal({validator:[0]},validator)
  self.assertTrue(C.legal({C.I:[0],'InternVL3.5-4B-HF':[0]},C.VALIDATORS['qwen']))
 def test_native_layers(self):
  self.assertEqual(C.native(C.G,[.25,.5,.75,1]),[3,7,11,15])
  self.assertEqual(C.native(C.I,[.25,.5,.75,1]),[5,11,17,23])
  with self.assertRaises(ValueError):C.legal({C.G:[16]},'qwen')
 def test_condition_identity(self):
  a=C.make('qwen',{C.I:[0]},label='one');b=C.make('qwen',{C.I:[0]},label='alias');self.assertEqual(a['id'],b['id'])
  self.assertEqual(C.digest(C.scientific(a)),C.digest(C.scientific(b)))
  self.assertNotEqual(a['id'],C.make('qwen',{C.I:[1]})['id'])
 def test_rng_partition(self):
  import torch
  ids=['a','b','c'];one=[torch.rand(4,generator=torch.Generator().manual_seed(C.seed(i,'start'))) for i in ids]
  two=[torch.rand(4,generator=torch.Generator().manual_seed(C.seed(i,'start'))) for chunk in [ids[:2],ids[2:]] for i in chunk]
  for a,b in zip(one,two):self.assertTrue(torch.equal(a,b))
 def test_atomic_corrupt_and_resume(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.json';C.atomic(p,{'a':1});C.freeze(p,{'a':1})
   with self.assertRaises(RuntimeError):C.freeze(p,{'a':2})
   img=Path(d)/'x.png';img.write_bytes(b'original')
   C.atomic(p,dict(status='COMPLETE',png_hashes={str(img):C.sha(img)},linf_png=0,target_access=False))
   self.assertTrue(C.valid_receipt(p));img.write_bytes(b'corrupt');self.assertFalse(C.valid_receipt(p))
 def test_memory_admission_and_OOM(self):
  t=dict(total=80,used=0,free=80,processes=[],host_available=100)
  self.assertTrue(R.admit(t,[],24,4))
  self.assertFalse(R.admit({**t,'used':60,'free':20},[],24,4))
  self.assertFalse(R.admit(t,[],74,4))
  self.assertEqual(R.failure_kind('CUDA out of memory',1),'oom')
  self.assertEqual(R.failure_kind('shape mismatch',1),'code')
 def test_process_starttime_SIGTERM_SIGKILL(self):
  for sig in [signal.SIGTERM,signal.SIGKILL]:
   p=subprocess.Popen(['sleep','10']);old=C.proc_identity(p.pid);self.assertTrue(C.same_process(old));p.send_signal(sig);p.wait();self.assertFalse(C.same_process(old))
 def test_lock_and_restart(self):
  import fcntl
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lock'
   with p.open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code='import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
    self.assertNotEqual(subprocess.run([sys.executable,'-c',code,str(p)],capture_output=True).returncode,0)
   self.assertEqual(subprocess.run([sys.executable,'-c',code,str(p)],capture_output=True).returncode,0)
 def test_partial_never_complete(self):
  self.assertFalse(C.valid_receipt('/tmp/no_such_v8_file'))
 def test_PP_non_source_target_invariance(self):
  import torch
  from representation_objectives import original_pull_push_loss
  z=torch.randn(2,8,requires_grad=True);bank=torch.randn(10,48,8);copy=bank.clone();copy[2:]=torch.randn_like(copy[2:])*9
  a=original_pull_push_loss(z,bank,1,2,pull_weight=.5,push_weight=.5);ga=torch.autograd.grad(a,z)[0]
  b=original_pull_push_loss(z,copy,1,2,pull_weight=.5,push_weight=.5);gb=torch.autograd.grad(b,z)[0]
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ga,gb))
if __name__=='__main__':unittest.main()
