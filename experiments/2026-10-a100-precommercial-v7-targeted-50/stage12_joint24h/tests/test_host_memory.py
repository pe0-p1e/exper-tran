import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import resources as R

class HostMemoryTests(unittest.TestCase):
 def test_steady_cap_waits_for_actual_model_execution(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'memory.stat').write_text('anon 10737418240\nfile 21474836480\nkernel 0\n')
   job=dict(host_cgroup=d,host_reservation_gib=30)
   self.assertIsNone(R.steady_host_cap(job,dict(phase='starting')))
   cap,high=R.steady_host_cap(job,dict(step=1))
   self.assertEqual(high,10);self.assertEqual(cap,15.5)
   self.assertEqual(R.steady_host_cap(job,dict(phase='validation'))[0],15.5)
 def test_steady_cap_preserves_highwater_and_load_limit(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'memory.stat').write_text('anon 1073741824\nkernel 0\n')
   cap,high=R.steady_host_cap(dict(host_cgroup=d,host_reservation_gib=30,host_load_reservation_gib=30,steady_host_highwater_gib=12),dict(step=50))
   self.assertEqual(cap,18);self.assertEqual(high,12)
 def test_logical_pressure_can_use_spare_vram(self):
  self.assertTrue(R.prefer_full_gpu(dict(host_total=118,host_available=74),22,[dict(host_reservation_gib=70)]))
 def test_full_gpu_load_host_bound_is_max_not_sum(self):
  values=[dict(model_name=n,revision='r',all_floating_parameters_bf16=True,parameter_bytes_by_dtype={'bfloat16':v*2**30}) for n,v in [('a',4),('b',10)]]
  def read(p,default=None):return values if 'audit' in str(p) else {}
  with patch.object(R,'read',side_effect=read),patch.object(R,'REG',{'a':{'revision':'r'},'b':{'revision':'r'}}):
   self.assertEqual(R.full_gpu_host_estimate(dict(layers={'a':[1],'b':[1]})),19)
 def test_clip_only_uses_measured_model_specific_loader_budget(self):
  import common as C
  with tempfile.TemporaryDirectory() as d,patch.object(R,'ROOT',Path(d)):
   root=Path(d)
   C.atomic(root/'calibration/x.json',dict(status='PASS',measurements=[dict(model=R.V,batch=1,reserved_gib=1.04),dict(model=R.V,batch=2,reserved_gib=1.2)]))
   C.atomic(root/'jobs/x_calibrate.json',dict(condition=dict(parameter_dtype='bfloat16',CLS=0,layers={R.V:[5,11,17,23]})))
   profile=R.resident_profile(dict(layers={R.V:[11,14]}))
   self.assertAlmostEqual(profile['steady_GiB'],2.56);self.assertAlmostEqual(profile['load_cap_GiB'],2.56)
   self.assertIsNone(R.resident_profile(dict(layers={R.V:[1,2,3,4,5]})))
 def test_spare_host_ram_favors_vision_only_gpu(self):
  self.assertFalse(R.prefer_full_gpu(dict(host_total=118,host_available=106),20))
  self.assertTrue(R.prefer_full_gpu(dict(host_total=118,host_available=35),20))
 def test_future_loading_growth_is_reserved(self):
  jobs=[dict(host_reservation_gib=20,process_identity=dict(pid=1))]
  with patch.object(R,'proc_rss_gib',return_value=2):
   self.assertFalse(R.host_admit(dict(host_total=118,host_available=45),jobs,20))
   self.assertTrue(R.host_admit(dict(host_total=118,host_available=52),jobs,20))
 def test_aggregate_physical_caps(self):
  jobs=[dict(host_reservation_gib=20,process_identity=dict(pid=i)) for i in range(5)]
  with patch.object(R,'proc_rss_gib',return_value=20):
   self.assertFalse(R.host_admit(dict(host_total=118,host_available=40),jobs,20))
 def test_idle_vram_does_not_override_host_guard(self):
  t=dict(total=80,used=0,free=80,processes=[],host_total=118,host_available=20)
  self.assertFalse(R.admit(t,[],8,8,host_request=20))
 def test_full_gpu_placement_reserves_parameter_bytes(self):
  with patch.object(R,'read',return_value=[dict(model_name='x',revision='r',all_floating_parameters_bf16=True,parameter_bytes_by_dtype=dict(bfloat16=10*2**30))]),patch.object(R,'REG',{'x':dict(revision='r')}):
   self.assertEqual(R.full_gpu_reservation(dict(layers={'x':[1]}),7),18)

