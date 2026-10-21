import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import resources as R

class HostMemoryTests(unittest.TestCase):
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

