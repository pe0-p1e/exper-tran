"""Reuse existing predictive reservations, add measured free-memory guard."""
from common import *
import subprocess

def telemetry():
 line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 if len(line)!=1:raise RuntimeError('Expected one GPU; explicit multi-GPU routing unimplemented')
 total,used,free,util=map(float,line[0].split(','));pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();processes=[]
 for l in pids:
  parts=l.split(',')
  if len(parts)==2 and parts[0].strip().isdigit():processes.append(dict(pid=int(parts[0]),gib=float(parts[1])/1024 if parts[1].strip().isdigit() else 0))
 mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l and len(l.split())>1 and l.split()[1].isdigit()}
 return dict(total=total/1024,used=used/1024,free=free/1024,utilization=util,processes=processes,host_available=mem['MemAvailable']/2**20)

def estimate(c,mode):
 if mode in ['ila','ila_smoke']:return 20.0
 if mode=='attack':
  measured_profile=resident_profile(c)
  if measured_profile:return measured_profile['steady_GiB']
 import run_proxy_study as previous
 names=[c['validator']] if mode in ['evaluate','evaluate30'] else list(c['layers'])
 try:pred=previous.reservation(names,'evaluation' if mode in ['evaluate','evaluate30'] else 'attack',previous.weight_events())
 except KeyError:pred=32 if mode in ['evaluate','evaluate30'] else 28
 measurements=[]
 for path in (ROOT/'calibration').glob('*.json'):
  d=read(path)
  measurements += [m['reserved_gib'] for m in d.get('measurements',[]) if m['model'] in names]
 actual=[]
 for r in cohort(c['split'])['images']:
  d=read(image_receipt(c,r['image_id']))
  if d:actual.append(d['peak_reserved_gib'])
 # A maximum, not a falsely precise p95 from two samples.
 measured=max(measurements+actual,default=0)
 return max(pred,measured*1.15+1.5)

def admit(t,active,request,limit,startup_extra=0):
 if len(active)>=min(8,limit):return False
 headroom=max(8,t['total']*.1)
 own={j['process_identity']['pid'] for j in active}
 measured_own=sum(p['gib'] for p in t['processes'] if p['pid'] in own)
 foreign=max(0,t['used']-measured_own)
 # Full reservations account for growth of workers not yet at their peak.
 reserved=sum(j['reservation'] for j in active)
 return (reserved+request+foreign+startup_extra<=min(72,t['total']-headroom) and t['free']>=request+startup_extra+headroom and t['host_available']>=max(12,request*.8+6))

def failure_kind(text,returncode):
 s=text.lower()
 if 'out of memory' in s or returncode==-9:return 'oom'
 if any(x in s for x in ['connection reset','connection timed out','temporarily unavailable']):return 'transport'
 return 'code'


def resident_profile(c):
 # Use real resident batch1/2 measurements only for covered model sets and <=4 taps/model.
 candidates=[]
 for p in (ROOT/'calibration_resident_v2').glob('*.json'):
  d=read(p);plan=read(ROOT/'residency_plans'/p.name)
  job=read(ROOT/'jobs'/(p.stem+'_calibrate_resident.json'))
  if not job or not plan or not plan.get('resident') or d.get('status')!='PASS':continue
  bench=job['condition']
  if bench.get('parameter_dtype')!='bfloat16' or bench.get('CLS')!=0:continue
  if not all(n in bench['layers'] and len(ls)<=len(bench['layers'][n]) for n,ls in c['layers'].items()):continue
  peak=max(m['reserved_gib'] for m in d['measurements'])
  if not 0<peak<60:raise RuntimeError('Invalid resident memory measurement')
  candidates.append(dict(steady_GiB=peak*1.3+1.5,load_cap_GiB=12.,evidence=str(p),
   scope='calibrated small visual proxies; exact registry and <=4 taps; serialized initialization'))
 return min(candidates,key=lambda d:d['steady_GiB']) if candidates else None
