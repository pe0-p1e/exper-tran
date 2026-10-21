"""Reuse existing predictive reservations, add measured free-memory guard."""
from common import *
import subprocess
from host_memory import proc_rss_gib,snapshot,BASE

def telemetry():
 line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 if len(line)!=1:raise RuntimeError('Expected one GPU; explicit multi-GPU routing unimplemented')
 total,used,free,util=map(float,line[0].split(','));pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();processes=[]
 for l in pids:
  parts=l.split(',')
  if len(parts)==2 and parts[0].strip().isdigit():processes.append(dict(pid=int(parts[0]),gib=float(parts[1])/1024 if parts[1].strip().isdigit() else 0))
 mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l and len(l.split())>1 and l.split()[1].isdigit()}
 return dict(total=total/1024,used=used/1024,free=free/1024,utilization=util,processes=processes,host_available=mem['MemAvailable']/2**20,
  host_total=mem['MemTotal']/2**20,host_cgroup=snapshot(BASE) if (BASE/'memory.max').exists() else None)

def host_estimate(c,mode):
 # Parameter bytes are actual audited full BF16 checkpoints, including unused CPU language weights.
 names=[c['validator']] if mode in ['evaluate','evaluate30'] else list(c['layers'])
 audit=read(ROOT/'audits/model_dtype_audit.json',[]);weights={}
 for a in audit:
  if a.get('all_floating_parameters_bf16') and a.get('revision')==REG.get(a['model_name'],{}).get('revision'):
   weights[a['model_name']]=sum(a['parameter_bytes_by_dtype'].values())/2**30
 missing=set(names)-weights.keys()
 if missing:raise RuntimeError('Host RAM calibration missing audited model bytes: '+str(sorted(missing)))
 # Full-parameter loader/offload peak + framework/cache allowance. Never shrink parameters to fit.
 baseline=sum(weights[n] for n in names)*1.15+3
 history=read(ROOT/'host_memory_peaks.json',{})
 observed=history.get(digest([mode,sorted(names)]),0)
 return max(6.,baseline,observed*1.15+1)

def full_gpu_reservation(c,base):
 audit=read(ROOT/'audits/model_dtype_audit.json',[]);weights={}
 for a in audit:
  if a.get('all_floating_parameters_bf16') and a.get('revision')==REG.get(a['model_name'],{}).get('revision'):
   weights[a['model_name']]=sum(a['parameter_bytes_by_dtype'].values())/2**30
 if any(n not in weights for n in c['layers']):return None
 # Deliberately overestimate by adding all full weights to the vision-calibrated reservation.
 return base+sum(weights[n] for n in c['layers'])*1.1

def prefer_full_gpu(t,host_request):
 # CPU offload is the default while RAM is plentiful: unused LM weights must
 # not consume scarce GPU admission slots ahead of runnable vision work.
 return t['host_available'] < 2*host_request+max(12,t['host_total']*.1)

def host_admit(t,active,request):
 reserve=max(12,t['host_total']*.1)
 caps=sum(j.get('host_reservation_gib',proc_rss_gib(j['process_identity']['pid'])+4) for j in active)
 growth=sum(max(0,j.get('host_reservation_gib',proc_rss_gib(j['process_identity']['pid'])+4)-proc_rss_gib(j['process_identity']['pid'])) for j in active)
 return caps+request<=t['host_total']-reserve and t['host_available']>=request+growth+reserve

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

def admit(t,active,request,limit,startup_extra=0,host_request=None):
 if len(active)>=min(8,limit):return False
 headroom=max(8,t['total']*.1)
 own={j['process_identity']['pid'] for j in active}
 measured_own=sum(p['gib'] for p in t['processes'] if p['pid'] in own)
 foreign=max(0,t['used']-measured_own)
 # Full reservations account for growth of workers not yet at their peak.
 reserved=sum(j['reservation'] for j in active)
 return (reserved+request+foreign+startup_extra<=min(72,t['total']-headroom) and t['free']>=request+startup_extra+headroom and
  (host_admit(t,active,host_request) if host_request is not None else t['host_available']>=max(12,request*.8+6)))

def failure_kind(text,returncode):
 s=text.lower()
 if 'out of memory' in s or returncode==-9:return 'oom'
 if any(x in s for x in ['connection reset','connection timed out','temporarily unavailable']):return 'transport'
 return 'code'


def resident_profile(c):
 # CLIP-only jobs must not inherit a three-model pool's loader/activation reserve.
 # The isolated first-model batch1/2 measurements include the exact four taps,
 # full CLIP loader and references; other measured pool contamination only raises
 # this number, so use a verified complete batch pair and retain explicit margin.
 if list(c['layers'])==[V] and len(c['layers'][V])<=4:
  profiles=[]
  for p in (ROOT/'calibration').glob('*.json'):
   d=read(p);job=read(ROOT/'jobs'/(p.stem+'_calibrate.json'))
   if d.get('status')!='PASS' or not job:continue
   bench=job['condition']
   if bench.get('parameter_dtype')!='bfloat16' or bench.get('CLS')!=0 or V not in bench['layers'] or len(bench['layers'][V])<len(c['layers'][V]):continue
   ms=[m for m in d.get('measurements',[]) if m['model']==V]
   if {m['batch'] for m in ms}!={1,2}:continue
   peak=max(m['reserved_gib'] for m in ms)
   profiles.append(dict(steady_GiB=peak*1.3+1.,load_cap_GiB=peak*1.3+1.,evidence=str(p),scope='CLIP-only full loader plus measured batch1/2; <=4 taps; exact pinned study registry'))
  if profiles:return min(profiles,key=lambda d:d['steady_GiB'])
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
