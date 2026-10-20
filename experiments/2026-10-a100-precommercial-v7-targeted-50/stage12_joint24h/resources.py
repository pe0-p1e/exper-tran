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
 import run_proxy_study as previous
 names=[c['validator']] if mode=='evaluate' else list(c['layers'])
 try:pred=previous.reservation(names,'evaluation' if mode=='evaluate' else 'attack',previous.weight_events())
 except KeyError:pred=32 if mode=='evaluate' else 28
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

def admit(t,active,request,limit):
 if len(active)>=min(4,limit):return False
 headroom=max(8,t['total']*.1)
 own={j['process_identity']['pid'] for j in active}
 measured_own=sum(p['gib'] for p in t['processes'] if p['pid'] in own)
 foreign=max(0,t['used']-measured_own)
 # Full reservations account for growth of workers not yet at their peak.
 reserved=sum(j['reservation'] for j in active)
 return (reserved+request+foreign<=min(72,t['total']-headroom) and t['free']>=request+headroom and t['host_available']>=max(12,request*.8+6))

def failure_kind(text,returncode):
 s=text.lower()
 if 'out of memory' in s or returncode==-9:return 'oom'
 if any(x in s for x in ['connection reset','connection timed out','temporarily unavailable']):return 'transport'
 return 'code'
