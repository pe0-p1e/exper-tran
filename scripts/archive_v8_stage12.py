#!/usr/bin/env python3
"""CPU-only compact V8 publication; never copies model weights, images or auth."""
import csv,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
EXP=REPO/'experiments/2026-10-a100-precommercial-v7-targeted-50'
MAIN=Path('/home/ubuntu/exper-tran');ROOT=MAIN/'outputs/a100_precommercial_v8_joint_selection24h'
DEST=EXP/'reports/v8_stage12_joint24h'
LOCK=MAIN/'outputs/a100_precommercial_v7_targeted50/proxy_composition_18h/publication.lock'
PATTERNS=['study_lock.json','window.json','method_lock.json','status.json','manifests/*.json','conditions/*.json','calibration/*.json','fingerprints/*.json',
 'amendments/*.json','preflight/environment.json','preflight/controller_restart_test_*.json','reports/*.csv','reports/*.md','reports/*.png',
 'reports/*.json','stage2/summary.json','stage2/blocker.json','stage2/pack_lock.json']
def publish():
 from analyze_v8_pairwise import generate
 generate(ROOT)
 DEST.mkdir(parents=True,exist_ok=True)
 for pattern in PATTERNS:
  for path in ROOT.glob(pattern):
   if path.name=='gradient_compatibility.csv' or path.stat().st_size>10*2**20:continue
   target=DEST/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
 audit=ROOT/'audits/model_dtype_audit.json'
 if audit.exists():
  rr=json.loads(audit.read_text());latest={}
  for r in rr:latest[(r['model_name'],r['role'])]=r
  (DEST/'BF16_audit.json').write_text(json.dumps(dict(load_events=len(rr),all_pass=all(r['status']=='pass' for r in rr),latest=list(latest.values())),indent=2))
 trace=ROOT/'reports/gradient_compatibility.csv'
 if trace.exists():
  groups={}
  with trace.open() as f:
   for r in csv.DictReader(f):
    key=(r['condition'],r['left'],r['right']);v=groups.setdefault(key,[0,0.,0.,0.,0.]);v[0]+=1;v[1]+=float(r['cos']);v[2]+=float(r['cancellation']);v[3]+=float(r['negative_dot']=='True');v[4]+=float(r.get('negative_pixel_fraction',0))
  with (DEST/'gradient_compatibility_aggregate.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['condition','left','right','observations','mean_cos','mean_cancellation','negative_dot_fraction','negative_pixel_fraction'])
   for key,v in groups.items():w.writerow([*key,v[0],*[x/v[0] for x in v[1:]]])
 with LOCK.open('a') as h:
  fcntl.flock(h,fcntl.LOCK_EX)
  subprocess.run(['git','add',str(DEST.relative_to(REPO))],cwd=REPO,check=True)
  if subprocess.run(['git','diff','--cached','--quiet','--',str(DEST.relative_to(REPO))],cwd=REPO).returncode:
   subprocess.run(['git','commit','-m','Archive compact V8 live measurements and provenance'],cwd=REPO,check=True)
  subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:experiment/a100-precommercial-v8-joint-selection-24h'],cwd=REPO,check=True)
if __name__=='__main__':
 while True:
  try:publish()
  except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
  state=json.loads((ROOT/'status.json').read_text()).get('state','') if (ROOT/'status.json').exists() else ''
  if state in ['COMPLETE','PARTIAL_OR_BLOCKED','PARTIAL_DEADLINE','PARTIAL_STAGE1_TIMEBOX','BLOCKED_REQUIRES_INTERVENTION']:break
  time.sleep(900)
