#!/usr/bin/env python3
"""Read-only10-minute resource/throughput observation; never admits GPU work."""
import argparse,csv,json,os,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from resources import telemetry

def main():
 p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=10);p.add_argument('--interval',type=float,default=30);a=p.parse_args()
 dest=C.ROOT/'paper_closeout/reports/concurrency_throughput.csv';dest.parent.mkdir(parents=True,exist_ok=True)
 end=time.time()+a.minutes*60
 while time.time()<end:
  now=time.time();s=C.read(C.ROOT/'scheduler.json',{});t=telemetry();active=[j for j in s.get('active',[]) if C.same_process(j['process_identity'])]
  attacks=sum(x.stat().st_mtime>=now-300 for x in (C.ROOT/'attacks').glob('*/*/state.json'))
  evals=sum(x.stat().st_mtime>=now-300 for pattern in ['validation_images/*/*.json','large_validation_images/*/*/*.json'] for x in C.ROOT.glob(pattern))
  calls=set()
  for x in (C.ROOT/'paper_closeout').glob('*/responses/*/*/result.json'):
   if x.stat().st_mtime<now-300:continue
   d=C.read(x)
   if d.get('finished',0)>=now-300:calls.add((d['identity']['model'],d['request_id']))
  row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),workers=len(active),full_gpu_workers=sum(j.get('placement')=='full_gpu' for j in active),
   GPU_used_GiB=t['used'],GPU_utilization=t['utilization'],host_available_GiB=t['host_available'],
   reserved_host_caps_GiB=sum(j.get('host_reservation_gib',0) for j in active),
   attack_images_last5min=attacks,attack_images_per_minute_5min=attacks/5,
   open_target_images_last5min=evals,actual_GPT_requests_last5min=len(calls))
  exists=dest.exists()
  with dest.open('a') as f:
   w=csv.DictWriter(f,fieldnames=list(row))
   if not exists:w.writeheader()
   w.writerow(row);f.flush()
  C.atomic(dest.with_suffix('.json'),dict(latest=row,interpretation='Descriptive rolling5min; includes mixed conditions/evaluation phases; not a matched causal speedup benchmark',GPU_admission=False))
  time.sleep(a.interval)

if __name__=='__main__':main()
