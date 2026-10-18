#!/usr/bin/env python3
"""Actual full-objective gradient calibration with native visual preprocessing."""
import argparse
import gc
import time
import torch
from v7_common import OUT, TIDS, records, images, atomic_json
from vision_features import VisionProxy
from run_v7_attack import objective, METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);a=p.parse_args()
    results=[]
    for name in a.models:
        m=VisionProxy(name,(.25,.5,.75,1.));m.prepare_bank()
        trials=[]
        try:
            source=records(OUT/'manifests/dev/T01.jsonl')
            for bs in [2,8,16,30]:
                try:
                    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    x=images([source[i%len(source)] for i in range(bs)]).requires_grad_(True)
                    torch.cuda.synchronize();start=time.perf_counter()
                    loss,_=objective(m,x,1,2,METHODS['A3']);g=torch.autograd.grad(loss.sum(),x)[0]
                    torch.cuda.synchronize();sec=time.perf_counter()-start
                    peak=torch.cuda.max_memory_reserved()/2**30
                    assert torch.isfinite(g).all() and g.abs().sum()>0
                    trials.append(dict(batch=bs,seconds=sec,peak_reserved_gib=peak,safe=peak<=70))
                    del x,g,loss
                    if peak>70:break
                except torch.OutOfMemoryError:
                    trials.append(dict(batch=bs,oom=True,safe=False));gc.collect();torch.cuda.empty_cache();break
            results.append(dict(model=name,trials=trials,largest_safe_batch=max((r['batch'] for r in trials if r.get('safe')),default=0)))
            atomic_json(OUT/'audits/performance_benchmark.json',dict(native_resolution=True,objective='A3 four-layer margin',max_reserved_gib=75,results=results))
        finally:m.close()

if __name__=='__main__':main()
