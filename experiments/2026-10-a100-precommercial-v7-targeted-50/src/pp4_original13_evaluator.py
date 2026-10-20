"""Grouped model reuse while preserving the historical clean/adv batch of two."""
import argparse
import gc
import json
import time
import pp4_worker as base
from pp4_original13_protocol import ROOT,OUT,EXP,plan,evidence,sha,atomic_json


def lock():
    from commercial_matrix_pack import freeze
    repo=EXP.parents[1]
    paths=[EXP/'src/pp4_original13_evaluator.py',EXP/'src/evaluate_v7.py',
           repo/'experiments/2026-10-a100-precommercial-v6/config/runner_template.yaml']
    freeze(ROOT/'evaluation_runner_lock.json',dict(source_hashes={str(p):sha(p) for p in paths},
        dtype='bf16',inference_batch='historical clean/adv pair of2',reuse='model weights only; no prediction cache',
        commercial_testing=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',required=True);args=parser.parse_args()
    locked=json.loads((ROOT/'protocol_lock.json').read_text())
    for n,h in locked['source_hashes'].items():
        if sha(EXP/'src'/n)!=h:raise RuntimeError('Scientific source changed: '+n)
    lock();base.ROOT=ROOT;base.install_audit()
    group=[r for r in plan() if r['target']==args.target]
    if not group or not all(evidence(r['condition'],deep=True)['attack_complete'] for r in group):
        raise RuntimeError('All target-group PNGs must be frozen')
    from primary_ml_cka.models.backends import target_transformers_generation as loader
    import evaluate_v7
    original=loader.load_target_for_generation;cache={}
    def reuse_model(snapshot,device,precision):
        if precision!='bf16':raise RuntimeError('BF16 required')
        key=(str(snapshot),str(device),precision)
        if cache and key not in cache:raise RuntimeError('Only one target per process')
        if key not in cache:cache[key]=original(snapshot,device,precision=precision)
        return cache[key]
    loader.load_target_for_generation=reuse_model;began=time.time()
    try:
        for row in group:
            c=row['condition'];evaluate_v7.test_evaluate(OUT/'attacks/test'/c['name'],args.target,
                ROOT/'target_locks'/(args.target.replace('/','_')+'.json'))
        atomic_json(ROOT/'performance'/(args.target.replace('/','_')+'.json'),dict(target=args.target,
            seconds=time.time()-began,conditions=len(group),target_loads=len(cache),
            inference_batch=2,evaluation_runner_lock_sha256=sha(ROOT/'evaluation_runner_lock.json')))
    finally:
        loader.load_target_for_generation=original;cache.clear();gc.collect()


if __name__=='__main__':main()
