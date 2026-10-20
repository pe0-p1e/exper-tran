"""Test attack or post-freeze grouped target evaluation; strictly separate processes."""
import argparse
import json
import sys
import time
from pathlib import Path
import pp4_worker as base
from pp4_original13_protocol import ROOT,OUT,EXP,plan,smoke,evidence,sha,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['attack','evaluation'])
    parser.add_argument('--condition');parser.add_argument('--target');args=parser.parse_args()
    locked=json.loads((ROOT/'protocol_lock.json').read_text())
    for name,h in locked['source_hashes'].items():
        if sha(EXP/'src'/name)!=h:raise RuntimeError('Scientific source changed: '+name)
    base.ROOT=ROOT;base.install_audit()
    rows=plan()
    if args.mode=='attack':
        c=next(c for c in [r['condition'] for r in rows]+[smoke()] if c['id']==args.condition)
        if c['split']=='dev':
            base.load_condition=lambda _:c
            sys.argv=[sys.argv[0],'attack','--condition',c['id']];base.main();return
        import run_v7_attack as runner
        from pp4_objective import objective
        from family_protocol import require_excluded_family
        require_excluded_family(c['models'],c['options']['excluded_proxy_family'],runner.REGISTRY)
        # After every model has loaded, distinguish initialization peak from
        # steady vision-only execution. This allows measured safe concurrency.
        original_proxy=runner.VisionProxy;proxies=[];marked=False
        def proxy_factory(*a,**kw):
            model=original_proxy(*a,**kw);proxies.append(model);return model
        def marked_objective(*a,**kw):
            nonlocal marked
            result=objective(*a,**kw)
            if not marked:
                import torch
                from run_proxy_study import weight_events
                events=weight_events();resident=[];activations=[];cpu_bytes=0
                for proxy in proxies:
                    event=events[proxy.name]
                    total=sum(event['parameter_bytes_by_dtype'].values())/2**30
                    visual=sum(event['vision_encoder_parameter_bytes_by_dtype'].values())/2**30
                    r=visual if proxy.spec['family']=='gemma4' else total;resident.append(r)
                    calibration=base.shared.ROOT/'calibration'/(proxy.name.replace('/','_')+'.json')
                    measurements=json.loads(calibration.read_text())['measurements'] if calibration.exists() else []
                    activations.append(max(2.5,max((m['peak_reserved_gib']-r for m in measurements),default=2.5)))
                    cpu_bytes+=sum(p.numel()*p.element_size() for p in proxy.model.parameters() if p.device.type=='cpu')
                steady=(sum(resident) if sum(resident)<=35 else max(resident))+max(activations)+4
                atomic_json(ROOT/'runtime'/(c['id']+'.json'),dict(initialization_complete=True,
                    steady_reservation_gib=max(steady,torch.cuda.memory_reserved()/2**30+4),
                    steady_cpu_parameter_gib=cpu_bytes/2**30,at=time.time()))
                marked=True
            return result
        runner.VisionProxy=proxy_factory;runner.objective=marked_objective
        runner.METHODS['PP4']=dict(c['options']);runner.prepare_data=base.shared.verify_existing_data
        sys.argv=[sys.argv[0],'--models',*c['models'],'--method','PP4','--fusion','equal','--split','test',
            '--steps','50','--seed','42','--restarts','1','--batch-size','2','--input-diversity',
            '--run-label',c['id'],'--excluded-proxy-family',c['options']['excluded_proxy_family'],
            '--config-json',str(ROOT/'configs'/(c['id']+'.json'))]
        runner.main()
    else:
        group=[r for r in rows if r['target']==args.target]
        if not group or not all(evidence(r['condition'],deep=True)['attack_complete'] for r in group):
            raise RuntimeError('All target-group PNGs must be frozen before target loading')
        from primary_ml_cka.models.backends import target_transformers_generation as loader
        from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
        import evaluate_v7
        original_loader=loader.load_target_for_generation;original_generate=TransformersTargetGenerator.generate_labels
        cache={};answers={};counts=dict(cache_hits=0,new_image_generations=0)
        def cached_loader(snapshot,device,precision):
            if precision!='bf16':raise RuntimeError('BF16 required')
            key=(str(snapshot),str(device),precision)
            if cache and key not in cache:raise RuntimeError('Only one target per process')
            if key not in cache:cache[key]=original_loader(snapshot,device,precision=precision)
            return cache[key]
        def cached_generate(self,paths,prompt):
            result=[]
            for path in paths:
                key=(sha(path),prompt)
                if key not in answers:
                    answers[key]=original_generate(self,[path],prompt)[0];counts['new_image_generations']+=1
                else:counts['cache_hits']+=1
                result.append(answers[key])
            return result
        loader.load_target_for_generation=cached_loader;TransformersTargetGenerator.generate_labels=cached_generate
        began=time.time()
        try:
            for row in group:
                c=row['condition'];evaluate_v7.test_evaluate(OUT/'attacks/test'/c['name'],args.target,
                    ROOT/'target_locks'/(args.target.replace('/','_')+'.json'))
            atomic_json(ROOT/'performance'/(args.target.replace('/','_')+'.json'),dict(target=args.target,
                seconds=time.time()-began,conditions=len(group),target_loads=len(cache),**counts))
        finally:
            loader.load_target_for_generation=original_loader;TransformersTargetGenerator.generate_labels=original_generate


if __name__=='__main__':main()
