#!/usr/bin/env python3
"""One locked ten-image final-target pilot; never a tuning callback."""
import csv
import gc
import json
import os
from pathlib import Path
import sys
import subprocess
import torch
import yaml

from v7_common import OUT,EXP,V6,TARGETS,REGISTRY,TIDS,fingerprint,sha,atomic_json
from campaign_v7 import ensure_models
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from strict_model_audit import audit_loaded_model
from common import classification_prompt


def main():
    focus=OUT/'focused_optimization';lp=focus/'final_pilot_lock.json'
    lock=json.loads(lp.read_text());cfg=lock['selected']['config'];label='focus_final_'+cfg['id'];cp=focus/'configs'/f'{label}.json'
    atomic_json(cp,cfg.get('options',{}))
    args=['--models',*cfg['models'],'--method',cfg.get('method','A3'),'--fusion',cfg.get('fusion','equal'),'--split','target_pilot','--limit-per-direction','2','--steps','50','--batch-size','2','--run-label',label,'--config-json',str(cp)]
    if cfg.get('di',True):args+=['--input-diversity']
    if cfg.get('ti'):args+=['--smoothing']
    # PNGs are generated and hashed completely BEFORE any final target load.
    from focus_search import command
    barrier=focus/'audits/final_target_access.json'
    if not barrier.exists():
        ensure_models(cfg['models']);command(EXP/'src/run_v7_attack.py',args,'final_pilot_attack.log')
    matches=list((OUT/'attacks/target_pilot').glob(label+'__*'))
    if len(matches)!=1:raise RuntimeError('Ambiguous frozen pilot')
    root=matches[0];states=[json.loads(p.read_text()) for p in sorted(root.glob('T*/batch_*/state.json'))]
    if len(states)!=5:raise RuntimeError('Missing frozen target pilot direction')
    for d in states:
        if d['image_ids']!=lock['pilot_ids'][d['transition_id']]:raise RuntimeError('Changed target cohort')
        for rel,h in d['png_hashes'].items():
            if sha(OUT/rel)!=h:raise RuntimeError('Frozen PNG changed')
    freeze=fingerprint([d['png_hashes'] for d in states])
    if barrier.exists():
        previous=json.loads(barrier.read_text())
        if previous['method_lock_sha256']!=sha(lp) or previous['frozen_pngs']!=freeze:raise RuntimeError('Refusing changed method/PNGs after final target access')
    else:atomic_json(barrier,dict(method_lock_sha256=sha(lp),frozen_pngs=freeze,feedback_to_search='forbidden'))
    prompt=classification_prompt(yaml.safe_load((V6/'config/runner_template.yaml').read_text()))
    summary=[]
    for name in TARGETS:
        dest=focus/'final_target_pilot'/f'{name.replace("/","_")}.json';identity=fingerprint([freeze,sha(lp),REGISTRY[name],prompt])
        if dest.exists() and json.loads(dest.read_text()).get('identity')==identity:
            result=json.loads(dest.read_text());summary+=result['directions'];continue
        ensure_models([name]);spec=REGISTRY[name];MODEL_REVISIONS[spec['repo_id']]=spec['revision']
        snapshot=local_snapshot(Path(os.environ['HF_HOME']),spec['repo_id'],spec['revision'])
        processor=load_processor(snapshot);model=load_target_for_generation(snapshot,torch.device('cuda'),precision='bf16')
        audit_loaded_model(model,model_name=name,revision=spec['revision'],role='final_target_pilot_evaluation_only',repo_id=spec['repo_id'],output_dir=OUT)
        gen=TransformersTargetGenerator(model,processor);rows=[]
        try:
            for d in states:
                for i,iid in enumerate(d['image_ids']):
                    with torch.inference_mode():clean,adv=gen.generate_labels([OUT/d['clean_paths'][i],OUT/d['adv_paths'][i]],prompt)
                    rows.append(dict(image_id=iid,transition_id=d['transition_id'],source_label=d['source_label'],target_label=d['target_label'],clean_raw=clean.raw_output,adv_raw=adv.raw_output,clean_label=clean.parsed_label,adv_label=adv.parsed_label,clean_status=clean.parser_status,adv_status=adv.parser_status,target_hit=adv.parsed_label==d['target_label']))
            directions=[dict(model=name,transition_id=t,hits=sum(r['target_hit'] for r in rows if r['transition_id']==t),denominator=sum(r['transition_id']==t for r in rows)) for t in TIDS]
            result=dict(identity=identity,method_lock_sha256=sha(lp),frozen_pngs=freeze,images=rows,directions=directions,pilot_only=True)
            atomic_json(dest,result);summary+=directions
        finally:del gen,model,processor;gc.collect();torch.cuda.empty_cache()
    report=focus/'reports';report.mkdir(parents=True,exist_ok=True)
    with (report/'final_target_pilot.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    high=len(summary)==15 and min(r['hits']/r['denominator'] for r in summary)>=.5 and sum(r['hits'] for r in summary)/sum(r['denominator'] for r in summary)>=.8
    atomic_json(report/'final_target_pilot_status.json',dict(high_native_pilot_gate=high,rows=summary,primary_30_image_benchmark_complete=False,small_sample_caveat='Only two images per direction; never claim the formal 15/30 criterion from this pilot'))

if __name__=='__main__':main()
