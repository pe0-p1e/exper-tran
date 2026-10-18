#!/usr/bin/env python3
"""Small frozen-image native-output validation on HELD-OUT SURROGATES only."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import yaml

from v7_common import OUT,V6,TARGETS,REGISTRY,TIDS,records,sha,fingerprint,atomic_json
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from strict_model_audit import audit_loaded_model
from common import classification_prompt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--models',nargs='+',default=['Qwen3.5-9B','InternVL3.5-8B-HF','Gemma 4 E4B-it']);p.add_argument('--limit-per-direction',type=int,default=2)
    a=p.parse_args()
    if any(n in TARGETS for n in a.models):raise RuntimeError('Final target is forbidden in adaptive development')
    if not 1<=a.limit_per_direction<=10:raise ValueError('Pilot uses 1..10 development images per direction')
    selected={t:records(OUT/'manifests/dev'/f'{t}.jsonl')[:a.limit_per_direction] for t in TIDS}
    images={t:{} for t in TIDS};attack_models=set()
    for path in sorted(a.root.glob('T*/batch_*/state.json')):
        d=json.loads(path.read_text());attack_models.update(d['models']);tid=d['transition_id']
        if d['split']!='dev':raise RuntimeError('Surrogate tuning may only read development attacks')
        for i,iid in enumerate(d['image_ids']):
            if iid not in {r['image_id'] for r in selected[tid]}:continue
            c,adv=d['clean_paths'][i],d['adv_paths'][i]
            for rel in [c,adv]:
                if sha(OUT/rel)!=d['png_hashes'][rel]:raise RuntimeError('Modified frozen PNG')
            images[tid][iid]=dict(clean=c,adv=adv,source=d['source_label'],target=d['target_label'])
    if set(a.models)&attack_models:raise RuntimeError('Validation model was used to generate attack')
    if any(len(images[t])!=a.limit_per_direction for t in TIDS):raise RuntimeError('Incomplete pilot images')
    spec_manifest={t:[r['image_id'] for r in selected[t]] for t in TIDS}
    pilot_id=fingerprint(spec_manifest)[:12]
    atomic_json(OUT/'focused_optimization/manifests'/f'pilot_{pilot_id}.json',spec_manifest)
    prompt=classification_prompt(yaml.safe_load((V6/'config/runner_template.yaml').read_text()))
    for name in a.models:
        spec=REGISTRY[name];MODEL_REVISIONS[spec['repo_id']]=spec['revision']
        dest=OUT/'focused_optimization/native_validation'/pilot_id/a.root.name/(name.replace('/','_')+'.json')
        signature=fingerprint([spec,prompt,spec_manifest,{t:{i:{**v,'clean_hash':sha(OUT/v['clean']),'adv_hash':sha(OUT/v['adv'])} for i,v in rows.items()} for t,rows in images.items()}])
        if dest.exists() and json.loads(dest.read_text()).get('signature')==signature:
            print('[resume-native]',name,a.root.name,flush=True);continue
        snapshot=local_snapshot(Path(os.environ['HF_HOME']),spec['repo_id'],spec['revision'])
        processor=load_processor(snapshot);model=load_target_for_generation(snapshot,torch.device('cuda'),precision='bf16')
        audit_loaded_model(model,model_name=name,revision=spec['revision'],role='held_out_surrogate_native_validation',repo_id=spec['repo_id'],output_dir=OUT)
        generator=TransformersTargetGenerator(model,processor);rows=[]
        try:
            for tid in TIDS:
                for iid,entry in sorted(images[tid].items()):
                    # Model-, image- and prompt-specific clean answers are reusable.
                    ck=fingerprint([spec,prompt,sha(OUT/entry['clean'])])
                    cp=OUT/'focused_optimization/cache/clean_predictions'/f'{ck}.json'
                    with torch.inference_mode():
                        if cp.exists():
                            clean=json.loads(cp.read_text());adv=generator.generate_labels([OUT/entry['adv']],prompt)[0]
                        else:
                            c,adv=generator.generate_labels([OUT/entry['clean'],OUT/entry['adv']],prompt)
                            clean=dict(raw=c.raw_output,label=c.parsed_label,status=c.parser_status);atomic_json(cp,clean)
                    row=dict(transition_id=tid,image_id=iid,source_label=entry['source'],target_label=entry['target'],clean_raw=clean['raw'],clean_label=clean['label'],clean_status=clean['status'],adv_raw=adv.raw_output,adv_label=adv.parsed_label,adv_status=adv.parser_status,target_hit=adv.parsed_label==entry['target'])
                    rows.append(row)
                    print(f'[native-pilot] {name} {tid}: intended={entry["target"]} actual={adv.parsed_label}',flush=True)
            summaries=[dict(transition_id=t,hits=sum(r['target_hit'] for r in rows if r['transition_id']==t),denominator=sum(r['transition_id']==t for r in rows)) for t in TIDS]
            atomic_json(dest,dict(signature=signature,model=name,role='held_out_surrogate',condition=a.root.name,pilot_id=pilot_id,images=rows,directions=summaries,hits=sum(r['target_hit'] for r in rows),denominator=len(rows),final_target_feedback=False))
        finally:
            del generator,model,processor;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
