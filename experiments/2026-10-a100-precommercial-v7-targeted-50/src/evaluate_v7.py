#!/usr/bin/env python3
"""Post-freeze evaluation. Dev uses ONLY held-out surrogates, test requires method lock."""
import argparse
import gc
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml

from v7_common import OUT, V6, TARGETS, REGISTRY, records, images, atomic_json, sha, fingerprint
from vision_features import VisionProxy
from representation_objectives import class_scores


def load_pngs(paths):
    from primary_ml_cka.artifacts.png import load_png_tensor
    return torch.stack([load_png_tensor(OUT/p) for p in paths]).cuda()


def dev_evaluate(root, validators):
    paths=sorted(root.glob('*/batch_*/state.json'))
    if not paths:raise RuntimeError(f'No frozen attacks under {root}')
    for name in validators:
        # Held-out proxy weights must NEVER be in the attack model list.
        for path in paths:
            if name in json.loads(path.read_text())['models']:raise RuntimeError('Validator in attack model set')
        m=VisionProxy(name,role="held_out_surrogate");m.prepare_bank()
        try:
            for path in paths:
                state=json.loads(path.read_text())
                dest=OUT/'development_evaluations'/root.name/name.replace('/','_')/path.parent.parent.name/path.parent.name/'metrics.json'
                identity=fingerprint([state['identity'],state['png_hashes'],REGISTRY[name]])
                if dest.exists() and json.loads(dest.read_text()).get('identity')==identity:continue
                for png,h in state['png_hashes'].items():
                    if sha(OUT/png)!=h:raise RuntimeError('Frozen PNG changed')
                src,tgt=state['source_label'],state['target_label']; rows=[]
                with torch.inference_mode():
                    for i in range(len(state['image_ids'])):
                        clean=load_pngs([state['clean_paths'][i]]);adv=load_pngs([state['adv_paths'][i]])
                        cz=m.features(clean)[m.layers[0]][0]; az=m.features(adv)[m.layers[0]][0]
                        refs=m.bank[m.layers[0]][0]
                        cs=class_scores(cz,refs); ads=class_scores(az,refs)
                        other=ads.clone();other[:,tgt-1]=-torch.inf
                        rows.append(dict(image_id=state['image_ids'][i],representation_hit=bool(ads.argmax(1).item()==tgt-1),target_margin=float(ads[0,tgt-1]-other.max()),delta_R=float((ads[0,tgt-1]-cs[0,tgt-1])+(cs[0,src-1]-ads[0,src-1]))))
                atomic_json(dest,dict(identity=identity,validator=name,role='held_out_surrogate',target_model=False,images=rows,source_state=str(path.relative_to(OUT))))
        finally:m.close()


def test_evaluate(root, target, lock_override=None):
    lock=lock_override or OUT/'audits/method_lock.json'
    if not lock.is_file():raise RuntimeError('Final target evaluation requires immutable method lock')
    decision=json.loads(lock.read_text())
    if root.name not in decision['allowed_test_conditions']:raise RuntimeError('Condition not predeclared in lock')
    from family_protocol import require_condition_target
    for state_path in sorted(root.glob('*/batch_*/state.json')):
        actual=json.loads(state_path.read_text())
        require_condition_target(root.name,target,actual['models'],decision,REGISTRY)
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
    from strict_model_audit import audit_loaded_model
    from common import classification_prompt
    import os
    spec=REGISTRY[target];MODEL_REVISIONS[spec['repo_id']]=spec['revision']
    snapshot=local_snapshot(Path(os.environ['HF_HOME']),spec['repo_id'],spec['revision'])
    processor=load_processor(snapshot);model=load_target_for_generation(snapshot,torch.device('cuda'),precision='bf16')
    audit_loaded_model(model,model_name=target,revision=spec['revision'],role='target',repo_id=spec['repo_id'],output_dir=OUT)
    generator=TransformersTargetGenerator(model,processor)
    prompt=classification_prompt(yaml.safe_load((V6/'config/runner_template.yaml').read_text()))
    try:
        for path in sorted(root.glob('*/batch_*/state.json')):
            state=json.loads(path.read_text());dest=OUT/'evaluations'/root.name/target.replace('/','_')/path.parent.parent.name/path.parent.name/'metrics.json'
            identity=fingerprint([state['identity'],state['png_hashes'],REGISTRY[target],sha(lock)])
            if dest.exists() and json.loads(dest.read_text()).get('identity')==identity:continue
            for png,h in state['png_hashes'].items():
                if sha(OUT/png)!=h:raise RuntimeError('Frozen PNG changed')
            rows=[]
            for i,iid in enumerate(state['image_ids']):
                imagepaths=[OUT/state['clean_paths'][i],OUT/state['adv_paths'][i]]
                with torch.inference_mode():clean,adv=generator.generate_labels(imagepaths,prompt)
                rows.append(dict(image_id=iid,clean_raw=clean.raw_output,adv_raw=adv.raw_output,clean_label=clean.parsed_label,adv_label=adv.parsed_label,clean_parser=clean.parser_status,adv_parser=adv.parser_status,target_hit=adv.parsed_label==state['target_label'],clean_correct=clean.parsed_label==state['source_label']))
            atomic_json(dest,dict(identity=identity,target=target,images=rows,source_state=str(path.relative_to(OUT)),cohort='test',method_lock_sha256=sha(lock)))
    finally:
        del generator,model,processor;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--validators',nargs='+',default=['Qwen3.5-4B','InternVL3.5-4B-HF']);p.add_argument('--target',choices=TARGETS);p.add_argument('--method-lock',type=Path)
    a=p.parse_args()
    if a.target:test_evaluate(a.root,a.target,a.method_lock)
    else:dev_evaluate(a.root,a.validators)

if __name__=='__main__':main()
