#!/usr/bin/env python3
"""Unchanged four-layer visual objective on a frozen arbitrary-C dictionary."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from v7_common import EXP,REGISTRY,atomic_json,sha,fingerprint
from category_data import validate_dictionary
from category_features import CategoryVisionProxy,image_batch
from family_protocol import require_excluded_family
from run_v7_attack import METHODS,objective,project
from attack_optimizers import differentiable_resize_pad,normalize_pixel_gradients
from primary_ml_cka.experiment.attack_generation import _save_png_batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--target',required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    require_excluded_family(a.models,REGISTRY[a.target]['family'],REGISTRY)
    from run_category_generalization import require_selection_complete
    require_selection_complete()
    data=validate_dictionary(json.loads(a.manifest.read_text()),a.data_root)
    cfg={**METHODS['A3'],'patch_weight':.35,'fusion':'equal','input_diversity':True,'smoothing':False}
    source_hashes={n:sha(EXP/'src'/n) for n in ['category_attack.py','category_features.py','category_data.py','run_v7_attack.py','representation_objectives.py','vision_features.py','attack_optimizers.py']}
    identity_base=dict(config=cfg,models={n:REGISTRY[n] for n in a.models},manifest_sha256=sha(a.manifest),source_hashes=source_hashes,seed=42,steps=2 if a.smoke else 50)
    models=[]
    try:
        for name in a.models:
            m=CategoryVisionProxy(name,cfg['layers'],role='held_out_category_proxy')
            m.prepare_category_bank(data,a.data_root,a.output.parents[1]/'reference_cache');m.to('cpu');models.append(m)
            gc.collect();torch.cuda.empty_cache()
        total=sum(sum(p.numel()*p.element_size() for p in (m.model.model.vision_tower if m.spec['family']=='gemma4' else m.model).parameters()) for m in models)/2**30
        resident=total<=35
        if resident:
            for m in models:m.to('cuda')
        for task in (data['transitions'][:1] if a.smoke else data['transitions']):
            rows=task['images'][:2] if a.smoke else task['images']
            for start in range(0,len(rows),2):
                batch=rows[start:start+2];ids=[r['image_id'] for r in batch]
                identity=fingerprint(dict(base=identity_base,transition=task['id'],image_ids=ids))
                dest=a.output/task['id']/f'batch_{start:03d}';statefile=dest/'state.json'
                if statefile.exists():
                    state=json.loads(statefile.read_text())
                    if state.get('status')=='attack_complete' and state['identity']==identity and all((a.output/rel).is_file() and sha(a.output/rel)==h for rel,h in state['png_hashes'].items()):continue
                clean=image_batch(a.data_root,batch);noise_rng=torch.Generator(device='cuda').manual_seed(42)
                noise=torch.empty((len(rows),3,224,224),device='cuda').uniform_(-16/255,16/255,generator=noise_rng)[start:start+len(batch)]
                adv=project(clean+noise,clean).detach().requires_grad_(True);momentum=torch.zeros_like(adv)
                di_rng=torch.Generator().manual_seed(42+start);trace=[];counts=[(m.forward_calls,m.backward_calls,m.image_forwards) for m in models]
                torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
                for step in range(identity_base['steps']):
                    grads=[];losses=[];margins=[];shared_rng=di_rng.get_state()
                    for m in models:
                        if not resident:m.to('cuda')
                        di_rng.set_state(shared_rng);view=differentiable_resize_pad(adv,generator=di_rng)
                        loss,margin=objective(m,view,task['source_label'],task['target_label'],cfg)
                        grad=torch.autograd.grad(loss.sum(),adv)[0].detach().float();m.backward_calls+=1
                        if not torch.isfinite(grad).all() or (grad.flatten(1).abs().sum(1)==0).any():raise RuntimeError('Invalid arbitrary-C image gradient')
                        grads.append(grad);losses.append(loss.detach().cpu().tolist());margins.append(margin.detach().cpu().tolist())
                        if not resident:m.to('cpu');gc.collect();torch.cuda.empty_cache()
                    normalized=normalize_pixel_gradients(torch.stack(grads));fused=normalized.mean(0);momentum+=fused
                    cos=[torch.nn.functional.cosine_similarity(normalized[i].flatten(1),normalized[j].flatten(1)).tolist() for i in range(len(models)) for j in range(i+1,len(models))]
                    trace.append(dict(step=step+1,losses=losses,margins=margins,pairwise_cosines=cos))
                    adv=project(adv-momentum.sign()/255,clean).detach().requires_grad_(True)
                dest.mkdir(parents=True,exist_ok=True);lf,lp,_=_save_png_batch(dest,clean,adv.detach(),16/255)
                if lp>16/255+1e-7:raise RuntimeError('Arbitrary-C PNG epsilon violation')
                cleanpaths=[dest/f'{i:02d}_clean.png' for i in range(len(batch))];advpaths=[dest/f'{i:02d}_adv.png' for i in range(len(batch))]
                if torch.cuda.max_memory_reserved()/2**30>75:raise RuntimeError('Reserved VRAM exceeded75GiB')
                atomic_json(statefile,dict(status='attack_complete',identity=identity,manifest_sha256=sha(a.manifest),class_count=len(data['classes']),config=cfg,models=a.models,excluded_family=REGISTRY[a.target]['family'],source_label=task['source_label'],target_label=task['target_label'],transition=task['id'],image_ids=ids,steps=identity_base['steps'],seed=42,restarts=1,clean_paths=[str(x.relative_to(a.output)) for x in cleanpaths],adv_paths=[str(x.relative_to(a.output)) for x in advpaths],png_hashes={str(x.relative_to(a.output)):sha(x) for x in cleanpaths+advpaths},linf_float=lf,linf_png=lp,target_access=False,trace=trace,runtime_seconds=time.perf_counter()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,model_calls=[dict(model=m.name,vision_forwards=m.forward_calls-c[0],backwards=m.backward_calls-c[1],image_forwards=m.image_forwards-c[2]) for m,c in zip(models,counts)]))
                print('[category-attack]',len(data['classes']),task['id'],start+len(batch),'/',len(rows),flush=True)
    finally:
        for m in models:m.close()


if __name__=='__main__':main()
