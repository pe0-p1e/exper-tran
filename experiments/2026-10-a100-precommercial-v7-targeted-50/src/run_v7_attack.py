#!/usr/bin/env python3
"""Resumable surrogate-only V7 attack; no target loaders or feedback."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from v7_common import OUT, TARGETS, REGISTRY, TIDS, records, images, sha, fingerprint, atomic_json, prepare_data
from vision_features import VisionProxy
from representation_objectives import class_scores, all_class_margin_loss, original_pull_push_loss, local_patch_target_loss
from attack_optimizers import fuse_proxy_gradients, normalize_pixel_gradients, adaptive_weights_from_surrogate_lookahead, differentiable_resize_pad
from primary_ml_cka.experiment.attack_generation import _save_png_batch

# Predeclared controlled ladder; each row changes one mechanism from a named parent.
METHODS={
 'A0':dict(loss='pull_push',aggregation='centroid',layers=[1.]),
 'A1':dict(loss='margin',aggregation='centroid',layers=[1.]),
 'A2':dict(loss='margin',aggregation='soft_reference',layers=[1.]),
 'A3':dict(loss='margin',aggregation='centroid',layers=[.25,.5,.75,1.]),
 'A4':dict(loss='margin',aggregation='centroid',layers=[1.],patch_weight=.2),
 'A5':dict(loss='margin',aggregation='soft_reference',layers=[.25,.5,.75,1.]),
 'L0L1':dict(loss='combined',aggregation='centroid',layers=[1.]),
 'PATCH_ONLY':dict(loss='margin',aggregation='centroid',layers=[1.],patch_weight=1.),
 'LAYER25':dict(loss='margin',aggregation='centroid',layers=[.25]),
 'LAYER50':dict(loss='margin',aggregation='centroid',layers=[.5]),
 'LAYER75':dict(loss='margin',aggregation='centroid',layers=[.75]),
 'TWO_LAYER':dict(loss='margin',aggregation='centroid',layers=[.5,1.]),
}


def project(x,clean):
    return torch.maximum(torch.minimum(x,clean+16/255),clean-16/255).clamp(0,1)


def objective(proxy,x,source,target,config):
    fs=proxy.features(x);losses=[];margins=[]
    for l,(z,tokens,mask) in fs.items():
        bank,patches=proxy.bank[l]
        scores=class_scores(z,bank,aggregation=config['aggregation'])
        negatives=scores.clone();negatives[:,target-1]=-torch.inf
        margins.append(scores[:,target-1]-negatives.max(1).values)
        if config['loss']=='pull_push':loss=original_pull_push_loss(z,bank,source,target,reduction='none')
        else:
            loss=all_class_margin_loss(scores,target,reduction='none')
            if config['loss']=='combined':loss=.5*loss+.5*original_pull_push_loss(z,bank,source,target,reduction='none')
        pw=config.get('patch_weight',0.)
        if pw:
            patch=local_patch_target_loss(tokens,patches[target-1],patches[source-1],patch_mask=mask,reduction='none')
            loss=(1-pw)*loss+pw*patch
        losses.append(loss)
    return torch.stack(losses).mean(0),torch.stack(margins).mean(0)


def valid_state(path,identity):
    if not path.is_file():return None
    d=json.loads(path.read_text())
    if d.get('identity')!=identity or d.get('status')!='attack_complete':return None
    for p,h in d['png_hashes'].items():
        f=OUT/p
        if not f.is_file() or sha(f)!=h:return None
    return d


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--method',choices=METHODS,default='A0')
    p.add_argument('--fusion',choices=['equal','weighted','pcgrad','weighted_pcgrad','consensus'],default='equal')
    p.add_argument('--transitions',nargs='+',default=TIDS)
    p.add_argument('--split',choices=['dev','test','smoke'],default='dev')
    p.add_argument('--steps',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--input-diversity',action='store_true');p.add_argument('--smoothing',action='store_true')
    p.add_argument('--restarts',type=int,default=1)
    args=p.parse_args()
    if any(n in TARGETS for n in args.models):raise RuntimeError('Final target models forbidden in attack process')
    if args.steps<1 or args.batch_size<1 or args.restarts<1:raise ValueError('Positive budgets required')
    prepare_data()
    config=dict(METHODS[args.method]);config.update(fusion=args.fusion,input_diversity=args.input_diversity,smoothing=args.smoothing)
    name=f"{args.method}__{args.fusion}__"+'+'.join(n.replace('/','_').replace(' ','_') for n in args.models)
    if args.input_diversity:name+='__DI'
    if args.smoothing:name+='__TI'
    name+=f'__s{args.seed}__i{args.steps}__r{args.restarts}'
    config_hash=fingerprint(dict(config=config,models={n:REGISTRY[n] for n in args.models},code={f:sha(Path(__file__).parent/f) for f in ['run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py']}))
    directory=OUT/'attacks'/args.split/name
    proxies=[]
    try:
        # Loading one model at a time also verifies the exact pinned visual depth.
        for n in args.models:
            model=VisionProxy(n,config['layers']);model.prepare_bank();model.to('cpu');proxies.append(model)
            gc.collect();torch.cuda.empty_cache()
        # Models in this initial candidate pool fit together. Larger pools offload sequentially.
        total=sum(sum(p.numel()*p.element_size() for p in m.model.parameters()) for m in proxies)/2**30
        resident=total<=35
        if resident:
            for m in proxies:m.to('cuda')
        def activate(m):
            if not resident:m.to('cuda')
        def park(m):
            if not resident:m.to('cpu');gc.collect();torch.cuda.empty_cache()
        for tid in args.transitions:
            manifest=OUT/'manifests'/('dev' if args.split=='smoke' else args.split)/f'{tid}.jsonl'
            all_records=records(manifest)
            if args.split=='smoke':all_records=all_records[:2]
            source=int(all_records[0]['human_label']);target=source+1
            for start in range(0,len(all_records),args.batch_size):
                batch=all_records[start:start+args.batch_size]
                ids=[r['image_id'] for r in batch]
                identity=fingerprint(dict(config=config_hash,manifest=sha(manifest),ids=ids,seed=args.seed,steps=args.steps,restarts=args.restarts))
                dest=directory/tid/f'batch_{start:03d}';state=dest/'state.json'
                if valid_state(state,identity):print(f'[resume] {name}/{tid}/{start}',flush=True);continue
                clean=images(batch);best=None;best_loss=None;traces=[]
                counts_before=[(m.forward_calls,m.backward_calls,m.image_forwards) for m in proxies]
                torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
                rng=torch.Generator(device='cuda').manual_seed(args.seed)
                # Generate noise for entire cohort to make starts independent of batch size.
                noise=torch.empty((len(all_records),3,224,224),device='cuda').uniform_(-16/255,16/255,generator=rng)[start:start+len(batch)]
                for restart in range(args.restarts):
                    if restart:noise=torch.empty_like(clean).uniform_(-16/255,16/255,generator=rng)
                    adv=project(clean+noise,clean).detach().requires_grad_(True)
                    momentum=torch.zeros_like(adv)
                    di_rng=torch.Generator().manual_seed(args.seed+restart*10000+start)
                    history=[]
                    for step in range(args.steps):
                        grads=[];loss_values=[];margin_values=[]
                        # Shared DI view across proxies; candidates for adaptive weights are unaugmented.
                        di_state=di_rng.get_state()
                        for m in proxies:
                            activate(m);di_rng.set_state(di_state)
                            view=differentiable_resize_pad(adv,generator=di_rng) if args.input_diversity else adv
                            loss,margin=objective(m,view,source,target,config)
                            grad=torch.autograd.grad(loss.sum(),adv)[0].detach().float();m.backward_calls+=1
                            if not torch.isfinite(grad).all() or (grad.flatten(1).abs().sum(1)==0).any():raise RuntimeError(f'Invalid gradient: {m.name}/{tid}/step{step}')
                            grads.append(grad);loss_values.append(loss.detach());margin_values.append(margin.detach());park(m)
                        stack=torch.stack(grads);ng=normalize_pixel_gradients(stack)
                        weights=None
                        if args.fusion in ('weighted','weighted_pcgrad') and len(proxies)>1:
                            improvements=torch.zeros((len(proxies),len(proxies),len(batch)),device='cuda')
                            with torch.no_grad():
                                for j,m in enumerate(proxies):
                                    activate(m)
                                    current,_=objective(m,adv,source,target,config)
                                    for i in range(len(proxies)):
                                        if i==j:continue
                                        candidate=project(adv-ng[i].sign()/255,clean)
                                        candidate_loss,_=objective(m,candidate,source,target,config)
                                        improvements[i,j]=current-candidate_loss
                                    park(m)
                            weights=adaptive_weights_from_surrogate_lookahead(improvements).detach()
                        fused=fuse_proxy_gradients(stack,mode=args.fusion,weights=weights)
                        if args.smoothing:
                            axis=torch.arange(-2,3,device='cuda').float();kernel=torch.exp(-axis.square()/2);kernel=kernel[:,None]*kernel[None,:];kernel=kernel/kernel.sum()
                            fused=F.conv2d(fused,kernel[None,None].expand(3,1,5,5),padding=2,groups=3)
                        pair=[]
                        for i in range(len(proxies)):
                            for j in range(i+1,len(proxies)):pair.append(F.cosine_similarity(ng[i].flatten(1),ng[j].flatten(1)).tolist())
                        denom=ng.flatten(2).norm(dim=2).mean(0).clamp_min(1e-12)
                        history.append(dict(step=step+1,losses=torch.stack(loss_values).cpu().tolist(),margins=torch.stack(margin_values).cpu().tolist(),gradient_l1=stack.abs().flatten(2).sum(2).cpu().tolist(),pairwise_cosines=pair,cancellation_ratio=(ng.mean(0).flatten(1).norm(dim=1)/denom).tolist(),weights=weights.cpu().tolist() if weights is not None else None))
                        momentum=momentum+fused
                        adv=project(adv-momentum.sign()/255,clean).detach().requires_grad_(True)
                        if step%10==0:print(f'[{args.split}] {name}/{tid} images {start+1}-{start+len(batch)} step {step+1}/{args.steps}',flush=True)
                    with torch.no_grad():
                        final_losses=[]
                        for m in proxies:
                            activate(m);val,_=objective(m,adv,source,target,config);final_losses.append(val);park(m)
                        value=torch.stack(final_losses).mean(0)
                        take=torch.ones_like(value,dtype=torch.bool) if best_loss is None else value<best_loss
                        best=adv.detach().clone() if best is None else torch.where(take[:,None,None,None],adv.detach(),best)
                        best_loss=value.clone() if best_loss is None else torch.minimum(value,best_loss)
                    traces.append(history)
                dest.mkdir(parents=True,exist_ok=True)
                linf_float,linf_png,_=_save_png_batch(dest,clean,best,16/255)
                if linf_png>16/255+1e-7:raise RuntimeError('PNG epsilon violation')
                paths=[dest/f'{i:02d}_adv.png' for i in range(len(batch))]
                clean_paths=[dest/f'{i:02d}_clean.png' for i in range(len(batch))]
                atomic_json(state,dict(status='attack_complete',identity=identity,config_hash=config_hash,config=config,method=args.method,models=args.models,split=args.split,transition_id=tid,source_label=source,target_label=target,image_ids=ids,steps=args.steps,seed=args.seed,restarts=args.restarts,adv_paths=[str(p.relative_to(OUT)) for p in paths],clean_paths=[str(p.relative_to(OUT)) for p in clean_paths],png_hashes={str(p.relative_to(OUT)):sha(p) for p in paths+clean_paths},linf_png=linf_png,linf_float=linf_float,trace=traces,runtime_seconds=time.perf_counter()-t0,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,forward_counts=[dict(model=m.name,visual_calls=m.forward_calls-b[0],backward_calls=m.backward_calls-b[1],image_forwards=m.image_forwards-b[2]) for m,b in zip(proxies,counts_before)],target_access=False))
    finally:
        for m in proxies:m.close()


if __name__=='__main__':main()
