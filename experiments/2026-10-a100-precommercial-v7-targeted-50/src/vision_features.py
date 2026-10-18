"""One visual forward for multiple true block outputs, without language scoring."""
import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.cluster import MiniBatchKMeans

from v7_common import OUT, REGISTRY, sha, fingerprint, records, images
from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.models.proxies.visual import qwen_visual_inputs, internvl_visual_inputs, gemma_visual_inputs
from strict_model_audit import audit_loaded_model
from primary_ml_cka.models.taps.pooling import masked_mean_l2


def layer_indices(depth, fractions):
    import math
    return sorted(set(max(0, min(depth-1, math.ceil(float(f)*depth)-1)) for f in fractions))


class VisionProxy:
    def __init__(self, name, fractions=(1.,), patch_k=8, role="proxy"):
        if os.environ.get('PRIMARY_ML_CKA_STRICT_BF16')!='1':
            raise RuntimeError('Strict BF16 must be enabled')
        spec=REGISTRY[name]
        MODEL_REVISIONS[spec['repo_id']]=spec['revision']
        self.name=name; self.spec=spec; self.patch_k=patch_k
        self.proxy=load_proxy(spec['repo_id'],Path(os.environ['HF_HOME']),torch.device('cuda'),AttackConfig(generative_precision='bf16'))
        self.model=self.proxy.model
        config=getattr(self.model.config,'vision_config',self.model.config)
        depth=getattr(config,'depth',None) or getattr(config,'num_hidden_layers',None)
        if depth!=spec['expected_vision_depth']: raise RuntimeError(f'Depth mismatch {name}: {depth}')
        self.depth=depth; self.layers=layer_indices(depth,fractions)
        audit_loaded_model(self.model,model_name=name,revision=spec['revision'],role=role,repo_id=spec['repo_id'],output_dir=OUT,vision_depth=depth)
        self.forward_calls=0; self.backward_calls=0; self.image_forwards=0
        self.offload_seconds=0.
        self.bank={}
        self.projected_bank=None
        family=spec['family']
        visual=self.model.model.visual if family=='qwen' else self.model.model.vision_tower if family in ('internvl','gemma4') else self.model.vision_model if family in ('clip','siglip2') else self.model
        def count_forward(_module,args,kwargs):
            self.forward_calls+=1
            if family=='qwen':
                grid=kwargs.get('grid_thw')
                self.image_forwards+=len(grid) if grid is not None else 1
            else:
                px=kwargs.get('pixel_values')
                if px is None:px=args[0]
                self.image_forwards+=len(px)
        self.counter_handle=visual.register_forward_pre_hook(count_forward,with_kwargs=True)

    def to(self, device):
        import time
        start=time.perf_counter()
        if self.spec['family']=='gemma4':
            # Only vision participates; every unused language parameter stays BF16 CPU.
            self.model.to('cpu')
            self.model.model.vision_tower.to(device)
        else: self.model.to(device)
        self.offload_seconds += time.perf_counter()-start
        return self

    def features(self,x):
        family=self.spec['family']
        # Small microbatches bound native-resolution activation memory.
        size=1 if family=='gemma4' else 2 if family in ('dinov2','siglip2') else 8
        packed={l:[] for l in self.layers}; masks={l:[] for l in self.layers}
        for chunk in x.split(size):
            def forward(inp):
                if family=='qwen':
                    v=qwen_visual_inputs(inp)
                    output=self.model.model.visual(v['pixel_values'],grid_thw=v['image_grid_thw'],output_hidden_states=True,return_dict=True)
                    tokens=[output.hidden_states[l+1].reshape(len(inp),-1,output.hidden_states[l+1].shape[-1]) for l in self.layers]
                    mask=torch.ones(tokens[0].shape[:2],device=inp.device,dtype=torch.bool)
                elif family=='internvl':
                    v=internvl_visual_inputs(inp)
                    output=self.model.model.vision_tower(pixel_values=v['pixel_values'],output_hidden_states=True,return_dict=True)
                    tokens=[output.hidden_states[l+1][:,1:] for l in self.layers]
                    mask=torch.ones(tokens[0].shape[:2],device=inp.device,dtype=torch.bool)
                elif family=='gemma4':
                    v=gemma_visual_inputs(self.proxy.processor,inp)
                    captures={}; handles=[]
                    def capture(l):
                        def hook(_m,_i,o): captures[l]=o[0] if isinstance(o,tuple) else o
                        return hook
                    for l in self.layers: handles.append(self.model.model.vision_tower.encoder.layers[l].register_forward_hook(capture(l)))
                    try:
                        self.model.model.vision_tower(pixel_values=v['pixel_values'],pixel_position_ids=v['image_position_ids'],return_dict=True)
                    finally:
                        for h in handles:h.remove()
                    tokens=[captures[l] for l in self.layers]
                    mask=~v['image_position_ids'].eq(-1).all(-1)
                elif family in ('clip','siglip2'):
                    pixels=self.proxy.image_preprocess(inp)
                    output=self.model.vision_model(pixel_values=pixels,output_hidden_states=True,return_dict=True)
                    tokens=[output.hidden_states[l+1][:,int(family=='clip'):] for l in self.layers]
                    mask=torch.ones(tokens[0].shape[:2],device=inp.device,dtype=torch.bool)
                elif family=='dinov2':
                    pixels=F.interpolate(inp.float(),size=(self.proxy.image_size,)*2,mode='bicubic',align_corners=False,antialias=True)
                    pixels=((pixels-self.proxy.mean.to(inp.device))/self.proxy.std.to(inp.device)).to(torch.bfloat16)
                    output=self.model(pixel_values=pixels,output_hidden_states=True,return_dict=True)
                    tokens=[output.hidden_states[l+1][:,1:] for l in self.layers]
                    mask=torch.ones(tokens[0].shape[:2],device=inp.device,dtype=torch.bool)
                else: raise ValueError(family)
                return (*tokens,mask)
            # Non-reentrant checkpoint preserves raw-image gradients and counts recomputation.
            result=checkpoint(forward,chunk,use_reentrant=False) if x.requires_grad and torch.is_grad_enabled() else forward(chunk)
            for i,l in enumerate(self.layers):packed[l].append(result[i]);masks[l].append(result[-1])
        result={}
        for l in self.layers:
            tok=torch.cat(packed[l]); mask=torch.cat(masks[l]); w=mask.float()[...,None]
            z=masked_mean_l2(tok,mask)
            result[l]=(z,tok,mask)
        return result

    def prepare_bank(self):
        manifests=[OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl' for i in range(1,11)]
        key=fingerprint(dict(model=self.spec,refs=[sha(p) for p in manifests],layers=self.layers,k=self.patch_k,version='v7_features_2_v6_pooling'))
        path=OUT/'cache/reference_features'/f'{key}.npz'
        if path.exists():
            with np.load(path) as d:self.bank={l:(torch.from_numpy(d[f'z_{l}']).cuda(),torch.from_numpy(d[f'p_{l}']).cuda()) for l in self.layers}
            return
        banks={l:[] for l in self.layers}; patches={l:[] for l in self.layers}
        with torch.inference_mode():
            for p in manifests:
                zs={l:[] for l in self.layers}; ts={l:[] for l in self.layers}
                for r in records(p):
                    feats=self.features(images([r]))
                    for l,(z,t,m) in feats.items():
                        zs[l].append(z.cpu())
                        real=t[0][m[0]].float()
                        ix=torch.linspace(0,len(real)-1,min(32,len(real)),device=real.device).long()
                        ts[l].append(F.normalize(real[ix],dim=-1).cpu())
                for l in self.layers:
                    banks[l].append(torch.cat(zs[l]).numpy())
                    pts=torch.cat(ts[l]).numpy()
                    km=MiniBatchKMeans(n_clusters=self.patch_k,random_state=42,n_init=3,batch_size=256).fit(pts)
                    patches[l].append(km.cluster_centers_)
        payload={}
        for l in self.layers:
            payload[f'z_{l}']=np.stack(banks[l]);payload[f'p_{l}']=np.stack(patches[l])
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**payload);temp.replace(path)
        self.bank={l:(torch.from_numpy(payload[f'z_{l}']).cuda(),torch.from_numpy(payload[f'p_{l}']).cuda()) for l in self.layers}

    def projected_features(self, x):
        if self.spec['family'] not in ('qwen','internvl'):
            raise ValueError('Projector objective is currently audited for Qwen/InternVL only')
        return self.proxy.image_embeddings(x,representation_type='legacy_projected',layer=-1,pooling='mean').embeddings.float()

    def prepare_projected_bank(self):
        if self.projected_bank is not None:return
        manifests=[OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl' for i in range(1,11)]
        key=fingerprint(dict(model=self.spec,refs=[sha(p) for p in manifests],tap='actual_post_merger_projected_tokens',version=1))
        path=OUT/'cache/projected_references'/f'{key}.npy'
        if path.exists():self.projected_bank=torch.from_numpy(np.load(path)).cuda();return
        banks=[]
        with torch.inference_mode():
            for manifest in manifests:
                banks.append(torch.cat([self.projected_features(images([r])).cpu() for r in records(manifest)]).numpy())
        array=np.stack(banks);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp.npy');np.save(temp,array);temp.replace(path)
        self.projected_bank=torch.from_numpy(array).cuda()

    def close(self):
        self.counter_handle.remove()
        self.model.to('cpu');self.bank.clear();self.projected_bank=None;del self.proxy;del self.model
        gc.collect();torch.cuda.empty_cache()
