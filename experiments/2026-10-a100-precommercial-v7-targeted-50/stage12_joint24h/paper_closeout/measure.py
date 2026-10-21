"""Post-freeze target geometry on the already loaded, audited BF16 evaluator."""
import numpy as np
from types import SimpleNamespace
import common as C
OUT=C.ROOT/'paper_closeout'
REFERENCE_CACHE={}
STAT_CACHE={}

def array_save(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**values);C.os.replace(tmp,path)

def target_adapter(model,processor,name):
    from vision_features import VisionProxy
    m=VisionProxy.__new__(VisionProxy);m.name=name;m.spec=C.REG[name];m.model=model
    m.proxy=SimpleNamespace(processor=processor);m.layers=[C.REG[name]['expected_vision_depth']-1]
    return m

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def stats(z):
    z=unit(np.asarray(z,dtype=np.float64));center=unit(z.mean(0));s=np.linalg.svd(z-z.mean(0),compute_uv=False)**2/max(1,len(z)-1)
    p=s/s.sum() if s.sum()>0 else s
    return dict(dispersion=float((1-z@center).mean()),covariance_trace=float(s.sum()),
      effective_rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())) if s.sum()>0 else 0.)

def references(m):
    from worker import tensor
    name=m.name;l=m.layers[0];key=C.digest([C.REG[name],l,'target_deepest_spatial_mean',
      [C.sha(C.DATA/'manifests/class_references'/f'class_references_{i:02d}.jsonl') for i in range(1,11)]])
    path=OUT/'target_references'/(key+'.npz')
    if key in REFERENCE_CACHE:return REFERENCE_CACHE[key]
    if path.exists():
        if C.sha(path)!=C.read(path.with_suffix('.json'))['sha256']:raise RuntimeError('Target reference checksum mismatch')
        with np.load(path) as d:z=d['z']
        REFERENCE_CACHE[key]=z;return z
    import torch
    banks=[]
    with torch.inference_mode():
        for label in range(1,11):
            vectors=[]
            for r in C.rows(C.DATA/'manifests/class_references'/f'class_references_{label:02d}.jsonl'):
                x=tensor(dict(path=str(C.DATA/'canonical_images'/r['relative_path'])))
                vectors.append(m.features(x)[l][0].float().cpu().numpy()[0])
            banks.append(np.stack(vectors))
    z=np.stack(banks);array_save(path,z=z)
    C.atomic(path.with_suffix('.json'),dict(model=name,revision=C.REG[name]['revision'],layer=l,sha256=C.sha(path),
      per_class=[stats(v) for v in z],reference_N=48,definition='canonical224 native differentiable visual preprocessing; not language hidden states'))
    REFERENCE_CACHE[key]=z;return z

def per_image(model,processor,name,c,r,receipt):
    import torch
    from worker import tensor
    dest=OUT/'target_embeddings'/C.digest(name)[:12]/c['id']/(C.digest(r['image_id'])[:18]+'.npz')
    signature=C.digest([C.REG[name],receipt['png_hashes'],'native deepest spatial mean'])
    meta=C.read(dest.with_suffix('.json'))
    if meta:
        if meta['signature']!=signature or C.sha(dest)!=meta['sha256']:raise RuntimeError('Target feature identity changed')
        return meta
    m=target_adapter(model,processor,name);refs=references(m);l=m.layers[0]
    with torch.inference_mode():
        z0=m.features(tensor(dict(path=receipt['clean'])))[l][0].float().cpu().numpy()[0]
        z1=m.features(tensor(dict(path=receipt['adv'])))[l][0].float().cpu().numpy()[0]
    z0,z1=unit(z0),unit(z1);centers=unit(unit(refs).mean(1));s,t=r['source']-1,r['target']-1
    dp=float((z1-z0)@centers[t]);ds=float((z0-z1)@centers[s]);margin0=float(z0@(centers[t]-centers[s]));margin1=float(z1@(centers[t]-centers[s]))
    array_save(dest,clean=z0,adv=z1)
    for label in [s,t]:
        if (name,label) not in STAT_CACHE:STAT_CACHE[name,label]=stats(refs[label])
    meta=dict(signature=signature,sha256=C.sha(dest),image_id=r['image_id'],transition=r['transition'],condition=c['id'],model=name,layer=l,
      delta_pull=dp,delta_push=ds,DeltaR=dp+ds,clean_margin=margin0,adversarial_margin=margin1,margin_change=margin1-margin0,
      gap_closure=(margin1-margin0)/(-margin0) if margin0< -1e-8 else None,prototype_distance=float(1-centers[t]@centers[s]),
      source_stats=STAT_CACHE[name,s],target_stats=STAT_CACHE[name,t],target_feedback_to_attack=False)
    C.atomic(dest.with_suffix('.json'),meta);return meta
