"""ILA projection/propagation adapted to raw valid visual tokens; selection-only.

No validator outputs are read. Main attacks remain 50-step Pure PP.
"""
from common import *
import numpy as np

DIAGNOSTIC = dict(version=1, refinement_steps=10, alpha=1/255, epsilon=16/255,
                  initialization='clean', DI=False, momentum=False,
                  feature='flattened raw valid spatial tokens, CLS/padding excluded',
                  loss='negative dot with frozen baseline feature displacement',
                  baseline='existing single-proxy four-layer 50-step Pure PP PNG',
                  source_only=True, denominator_floor=1e-8,
                  peak_rule='deepest attack tap with a strict interior local maximum at its own native block; mean paired ratio',
                  fallback='NO_PEAK; no silent argmax replacement')
BASE_FAMILIES={V:'qwen',I:'qwen',G:'qwen',Q:'internvl'}

def conditions():
 return [make(f,{n:native(n,[.25,.5,.75,1])},label='ILA_frozen_PP_baseline') for n,f in BASE_FAMILIES.items()]
def directory(n):return ROOT/'ila'/digest([REG[n],DIAGNOSTIC])[:18]
def selected(n):return read(directory(n)/'selection.json')
def valid(path):
 d=read(path)
 return bool(d and d.get('complete') and d.get('diagnostic')==digest(DIAGNOSTIC) and all(Path(p).exists() and sha(p)==h for p,h in d.get('png_hashes',{}).items()))
def done(c):
 n=next(iter(c['layers']));p=directory(n)/'complete.json';d=read(p)
 if not d or d.get('diagnostic')!=digest(DIAGNOSTIC):return False
 return all(Path(p).exists() and sha(p)==h for p,h in d.get('artifacts',{}).items()) and bool(d.get('artifacts'))
def smoke_done(c):
 n=next(iter(c['layers']));d=read(directory(n)/'smoke'/'smoke_pass.json')
 return bool(d and d.get('complete') and d.get('N')==2 and d.get('diagnostic')==digest(DIAGNOSTIC) and all(Path(p).exists() and sha(p)==h for p,h in d.get('artifacts',{}).items()))
def latest_peak(matrix):
 """Rows=attack taps, columns=all observed native blocks. Missing/flat is not a peak."""
 matrix=np.asarray(matrix,dtype=float);peaks=[]
 for l in range(1,len(matrix)-1):
  a,b,c=matrix[l,l-1:l+2]
  if np.isfinite([a,b,c]).all() and b>=a and b>=c and (b>a+1e-6 or b>c+1e-6):peaks.append(l)
 return peaks[-1] if peaks else None,peaks

def projection_loss(current,clean,baseline):
 return -((current.float()-clean.float())*(baseline.float()-clean.float())).sum()
def displacement(current,clean,baseline):
 a=np.asarray(current)-np.asarray(clean);b=np.asarray(baseline)-np.asarray(clean)
 den=float(np.linalg.norm(b));return float(np.linalg.norm(a)/den) if den>DIAGNOSTIC['denominator_floor'] else None

def raw(features,l):
 _,tokens,mask=features[l]
 if tokens.shape[0]!=1:raise RuntimeError('ILA diagnostics require individual images')
 return tokens[0][mask[0]].float().reshape(-1)

def run(c,position,stopping,smoke_only=False):
 import torch,gc
 import torch.nn.functional as F
 from worker import load_models,tensor,bank
 from primary_ml_cka.artifacts.png import save_png_tensor,assert_png_linf,load_png_tensor
 from pp4_objective import objective
 from attack_optimizers import differentiable_resize_pad
 n=next(iter(c['layers']));legal(c['layers'],c['validator'])
 if not attack_done(c):raise RuntimeError('ILA requires frozen baseline images')
 root=directory(n)/'smoke' if smoke_only else directory(n);root.mkdir(parents=True,exist_ok=True);freeze(root/'protocol.json',dict(model=n,revision=REG[n],diagnostic=DIAGNOSTIC,cohort_sha256=sha(ROOT/'manifests/screen.json'),baseline_science=digest(scientific(c)),smoke_only=smoke_only))
 models=load_models(c);m=models[0];m.to('cuda');depth=m.depth;all_layers=list(range(depth));rr=cohort('screen')['images'];collected=[];started=time.time()
 if smoke_only:rr=rr[:2]
 try:
  for ix,r in enumerate(rr):
   if stopping():return
   clean=tensor(r);baseline_receipt=read(image_receipt(c,r['image_id']))
   baseline=load_png_tensor(Path(baseline_receipt['adv']))[None].cuda();m.layers=all_layers
   with torch.no_grad():
    fc=m.features(clean);fb=m.features(baseline)
    zero={l:raw(fc,l).cpu().numpy() for l in all_layers};base={l:raw(fb,l).cpu().numpy() for l in all_layers}
   del fc,fb
   for tap in ([0,depth-1] if smoke_only else all_layers):
    if stopping():return
    position.update(phase='ILA_projection',image=ix,total=len(rr),native_tap=tap,step=0)
    dest=root/digest(r['image_id'])[:18]/str(tap);receipt=dest/'state.json'
    if valid(receipt):collected.append(read(receipt));continue
    dest.mkdir(parents=True,exist_ok=True);norm=float(np.linalg.norm(base[tap]-zero[tap]))
    if norm<=DIAGNOSTIC['denominator_floor']:
     d=dict(complete=True,diagnostic=digest(DIAGNOSTIC),image_id=r['image_id'],tap=tap,invalid='ZERO_BASELINE_DISPLACEMENT',ratios=[None]*depth,png_hashes={})
     atomic(receipt,d);collected.append(d);continue
    z0=torch.from_numpy(zero[tap]).cuda();zb=torch.from_numpy(base[tap]).cuda();x=clean.detach().clone().requires_grad_(True);m.layers=[tap]
    t0=time.time();before=m.forward_calls
    for step in range(DIAGNOSTIC['refinement_steps']):
     loss=projection_loss(raw(m.features(x),tap),z0,zb);g=torch.autograd.grad(loss,x)[0]
     if not torch.isfinite(g).all() or not g.abs().sum():raise RuntimeError('Disconnected/nonfinite ILA projection gradient')
     x=torch.maximum(torch.minimum(x-g.sign()/255,clean+16/255),clean-16/255).clamp(0,1).detach().requires_grad_(True);position['step']=step+1
    path=dest/'adv.png';tmp=dest/'adv.tmp.png';save_png_tensor(x[0],tmp);os.replace(tmp,path)
    bound=assert_png_linf(Path(baseline_receipt['clean']),path)
    frozen=load_png_tensor(path)[None].cuda();m.layers=all_layers
    with torch.no_grad():
     fs=m.features(frozen);ratios=[];angles=[]
     for l in all_layers:
      a=raw(fs,l).cpu().numpy()-zero[l];b=base[l]-zero[l];den=np.linalg.norm(b);an=np.linalg.norm(a)
      ratios.append(float(np.linalg.norm(a)/den) if den>DIAGNOSTIC['denominator_floor'] else None)
      angles.append(float(a@b/(an*den)) if an*den>1e-12 else None)
    del fs
    d=dict(complete=True,diagnostic=digest(DIAGNOSTIC),image_id=r['image_id'],transition=r['transition'],tap=tap,ratios=ratios,direction_cosines=angles,
      baseline_condition=c['id'],baseline_png_sha256=sha(baseline_receipt['adv']),linf_png=bound,target_access=False,
      png_hashes={str(path):sha(path)},seconds=time.time()-t0,visual_forwards=m.forward_calls-before,backwards=DIAGNOSTIC['refinement_steps'],peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    atomic(receipt,d);collected.append(d);gc.collect()
  if smoke_only:
   artifacts={p:h for d in collected for p,h in d.get('png_hashes',{}).items()}
   if len(collected)!=4 or any(d.get('invalid') for d in collected):raise RuntimeError('ILA smoke requires four finite projection/PNG checks')
   atomic(root/'smoke_pass.json',dict(complete=True,diagnostic=digest(DIAGNOSTIC),N=2,taps=[0,depth-1],artifacts=artifacts,formal_results=False));return
  matrix=np.full((depth,depth),np.nan)
  for tap in all_layers:
   records=[x for x in collected if x['tap']==tap]
   if len(records)!=len(rr):raise RuntimeError('Incomplete ILA native-tap sweep')
   for l in all_layers:
    values=[x['ratios'][l] for x in records]
    # No silently changing per-cell sample denominators.
    if all(v is not None for v in values):matrix[tap,l]=np.mean(values)
  best,peaks=latest_peak(matrix)
  valid_diag=[l for l in all_layers if np.isfinite(matrix[l,l])]
  amplitude=max(valid_diag,key=lambda l:(matrix[l,l],l)) if valid_diag else None
  chosen=list(dict.fromkeys(l for l in [best,amplitude] if l is not None))
  atomic(root/'selection.json',dict(model=n,revision=REG[n]['revision'],complete=True,latest_peak=best,peaks=peaks,amplitude_control=amplitude,
    candidate_layers=chosen,rule=DIAGNOSTIC['peak_rule'],status='PASS' if best is not None else 'NO_PEAK',source_only=True,validator_outputs_used=False,
    mean_ratios=[[float(v) if np.isfinite(v) else None for v in row] for row in matrix],N=len(rr),native_blocks=all_layers))
  # Actual input gradients at identical random-start/DI states for all source units.
  candidates=sorted(set(native(n,[.125,.25,.5,.625,.75,1])+chosen));m.layers=candidates;bank(m)
  config=dict(loss='pull_push',patch_weight=0,projector_weight=0,pull_weight=.5,push_weight=.5)
  gradroot=ROOT/'selection_gradients'/digest(n)[:12];gradroot.mkdir(parents=True,exist_ok=True)
  for ix,r in enumerate(rr):
   if stopping():return
   dest=gradroot/(digest(r['image_id'])[:18]+'.npz');meta=dest.with_suffix('.json')
   if meta.exists() and read(meta).get('layers')==candidates and dest.exists() and sha(dest)==read(meta)['sha256']:continue
   clean=tensor(r);rng=torch.Generator(device='cuda').manual_seed(seed(r['image_id'],'start'))
   x=(clean+torch.empty_like(clean).uniform_(-16/255,16/255,generator=rng)).clamp(0,1).detach().requires_grad_(True)
   grads=[];qualities=[];timings=[];config['layers']=[]
   for tap in candidates:
    position.update(phase='source_gradient',image=ix,native_tap=tap)
    torch.cuda.synchronize();gt0=time.time()
    m.layers=[tap];config['layers']=[tap];di=torch.Generator().manual_seed(seed(r['image_id'],'DI'))
    loss,_=objective(m,differentiable_resize_pad(x,generator=di),r['source'],r['target'],config)
    g=torch.autograd.grad(loss.sum(),x)[0].detach().float()
    if not torch.isfinite(g).all() or not g.abs().sum():raise RuntimeError('Invalid source-only selection gradient')
    with torch.no_grad():
     step=torch.maximum(torch.minimum(x-g.sign()/255,clean+16/255),clean-16/255).clamp(0,1)
     di=torch.Generator().manual_seed(seed(r['image_id'],'DI'));after,_=objective(m,differentiable_resize_pad(step,generator=di),r['source'],r['target'],config)
    grads.append((g/g.abs().sum()).cpu().numpy()[0]);qualities.append(float((loss-after).detach().mean()));torch.cuda.synchronize();timings.append(time.time()-gt0)
   tmp=dest.with_suffix('.tmp.npz');np.savez(tmp,gradients=np.stack(grads));os.replace(tmp,dest)
   atomic(meta,dict(model=n,layers=candidates,image_id=r['image_id'],quality=qualities,seconds=timings,sha256=sha(dest),source_only=True,DI_state=seed(r['image_id'],'DI'),random_start_seed=seed(r['image_id'],'start')))
  artifacts={str(p):sha(p) for p in [root/'selection.json',*gradroot.glob('*.npz'),*gradroot.glob('*.json')]}
  # Retain the complete per-tap audit as part of resumability validation.
  artifacts.update({str(p):sha(p) for p in root.glob('*/*/state.json')})
  for d in collected:artifacts.update(d.get('png_hashes',{}))
  atomic(root/'complete.json',dict(complete=True,diagnostic=digest(DIAGNOSTIC),model=n,artifacts=artifacts,seconds=time.time()-started,native_taps=depth,N=len(rr)))
 finally:
  m.close()
