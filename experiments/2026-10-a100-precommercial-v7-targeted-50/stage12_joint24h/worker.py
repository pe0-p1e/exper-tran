"""Native per-model layer Pure PP worker; independent image RNG and atomic receipts."""
from common import *
import argparse,signal,threading,traceback
STOP=False

def main():
 global STOP
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['attack','evaluate','evaluate30','calibrate','calibrate_resident','ila','ila_smoke']+list(LARGE_EVAL_MODES));ap.add_argument('--job',required=True);ap.add_argument('--attempt',default='0');a=ap.parse_args()
 check_lock();job=read(ROOT/'jobs'/(a.job+'.json'));c=job['condition'];legal(c['layers'],c['validator'])
 os.environ['V8_JOB_ID']=a.job;os.environ['V8_ATTEMPT']=a.attempt
 dest=ROOT/'attempts'/a.job/a.attempt;dest.mkdir(parents=True,exist_ok=True)
 def sig(*_):
  global STOP
  STOP=True
 signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
 ended=threading.Event();position={'phase':'starting'}
 def heartbeat():
  while not ended.is_set():atomic(dest/'heartbeat.json',dict(at=time.time(),pid=os.getpid(),position=position.copy()));ended.wait(15)
 threading.Thread(target=heartbeat,daemon=True).start()
 try:
  from host_memory import enforce
  cap=float(os.environ['V8_HOST_CAP_GIB'])
  containment=enforce(cap)
  atomic(dest/'host_memory_cap.json',containment)
  from supplement_registry import install,memory_cap
  install();memory_cap()
  import torch
  import strict_model_audit,vision_features
  # Load the authoritative auditor unwrapped: supplement.install redirects its
  # wrapper to the historical supplementary output root.
  import importlib.util
  audit_spec=importlib.util.spec_from_file_location('v8_parameter_audit',EXP.parent/'2026-10-a100-precommercial-v6/src/strict_model_audit.py')
  audit_module=importlib.util.module_from_spec(audit_spec);audit_spec.loader.exec_module(audit_module)
  original=audit_module.audit_loaded_model
  def audit(model,**kw):
   with (ROOT/'audit_writer.lock').open('a') as h:
    fcntl.flock(h,fcntl.LOCK_EX);kw['output_dir']=ROOT;return original(model,**kw)
  strict_model_audit.audit_loaded_model=audit;vision_features.audit_loaded_model=audit
  if a.mode in ['ila','ila_smoke']:
   from ila_selection import run
   run(c,position,lambda:STOP,smoke_only=a.mode=='ila_smoke')
  elif a.mode in LARGE_EVAL_MODES:
   from stage1_methods.selected_confirmation import validate
   validate(c,a.mode);evaluate(c,position,large_mode=a.mode)
  elif a.mode in ['evaluate','evaluate30']:evaluate(c,position,limit=30 if a.mode=='evaluate30' else None)
  else:attack(c,position,calibrate='resident_v2' if a.mode=='calibrate_resident' else a.mode=='calibrate')
  atomic(dest/'exit.json',dict(status='STOPPED' if STOP else 'COMPLETE',at=time.time()))
 except BaseException as e:
  text=traceback.format_exc();(dest/'traceback.txt').write_text(text)
  atomic(dest/'exit.json',dict(status='FAILED',kind='oom' if 'out of memory' in text.lower() else 'code',error=str(e),at=time.time()))
  raise
 finally:ended.set()

def model_fingerprint(m):
 import hashlib,torch
 family=m.spec['family'];v=m.model.model.visual if family=='qwen' else m.model.model.vision_tower if family in ['internvl','gemma4'] else m.model.vision_model if family in ['clip','siglip2'] else m.model
 dest=ROOT/'fingerprints'/(digest(m.name)[:12]+'.json')
 if dest.exists():return
 h=hashlib.sha256()
 for n,p in v.named_parameters():
  h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 cfg=getattr(m.model.config,'vision_config',m.model.config)
 atomic(dest,dict(model=m.name,revision=m.spec['revision'],llm_family=family,vision_weight_sha256=h.hexdigest(),vision_config=cfg.to_dict(),
  depth=m.depth,native_layers=list(range(m.depth)),dtype='bfloat16',training_objective='unknown unless checkpoint documentation verified',
  pooling='existing spatial-token masked mean L2; CLS/padding excluded by adapter'))

def bank(m):
 import numpy as np,torch
 layers=m.layers
 # Reference cache from an audited prior all-layer scan: verify receipt hash and pinned identity.
 old=DATA/'all_layer_selection/models'/m.name.replace('/','_')
 receipt=read(old/'reference_features.json')
 if receipt and sha(old/'reference_features.npz')==receipt['sha256']:
  lock=read(DATA/'all_layer_selection/protocol_lock.json')
  refs=[sha(DATA/'manifests/class_references'/f'class_references_{i:02d}.jsonl') for i in range(1,11)]
  expected=proxy_study_protocol.fingerprint([lock['model_revisions'][m.name],lock['base_sources'],refs])
  if lock['model_revisions'][m.name]!=REG[m.name] or receipt['identity']!=expected:raise RuntimeError('Historical reference identity mismatch')
  with np.load(old/'reference_features.npz') as d:m.bank={l:(torch.from_numpy(d[f'z_{l}']).cuda(),None) for l in layers}
  return
 refs=[DATA/'manifests/class_references'/f'class_references_{i:02d}.jsonl' for i in range(1,11)]
 all_layers=sorted(set(native(m.name,[.125,.25,.5,.625,.75,1.])+list(layers)));key=digest([REG[m.name],all_layers,[sha(p) for p in refs],'v8_pooled_refs'])
 path=ROOT/'cache/refs'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
 with (path.with_suffix('.lock')).open('a') as h:
  fcntl.flock(h,fcntl.LOCK_EX)
  if not path.exists():
   from v7_common import images
   before=m.layers;m.layers=all_layers;packed={l:[] for l in all_layers}
   with torch.inference_mode():
    for p in refs:
     rr=rows(p);assert len(rr)==48
     by={l:[] for l in all_layers}
     for start in range(0,48,2):
      fs=m.features(images(rr[start:start+2]))
      for l in all_layers:by[l].append(fs[l][0].float().cpu().numpy())
     for l in all_layers:packed[l].append(np.concatenate(by[l]))
   m.layers=before
   tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**{f'z_{l}':np.stack(v) for l,v in packed.items()});os.replace(tmp,path)
   atomic(path.with_suffix('.json'),dict(sha256=sha(path),model=m.name,revision=REG[m.name]['revision']))
  if sha(path)!=read(path.with_suffix('.json'))['sha256']:raise RuntimeError('Reference cache corruption')
  with np.load(path) as d:m.bank={l:(torch.from_numpy(d[f'z_{l}']).cuda(),None) for l in layers}

def load_models(c):
 import torch,gc
 from vision_features import VisionProxy
 result=[];full_gpu=os.environ.get('V8_WEIGHT_PLACEMENT')=='full_gpu'
 for name,layers in c['layers'].items():
  with (ROOT/'model_initialization.lock').open('a') as h:
   fcntl.flock(h,fcntl.LOCK_EX)
   m=VisionProxy(name,[(l+.5)/REG[name]['expected_vision_depth'] for l in layers],role='v8_attack_proxy')
   if m.layers!=layers:raise RuntimeError('Native visual block mismatch')
   from device_residency import install
   install(m)
   if full_gpu:m.model.to('cuda')
   m.to('cuda');model_fingerprint(m);bank(m)
   if not full_gpu:m.to('cpu')
   result.append(m);gc.collect();torch.cuda.empty_cache()
 from device_residency import parameter_gib,may_reside
 total=sum(parameter_gib(m._v8_visual) for m in result)
 import torch
 free,total_gpu=torch.cuda.mem_get_info()
 reservation=float(os.environ.get('V8_RESERVED_GIB','16'))
 resident=full_gpu or may_reside(total,reservation,free/2**30,max(8,total_gpu/2**30*.1),activation_margin_gib=1 if list(c['layers'])==[V] else 3)
 atomic(ROOT/'residency_plans'/(c['id']+'.json'),dict(strategy='visual_resident_v2',models=list(c['layers']),visual_parameter_GiB=total,
   reservation_GiB=reservation,resident=resident,free_GiB=free/2**30,language_parameters='BF16_GPU_unused_no_forward' if full_gpu else 'BF16_CPU_unused_no_forward',steps_unchanged=True))
 for m in result:
  m._v8resident=resident
  if resident:m.to('cuda')
  from strict_model_audit import audit_loaded_model
  audit_loaded_model(m.model,model_name=m.name,revision=m.spec['revision'],role='v8_visual_residency_proxy',repo_id=m.spec['repo_id'],output_dir=ROOT,vision_depth=m.depth)
 steady_cap=os.environ.get('V8_STEADY_CAP_GIB')
 if steady_cap:
  torch.cuda.empty_cache();torch.cuda.set_per_process_memory_fraction(float(steady_cap)/(total_gpu/2**30))
 if full_gpu:
  import ctypes
  gc.collect();ctypes.CDLL(None).malloc_trim(0)
  atomic(ROOT/'attempts'/os.environ['V8_JOB_ID']/os.environ['V8_ATTEMPT']/'placement_ready.json',dict(full_BF16_parameters_GPU=True,no_language_forward=True))
 return result

def tensor(r):
 from primary_ml_cka.artifacts.png import load_png_tensor
 if r.get('image_sha256') and sha(r['path'])!=r['image_sha256']:raise RuntimeError('Input image checksum changed')
 return load_png_tensor(Path(r['path']))[None].cuda()

def attack(c,position,calibrate=False):
 import torch,gc
 import numpy as np
 import torch.nn.functional as F
 from pp4_objective import objective
 from attack_optimizers import differentiable_resize_pad,normalize_pixel_gradients
 from primary_ml_cka.artifacts.png import save_png_tensor,assert_png_linf
 loading_start=time.time();models=load_models(c);loading_seconds=time.time()-loading_start;rr=cohort(c['split'])['images']
 if c['split']=='confirm':
  from stage1_methods.early_confirm import ordered_cases
  rr=ordered_cases(rr)
 config=dict(loss='pull_push',patch_weight=0,projector_weight=0,pull_weight=c['pull'],push_weight=c['push'])
 began=time.time();nforward=0;nback=0
 try:
  if calibrate:
   measurements=[]
   for m in models:
    m.to('cuda');config['layers']=m.layers
    for size in [1,2]:
     torch.cuda.reset_peak_memory_stats();x=torch.cat([tensor(r) for r in rr[:size]]).requires_grad_(True)
     loss,_=objective(m,x,rr[0]['source'],rr[0]['target'],config);g=torch.autograd.grad(loss.sum(),x)[0]
     if not torch.isfinite(g).all() or (g.abs().flatten(1).sum(1)==0).any():raise RuntimeError('Disconnected calibration gradient')
     measurements.append(dict(model=m.name,batch=size,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30))
     del x,loss,g;torch.cuda.empty_cache()
    if not m._v8resident:m.to('cpu')
    gc.collect();torch.cuda.empty_cache()
   atomic(ROOT/('calibration_resident_v2' if calibrate=='resident_v2' else 'calibration')/(c['id']+'.json'),dict(status='PASS',measurements=measurements,seconds=time.time()-began));return
  # Deliberately serial models within one image. Aggregate FP32 gradients only.
  for ix,r in enumerate(rr):
   if STOP:break
   dest=image_receipt(c,r['image_id']).parent
   if valid_receipt(dest/'state.json',c):continue
   position.update(image=ix,total=len(rr),image_id=r['image_id'],step=0)
   counts_before=[m.forward_calls for m in models];clean=tensor(r);rng=torch.Generator(device='cuda').manual_seed(seed(r['image_id'],'start'))
   adv=(clean+torch.empty_like(clean).uniform_(-16/255,16/255,generator=rng)).clamp(0,1).detach().requires_grad_(True)
   mom=torch.zeros_like(adv);di=torch.Generator().manual_seed(seed(r['image_id'],'DI'));traces=[];torch.cuda.reset_peak_memory_stats();t0=time.time()
   for step in range(c['steps']):
    ds=di.get_state();grads=[];losses=[]
    for m in models:
     if not m._v8resident:m.to('cuda')
     di.set_state(ds);config['layers']=m.layers
     view=differentiable_resize_pad(adv,generator=di)
     loss,_=objective(m,view,r['source'],r['target'],config)
     g=torch.autograd.grad(loss.sum(),adv)[0].detach().float();nback+=1;m.backward_calls+=1
     if not torch.isfinite(g).all() or not g.abs().sum():raise RuntimeError('Nonfinite/zero gradient')
     grads.append(g);losses.append(float(loss.detach().mean()));del view,loss
     if not m._v8resident:m.to('cpu');gc.collect();torch.cuda.empty_cache()
    ng=normalize_pixel_gradients(torch.stack(grads));fused=ng.mean(0)
    pair=[dict(left=models[i].name,right=models[j].name,cos=float(F.cosine_similarity(ng[i].flatten(1),ng[j].flatten(1))),negative_dot=bool((ng[i]*ng[j]).sum()<0),negative_pixel_fraction=float(((ng[i]*ng[j])<0).float().mean())) for i in range(len(models)) for j in range(i+1,len(models))]
    directed=[]
    if step==0 and len(models)>1:
     end_di=di.get_state()
     with torch.no_grad():
      for j,m in enumerate(models):
       if not m._v8resident:m.to('cuda')
       config['layers']=m.layers;di.set_state(ds)
       base,_=objective(m,differentiable_resize_pad(adv,generator=di),r['source'],r['target'],config)
       for i,g in enumerate(ng):
        candidate=torch.maximum(torch.minimum(adv-g.sign()/255,clean+16/255),clean-16/255).clamp(0,1)
        di.set_state(ds);after,_=objective(m,differentiable_resize_pad(candidate,generator=di),r['source'],r['target'],config)
        directed.append(dict(gradient_model=models[i].name,objective_model=m.name,gain=float((base-after).mean())))
       if not m._v8resident:m.to('cpu');gc.collect();torch.cuda.empty_cache()
     di.set_state(end_di)
    traces.append(dict(step=step+1,losses=losses,pairs=pair,directed_gains=directed,cancellation=float(fused.norm()/ng.flatten(2).norm(dim=2).mean().clamp_min(1e-12))))
    mom+=fused;adv=torch.maximum(torch.minimum(adv-mom.sign()/255,clean+16/255),clean-16/255).clamp(0,1).detach().requires_grad_(True)
    position['step']=step+1
   semantic=[];embedding_values={}
   with torch.no_grad():
    for m in models:
     if not m._v8resident:m.to('cuda')
     cf=m.features(clean);af=m.features(adv)
     for l in m.layers:
      z0,z1=cf[l][0].float(),af[l][0].float();refs=F.normalize(m.bank[l][0].float(),dim=-1)
      prefix=digest(m.name)[:12]+'_l'+str(l)
      embedding_values[prefix+'_clean']=z0.cpu().numpy()[0];embedding_values[prefix+'_adv']=z1.cpu().numpy()[0]
      centers=F.normalize(refs.mean(1),dim=-1);source,target=r['source']-1,r['target']-1
      pull=float((z1-z0)@centers[target]);push=float((z0-z1)@centers[source])
      semantic.append(dict(model=m.name,layer=l,delta_pull=pull,delta_push=push,DeltaR=pull+push,
       clean_margin=float(z0@(centers[target]-centers[source])),adv_margin=float(z1@(centers[target]-centers[source])),
       prototype_distance=float(1-centers[target]@centers[source]),source_variance=float((1-refs[source]@centers[source]).mean()),
       target_variance=float((1-refs[target]@centers[target]).mean())))
     if not m._v8resident:m.to('cpu');gc.collect();torch.cuda.empty_cache()
   dest.mkdir(parents=True,exist_ok=True)
   for name,x in [('clean',clean[0]),('adv',adv[0])]:
    p=dest/(name+'.tmp.png');save_png_tensor(x,p);os.replace(p,dest/(name+'.png'))
   bound=assert_png_linf(dest/'clean.png',dest/'adv.png')
   from paper_closeout.measure import array_save
   array_save(dest/'features.npz',**embedding_values)
   atomic(dest/'state.json',dict(status='COMPLETE',execution_strategy='visual_resident_v2',science=digest(scientific(c)),condition=c['id'],image_id=r['image_id'],transition=r['transition'],source=r['source'],target=r['target'],
    models=list(c['layers']),layers=c['layers'],steps=c['steps'],seed=42,target_access=False,linf_png=bound,
    clean=str(dest/'clean.png'),adv=str(dest/'adv.png'),png_hashes={str(dest/(x+'.png')):sha(dest/(x+'.png')) for x in ['clean','adv']},
    embedding_npz=str(dest/'features.npz'),embedding_sha256=sha(dest/'features.npz'),semantic=semantic,visual_forward_counts=[dict(model=m.name,calls=m.forward_calls-before) for m,before in zip(models,counts_before)],seconds=time.time()-t0,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,trace=traces,model_backward_calls=c['steps']*len(models)))
   print('frozen',c['id'],ix+1,'/',len(rr),flush=True)
  if attack_done(c):atomic(ROOT/'attack_complete'/(c['id']+'.json'),dict(status='COMPLETE',loading_seconds=loading_seconds,seconds=time.time()-began,backwards=nback,forward_counts=[dict(model=m.name,visual_forwards=m.forward_calls,offload_seconds=m.offload_seconds) for m in models]))
 finally:
  for m in models:m.close()

def evaluate(c,position,limit=None,large_mode=None):
 import torch,yaml
 from primary_ml_cka.models.backends.transformers_backend import load_processor
 from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
 from primary_ml_cka.models.common.loading import local_snapshot
 from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
 from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
 from strict_model_audit import audit_loaded_model
 from common import ROOT as output_root
 from v7_common import V6
 from common import REG
 from common import legal
 name=LARGE_EVAL_MODES[large_mode] if large_mode else c['validator']
 legal(c['layers'],name)
 began=time.time();torch.cuda.reset_peak_memory_stats()
 if limit==30:
  from stage1_methods.early_confirm import frozen,cases,path as early_path
  if not frozen(c,30):raise RuntimeError('Early evaluation requires frozen method and all30 valid PNGs')
  evaluation_cases=cases(30);result_path=early_path(c)
 else:
  if not attack_done(c):raise RuntimeError('Evaluation before complete freeze')
  evaluation_cases=cohort(c['split'])['images'];result_path=eval_path(c)
 if large_mode:
  from stage1_methods.selected_confirmation import large_path
  result_path=large_path(c,large_mode)
 from common import rows as unused
 # Canonical prompt and parser exactly reused from the existing evaluator.
 import importlib.util
 spec=importlib.util.spec_from_file_location('v5_prompt_common',EXP.parent/'2026-09-a100-final-v5/src/common.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 prompt=module.classification_prompt(yaml.safe_load((V6/'config/runner_template.yaml').read_text()))
 s=REG[name];MODEL_REVISIONS[s['repo_id']]=s['revision']
 with (ROOT/'model_initialization.lock').open('a') as h:
  fcntl.flock(h,fcntl.LOCK_EX);snapshot=local_snapshot(Path(os.environ['HF_HOME']),s['repo_id'],s['revision']);processor=load_processor(snapshot)
  model=load_target_for_generation(snapshot,torch.device('cuda'),precision='bf16')
  audit_loaded_model(model,model_name=name,revision=s['revision'],role='v8_frozen_large_target' if large_mode else 'v8_held_out_dev_validator',repo_id=s['repo_id'],output_dir=ROOT)
 gen=TransformersTargetGenerator(model,processor);result=[]
 for ix,r in enumerate(evaluation_cases):
  if STOP:break
  position.update(image=ix,phase='validation');receipt=read(image_receipt(c,r['image_id']));dest=(ROOT/'large_validation_images'/name if large_mode else ROOT/'validation_images')/c['id']/(digest(r['image_id'])[:18]+'.json')
  if dest.exists():
   value=read(dest)
   if value['signature']!=digest([s,prompt,receipt['png_hashes']]):raise RuntimeError('Changed evaluation signature')
  else:
   ck=digest([s,prompt,sha(receipt['clean'])]);cp=ROOT/'cache/clean_predictions'/(ck+'.json')
   with torch.inference_mode():
    clean=read(cp)
    if clean is None:
     x=gen.generate_labels([Path(receipt['clean'])],prompt)[0];clean=dict(label=x.parsed_label,raw=x.raw_output,status=x.parser_status);atomic(cp,clean)
    adv=gen.generate_labels([Path(receipt['adv'])],prompt)[0]
   value=dict(signature=digest([s,prompt,receipt['png_hashes']]),image_id=r['image_id'],transition=r['transition'],source=r['source'],target=r['target'],
    clean_label=clean['label'],clean_raw=clean['raw'],adv_label=adv.parsed_label,adv_raw=adv.raw_output,parser_status=adv.parser_status,
    target_hit=adv.parsed_label==r['target'],clean_correct=clean['label']==r['source'],off_target=adv.parsed_label not in [None,r['source'],r['target']])
   atomic(dest,value)
  if (ROOT/'paper_closeout/activation.json').exists():
   from paper_closeout.measure import per_image
   per_image(model,processor,name,c,r,receipt)
  result.append(value)
 N=len(evaluation_cases);complete=len(result)==N
 atomic(result_path,dict(complete=complete,science=digest(scientific(c)),condition=c['id'],model=name,revision=s['revision'],images=result,N=N,completed=len(result),hits=sum(r['target_hit'] for r in result),ASR=sum(r['target_hit'] for r in result)/N if complete else None,role='FROZEN_LARGE_TARGET_CONFIRMATION' if large_mode else 'CONFIRMATION' if c['split']=='confirm' else 'USED_FOR_SELECTION',final_target_access=bool(large_mode),seconds=time.time()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
 del gen,model,processor;torch.cuda.empty_cache()
if __name__=='__main__':main()
