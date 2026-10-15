#!/usr/bin/env python3
"""Post-hoc target/proxy geometry, variance, CKA/RSA and joint PCA/t-SNE."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'experiments/2026-09-a100-final-v5/src'),str(ROOT/'src')]

def slug(s): return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def cka(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);x-=x.mean(0,keepdims=True);y-=y.mean(0,keepdims=True)
 k=x@x.T;l=y@y.T
 den=np.linalg.norm(k)*np.linalg.norm(l)
 return float(np.sum(k*l)/den) if den else float('nan')
def rsa(x,y):
 x=F.normalize(torch.as_tensor(x).float(),dim=-1).cpu().numpy();y=F.normalize(torch.as_tensor(y).float(),dim=-1).cpu().numpy()
 ix=np.triu_indices(len(x),1);return float(spearmanr((1-x@x.T)[ix],(1-y@y.T)[ix]).statistic)
def variance(x):
 x=np.asarray(x,dtype=np.float64); center=x.mean(0); unit=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12);cu=center/(np.linalg.norm(center)+1e-12)
 disp=float(np.mean(1-unit@cu)); centered=x-center; trace=float(np.square(centered).sum()/(len(x)-1))
 gram=centered@centered.T/(len(x)-1); vals=np.linalg.eigvalsh(gram); vals=np.clip(vals,0,None); total=vals.sum(); p=vals[vals>1e-12]/total if total else np.array([]); erank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0
 return disp,trace,erank

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args();out=a.output_dir.resolve()
 if os.environ.get('PRIMARY_ML_CKA_STRICT_BF16')!='1':raise RuntimeError('strict BF16 environment guard is required')
 from common import transitions
 from primary_ml_cka.config.schema import AttackConfig
 from primary_ml_cka.data.manifests import read_manifest
 from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
 from primary_ml_cka.experiment.attack_generation import _cuda_images
 from primary_ml_cka.artifacts.png import load_png_tensor
 from primary_ml_cka.models.common.loading import local_snapshot
 from primary_ml_cka.models.proxies.registry import load_proxy
 from strict_model_audit import audit_loaded_model
 from sklearn.decomposition import PCA
 from sklearn.manifold import TSNE
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 matrix=yaml.safe_load((ROOT/'experiment_matrix.yaml').read_text())['multiple_proxy']
 lock=json.loads((ROOT/'model_revisions.json').read_text());raw=yaml.safe_load((ROOT/'config/runner_template.yaml').read_text())
 tmap={t.transition_id:t for t in transitions(raw)};tids=[k for k,v in tmap.items() if v.source<v.target]
 ensembles=[];seen=set()
 for block in ('block_1','block_2'):
  for row in matrix[block]:
   for proxies in row['ensembles']:
    key=(row['target'],tuple(proxies))
    if key in seen:continue
    seen.add(key);ensembles.append({'ensemble_id':f"E{len(ensembles)+1:02d}_"+'_'.join(proxies),'target':row['target'],'proxies':proxies})
 cell_states=[];model_paths={};target_by_name={}
 for e in ensembles:
  eslug=slug(e['ensemble_id']);base=out/'multiple_proxy/attacks'/eslug
  for tid in tids:
   state_path=base/f'{tid}.json';eval_path=out/'multiple_proxy/evaluations'/eslug/f'{tid}.json'
   if not state_path.is_file() or not eval_path.is_file():continue
   state=json.loads(state_path.read_text());ev=json.loads(eval_path.read_text())
   if state.get('status')!='attack_complete' or len(ev.get('images',[]))!=30:continue
   ims={x['image_id']:x for x in ev['images']}
   # image order must exactly match the attack state and frozen manifest
   if [x['image_id'] for x in ev['images']]!=state['image_ids']:raise RuntimeError(f"evaluation order mismatch {eslug}/{tid}")
   item={'ensemble':e,'state':state,'eval':ev,'tid':tid,'images':ims}
   cell_states.append(item);target_by_name[e['target']]=target_by_name.get(e['target'],[])+[item]
   # Aggregate attack paths from the frozen evaluation records. The attack
   # runner stores paths batch-wise under state['batches']; the evaluation
   # records carry the authoritative image-aligned clean/PNG paths.
   for name in [e['target'],*e['proxies']]:
    model_paths.setdefault(name,set())
    for image in ev['images']:
     model_paths[name].update([
      str((out/image['clean_path']).resolve()),
      str((out/image['adv_path']).resolve()),
     ])
 if not cell_states:raise RuntimeError('No complete attack + target evaluation cells found')

 emb_root=out/'analysis/embeddings/features';emb_root.mkdir(parents=True,exist_ok=True)
 features={}; ref_features={}; cache_meta={}
 for model_index,(name,paths) in enumerate(sorted(model_paths.items()),1):
  spec=lock[name];model_id=spec['repo_id'];revision=spec['revision'];MODEL_REVISIONS[model_id]=revision
  cache_path=emb_root/f'{slug(name)}_{revision[:12]}.npz';meta_path=cache_path.with_suffix('.json')
  local={};meta={}
  if cache_path.is_file() and meta_path.is_file():
   meta=json.loads(meta_path.read_text());z=np.load(cache_path)
   local={path:z[key] for path,key in meta.items() if key in z}
  missing=[x for x in sorted(paths) if x not in local]
  print(f"[representation] {model_index}/{len(model_paths)} {name}: {len(missing)} uncached images",flush=True)
  if missing:
   proxy=load_proxy(model_id,Path(os.environ['HF_HOME']),torch.device('cuda'),AttackConfig(generative_precision='bf16'))
   model=proxy.model;depth=None
   for cfg in (getattr(model.config,'vision_config',None),model.config):
    if cfg is not None:
     for field in ('depth','num_hidden_layers','num_layers'):
      value=getattr(cfg,field,None)
      if isinstance(value,int) and value>0:depth=value;break
    if depth:break
   if depth is None:raise RuntimeError(f'Cannot discover vision depth: {name}')
   audit_loaded_model(model,model_name=name,revision=revision,role='target_representation' if name in target_by_name else 'proxy_representation',repo_id=model_id,output_dir=out,vision_depth=depth)
   for s in range(0,len(missing),2):
    chunk=missing[s:s+2];imgs=torch.stack([load_png_tensor(Path(x)) for x in chunk]).cuda()
    with torch.inference_mode():emb=proxy.image_embeddings(imgs,representation_type='vision_encoder',layer=depth-1,pooling='mean').embeddings.float().cpu().numpy()
    for path,vec in zip(chunk,emb,strict=True):local[path]=vec
    del imgs,emb
   del proxy,model;gc.collect();torch.cuda.empty_cache()
  # write hash-based keys so paths containing punctuation are harmless
  keys={};arrays={}
  for path,value in local.items():
   key=hashlib.sha256(path.encode()).hexdigest();keys[path]=key;arrays[key]=np.asarray(value,dtype=np.float32)
  np.savez_compressed(cache_path,**arrays);meta_path.write_text(json.dumps(keys,indent=2)+'\n')
  features[name]=local

 # Reference embeddings are extracted using each target model, on the same 224px preprocessing path.
 ref_dir=out/'manifests/class_references';canonical=out/'canonical_images'
 for target in sorted(target_by_name):
  ref_features[target]={}
  proxy=load_proxy(lock[target]['repo_id'],Path(os.environ['HF_HOME']),torch.device('cuda'),AttackConfig(generative_precision='bf16'))
  model=proxy.model;rev=lock[target]['revision'];MODEL_REVISIONS[lock[target]['repo_id']]=rev
  depth=next((getattr(cfg,f,None) for cfg in (getattr(model.config,'vision_config',None),model.config) for f in ('depth','num_hidden_layers','num_layers') if isinstance(getattr(cfg,f,None),int)),None)
  if depth is None:raise RuntimeError(f'No vision depth for target {target}')
  audit_loaded_model(model,model_name=target,revision=rev,role='target_reference_geometry',repo_id=lock[target]['repo_id'],output_dir=out,vision_depth=depth)
  bank={}
  for label in range(1,11):
   records=tuple(read_manifest(ref_dir/f'class_references_{label:02d}.jsonl'));parts=[]
   with torch.inference_mode():
    for s in range(0,len(records),4):
     imgs=_cuda_images(canonical,records[s:s+4],224);parts.append(proxy.image_embeddings(imgs,representation_type='vision_encoder',layer=depth-1,pooling='mean').embeddings.float().cpu().numpy())
   bank[label]=np.concatenate(parts);ref_features[target][label]=bank[label]
  del proxy,model;gc.collect();torch.cuda.empty_cache()
  # persist descriptive class dispersion in native target embedding space
  for label,x in bank.items():
   d,tr,er=variance(x); cache_meta.setdefault('variance_rows',[]).append({'target':target,'class_label':label,'class_dispersion':d,'covariance_trace':tr,'effective_rank':er,'n_reference':len(x)})

 shift_rows=[];proxy_shift_rows=[];gradient_rows=[];gradient_fusion_rows=[];geometry_rows=[];alignment_rows=[];cell_rows=[]
 for item in cell_states:
  e=item['ensemble'];state=item['state'];ev=item['eval'];tid=item['tid'];src=int(state['source_label']);tgt=int(state['target_label']);target=e['target'];tfeat=features[target];tr=ref_features[target]
  src_center=F.normalize(torch.as_tensor(tr[src]).float(),dim=-1).mean(0);src_center=F.normalize(src_center,dim=0)
  tgt_center=F.normalize(torch.as_tensor(tr[tgt]).float(),dim=-1).mean(0);tgt_center=F.normalize(tgt_center,dim=0)
  proxy_c,proxy_a={},{}
  clean_ts=[];adv_ts=[]
  for iid,rawim in item['images'].items():
   cp=str((out/rawim['clean_path']).resolve());ap=str((out/rawim['adv_path']).resolve());zc=F.normalize(torch.as_tensor(tfeat[cp]).float(),dim=0);za=F.normalize(torch.as_tensor(tfeat[ap]).float(),dim=0)
   stc=float(zc@tgt_center);sta=float(za@tgt_center);ssc=float(zc@src_center);ssa=float(za@src_center)
   dp=sta-stc;dq=ssc-ssa;dr=dp+dq
   clean_ts.append(zc.numpy());adv_ts.append(za.numpy())
   shift_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'transition_id':tid,'image_id':iid,'delta_pull':dp,'delta_push':dq,'DeltaR':dr,'clean_source_similarity':ssc,'clean_target_similarity':stc,'adv_source_similarity':ssa,'adv_target_similarity':sta,'clean_margin_target_minus_source':stc-ssc,'adversarial_margin_target_minus_source':sta-ssa,'margin_change':(sta-ssa)-(stc-ssc),'prototype_distance_cosine':1-float(src_center@tgt_center),'target_hit':int(rawim['adv_label']==tgt),'clean_correct':int(rawim['clean_label']==src)})
   for n in e['proxies']:
    proxy_c.setdefault(n,[]).append(features[n][cp]);proxy_a.setdefault(n,[]).append(features[n][ap])
  means=[r for r in shift_rows if r['ensemble_id']==e['ensemble_id'] and r['transition_id']==tid]
  pair_vals=[]
  for n in e['proxies']:
   xx=np.concatenate([proxy_c[n],proxy_a[n]],axis=0);yy=np.concatenate([clean_ts,adv_ts],axis=0)
   ck=cka(xx,yy);rs=rsa(xx,yy);alignment_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'transition_id':tid,'proxy':n,'CKA':ck,'RSA':rs,'n_clean_adv_pairs':len(clean_ts)})
   pair_vals.append((ck,rs))
  for batch_index,batch in enumerate(state['batches']):
   for proxy_name,values in batch.get('proxy_representation_deltas',{}).items():
    for value in values:
     proxy_shift_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'transition_id':tid,'proxy':proxy_name,**value})
   for pair,summary in batch.get('gradient_cosine_summary',{}).items():
    gradient_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'transition_id':tid,'batch_index':batch_index,'batch_image_count':len(batch.get('image_ids',[])),'proxy_pair':pair,**summary})
   trace=batch.get('gradient_trace',[])
   fused=[float(step['fused_gradient_l2_mean']) for step in trace if step.get('fused_gradient_l2_mean') is not None]
   if fused:
    gradient_fusion_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'transition_id':tid,'batch_index':batch_index,'batch_image_count':len(batch.get('image_ids',[])),'trace_steps':len(fused),'mean_fused_gradient_l2':float(np.mean(fused)),'median_fused_gradient_l2':float(np.median(fused))})
  imgs=list(item['images'].values()); hits=sum(x['adv_label']==tgt for x in imgs); clean_ok=sum(x['clean_label']==src for x in imgs);clean_hits=sum(x['clean_label']==src and x['adv_label']==tgt for x in imgs)
  states=item['state']['batches'];proxy_all=sum(sum(b.get('all_proxy_success_mask',[])) for b in states);legacy_num=sum(sum(bool(m) and bool(h) for m,h in zip(b.get('all_proxy_success_mask',[]),b.get('target_hit_mask',[]))) for b in states)
  # attack states carry proxy hit masks; join the all-proxy mask to frozen image IDs
  amask={}
  for b in states:
   for i,iid in enumerate(b['image_ids']):amask[iid]=bool(b['all_proxy_success_mask'][i])
  legacy_num=sum(bool(amask.get(iid)) and x['adv_label']==tgt for iid,x in item['images'].items())
  all_masks={iid:bool(hit) for b in states for iid,hit in zip(b['image_ids'],b.get('all_proxy_success_mask',[]),strict=True)}
  any_masks={iid:any(bool(b.get('proxy_hit_masks',{}).get(n,[False]*len(b['image_ids']))[i]) for n in e['proxies']) for b in states for i,iid in enumerate(b['image_ids'])}
  majority_masks={iid:sum(bool(b.get('proxy_hit_masks',{}).get(n,[False]*len(b['image_ids']))[i]) for n in e['proxies'])>(len(e['proxies'])/2) for b in states for i,iid in enumerate(b['image_ids'])}
  eval_clean_parse=sum(x['clean_status']=='ok' for x in imgs);eval_adv_parse=sum(x['adv_status']=='ok' for x in imgs)
  cell_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'proxies':'+'.join(e['proxies']),'proxy_count':len(e['proxies']),'transition_id':tid,'source_label':src,'target_label':tgt,'N':len(imgs),'target_ASR_numerator':hits,'target_ASR_denominator':len(imgs),'target_ASR_all':hits/len(imgs) if imgs else None,'target_clean_correct_ASR_numerator':clean_hits,'target_clean_correct_ASR_denominator':clean_ok,'target_ASR_clean_correct':clean_hits/clean_ok if clean_ok else None,'target_clean_accuracy':clean_ok/len(imgs) if imgs else None,'target_clean_parser_success':eval_clean_parse,'target_adv_parser_success':eval_adv_parse,'proxy_any_hit_numerator':sum(any_masks.values()),'proxy_majority_hit_numerator':sum(majority_masks.values()),'proxy_all_hit_numerator':proxy_all,'proxy_all_hit_denominator':len(imgs),'legacy_conditional_TASR_numerator':legacy_num,'legacy_conditional_TASR_denominator':proxy_all,'legacy_conditional_TASR':legacy_num/proxy_all if proxy_all else None,'mean_DeltaR':float(np.mean([r['DeltaR'] for r in means])),'mean_delta_pull':float(np.mean([r['delta_pull'] for r in means])),'mean_delta_push':float(np.mean([r['delta_push'] for r in means])),'mean_target_clean_margin':float(np.mean([r['clean_margin_target_minus_source'] for r in means])),'mean_target_margin_change':float(np.mean([r['margin_change'] for r in means])),'prototype_distance':float(means[0]['prototype_distance_cosine']),'mean_CKA':float(np.nanmean([v[0] for v in pair_vals])),'mean_RSA':float(np.nanmean([v[1] for v in pair_vals])),'attack_runtime_seconds':sum(float(b.get('runtime_seconds',0)) for b in states),'peak_reserved_gib':max(float(b.get('peak_reserved_gib',0)) for b in states)})
  for label,values in ((src,tr[src]),(tgt,tr[tgt])):
   pass
  geometry_rows.append({'ensemble_id':e['ensemble_id'],'target':target,'transition_id':tid,'source_variance':variance(tr[src])[0],'target_variance':variance(tr[tgt])[0],'source_covariance_trace':variance(tr[src])[1],'target_covariance_trace':variance(tr[tgt])[1],'source_effective_rank':variance(tr[src])[2],'target_effective_rank':variance(tr[tgt])[2],'prototype_distance':cell_rows[-1]['prototype_distance'],'clean_margin':cell_rows[-1]['mean_target_clean_margin'],'adversarial_margin':cell_rows[-1]['mean_target_clean_margin']+cell_rows[-1]['mean_target_margin_change'],'margin_change':cell_rows[-1]['mean_target_margin_change']})
 # Joint single-fit PCA/t-SNE for each target model and class pair. All groups share the identical fitted coordinates.
 reduction_rows=[];reduction_root=out/'analysis';
 for target,cells in target_by_name.items():
  tr=tmap[cells[0]['tid']]
  for tid in tids:
   matching=[x for x in cells if x['tid']==tid]
   if not matching:continue
   src,tgt=tmap[tid].source,tmap[tid].target;groups=[]
   for label in (src,tgt):
    for i,z in enumerate(ref_features[target][label]):groups.append((f'class_{label}_reference',f'ref{label}_{i}',z))
   for i in range(30):
    rawim=matching[0]['images'][matching[0]['state']['image_ids'][i]];path=str((out/rawim['clean_path']).resolve());groups.append(('source_clean',rawim['image_id'],features[target][path]))
   for cell in matching:
    for iid,rawim in cell['images'].items():
     path=str((out/rawim['adv_path']).resolve());groups.append((f"adv_{cell['ensemble']['ensemble_id']}",iid,features[target][path]))
   X=np.asarray([F.normalize(torch.as_tensor(z).float(),dim=0).numpy() for _,_,z in groups],dtype=np.float32)
   coords={'pca':PCA(n_components=2,random_state=42).fit_transform(X)}
   perp=max(2,min(30,(len(X)-1)//3));coords['tsne']=TSNE(n_components=2,perplexity=perp,random_state=42,init='pca',learning_rate='auto',max_iter=1000).fit_transform(PCA(n_components=min(50,X.shape[1],len(X)-1),random_state=42).fit_transform(X))
   for kind,xy in coords.items():
    dirpath=reduction_root/kind;dirpath.mkdir(parents=True,exist_ok=True)
    fname=f'{slug(target)}_{tid}_{kind}'
    rows=[]
    for (group,iid,_),(x,y) in zip(groups,xy,strict=True):rows.append({'target':target,'transition_id':tid,'group':group,'image_id':iid,'x':float(x),'y':float(y),'fit_scope':'joint classes/references/clean/ensemble adversarials','fit_n':len(X),'random_state':42 if kind=='tsne' else None})
    write_csv(dirpath/f'{fname}.csv',rows)
    fig,ax=plt.subplots(figsize=(11,8));names=sorted(set(r['group'] for r in rows))
    cmap=plt.get_cmap('tab20',max(20,len(names)))
    for gi,g in enumerate(names):
     ix=[j for j,r in enumerate(rows) if r['group']==g];ax.scatter(xy[ix,0],xy[ix,1],s=15 if 'adv_' in g else 24,alpha=.62,label=g,color=cmap(gi%20))
    ax.set_title(f'{target} {tid} joint {kind.upper()} (N={len(X)})');ax.legend(fontsize=6,loc='best',ncol=2);fig.tight_layout();fig.savefig(dirpath/f'{fname}.png',dpi=160);plt.close(fig)
    reduction_rows.extend(rows)

 analysis_dir=out/'analysis'
 from collections import defaultdict
 pair_groups=defaultdict(list)
 for row in gradient_rows:pair_groups[(row['ensemble_id'],row['target'],row['proxy_pair'])].append(row)
 gradient_summary_rows=[]
 for (ensemble_id,target,pair),values in pair_groups.items():
  observations=sum(int(x['n']) for x in values)
  gradient_summary_rows.append({'ensemble_id':ensemble_id,'target':target,'proxy_pair':pair,'batch_summaries':len(values),'transitions':len({x['transition_id'] for x in values}),'gradient_observations':observations,'weighted_mean_cosine':sum(float(x['mean'])*int(x['n']) for x in values)/observations,'weighted_negative_fraction':sum(float(x['negative_fraction'])*int(x['n']) for x in values)/observations,'median_of_batch_medians':float(np.median([float(x['median']) for x in values]))})
 movement_groups=defaultdict(list)
 for row in proxy_shift_rows:movement_groups[(row['ensemble_id'],row['target'],row['proxy'])].append(row)
 movement_summary_rows=[]
 for (ensemble_id,target,proxy),values in movement_groups.items():
  movement_summary_rows.append({'ensemble_id':ensemble_id,'target':target,'proxy':proxy,'N':len(values),**{name:float(np.mean([float(x[name]) for x in values])) for name in ('delta_pull','delta_push','delta_R','clean_margin','adversarial_margin')}})
 write_csv(analysis_dir/'representation_shift/target_representation_shift.csv',shift_rows)
 write_csv(analysis_dir/'decision_geometry/decision_geometry.csv',geometry_rows)
 write_csv(analysis_dir/'variance/class_variance.csv',cache_meta.get('variance_rows',[]))
 write_csv(analysis_dir/'cka/cka_rsa.csv',alignment_rows)
 write_csv(analysis_dir/'proxy_representation_shift/proxy_representation_shift.csv',proxy_shift_rows)
 write_csv(analysis_dir/'gradient_compatibility/gradient_compatibility.csv',gradient_rows)
 write_csv(analysis_dir/'gradient_compatibility/gradient_compatibility_summary.csv',gradient_summary_rows)
 write_csv(analysis_dir/'gradient_compatibility/gradient_fusion_trace.csv',gradient_fusion_rows)
 write_csv(analysis_dir/'proxy_representation_shift/proxy_representation_shift_summary.csv',movement_summary_rows)
 write_csv(analysis_dir/'pca_tsne_coordinates.csv',reduction_rows)
 write_csv(out/'multiple_proxy/summaries/multiple_proxy_recovery.csv',cell_rows)
 # Per-image data joined with categorical cell metrics for auditable statistics.
 write_csv(analysis_dir/'per_image_metrics.csv',shift_rows)
 corr_cols=['target_ASR_all','mean_CKA','mean_RSA','mean_DeltaR','mean_delta_pull','mean_delta_push','mean_target_clean_margin','mean_target_margin_change','prototype_distance','source_variance','target_variance']
 geom={(x['ensemble_id'],x['transition_id']):x for x in geometry_rows}
 for row in cell_rows:
  g=geom[(row['ensemble_id'],row['transition_id'])];row['source_variance']=g['source_variance'];row['target_variance']=g['target_variance']
 matrix=[]
 for x in corr_cols:
  rr={'metric':x}
  for y in corr_cols:
   good=[(r[x],r[y]) for r in cell_rows if r.get(x) is not None and r.get(y) is not None and np.isfinite(float(r[x])) and np.isfinite(float(r[y]))]
   rr[y]=float(spearmanr(*zip(*good)).statistic) if len(good)>=3 else None
  matrix.append(rr)
 corr_dir=analysis_dir/'correlations';corr_dir.mkdir(parents=True,exist_ok=True);write_csv(corr_dir/'spearman_correlation_matrix.csv',matrix)
 fig,ax=plt.subplots(figsize=(10,8));vals=np.array([[r[y] if r[y] is not None else np.nan for y in corr_cols] for r in matrix]);im=ax.imshow(vals,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(corr_cols)),corr_cols,rotation=70,ha='right',fontsize=7);ax.set_yticks(range(len(corr_cols)),corr_cols,fontsize=7);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(corr_dir/'spearman_correlation_heatmap.png',dpi=180);plt.close(fig)
 # Preserve per-cell metrics with complete hit numerators/denominators and model/output status.
 write_csv(out/'reports/completion_matrix.csv',[{'ensemble_id':r['ensemble_id'],'target':r['target'],'transition_id':r['transition_id'],'status':'complete' if r['N']==30 else 'incomplete','N':r['N'],'ASR_numerator':r['target_ASR_numerator'],'ASR_denominator':r['target_ASR_denominator']} for r in cell_rows])
 print(f"[analysis] cells={len(cell_rows)} shift_rows={len(shift_rows)} CKA/RSA_rows={len(alignment_rows)} joint_reductions={len(reduction_rows)}",flush=True)
if __name__=='__main__':main()
