"""Stage1 integrated ILA -> geometry prescreen -> FDA conditional joint search.

Small method development, not a full component-ablation campaign. Uniform and
matched completed singleton/pair results are reused with their actual identities.
"""
from common import *
import numpy as np

POLICY=dict(version='integrated_selection_v2_optional_ILA',shortlist_structure='one resident-model tap and one new-model unit when available',max_units=4,beam_width=2,shortlist_per_parent=2,
 source_pool='ILA optional peak, best two source gains and strongest evaluated singleton/model; at most4/model',
 shortlist='positive source gain * (1+CKA novelty) * (1+gradient residual novelty) * (1-negative-dot fraction) / predicted marginal seconds',
 no_peak='explicit source-gain fallback, never claim an ILA peak',
 search='FDA-inspired actual conditional attacks from measured strongest singleton and measured strongest pair; retain parents in cost Pareto beam; ILA not mandatory',
 final_tolerance_pp=5,ablation_stage=3,final_target_feedback=False)

def reuse_make(fam,layers,label,split='screen'):
 new=make(fam,layers,split=split,label=label)
 for p in sorted((ROOT/'conditions').glob('*.json')):
  old=read(p)
  # Keep the old model order and identity, rather than calling a permutation identical.
  if old['layers']==new['layers'] and {k:v for k,v in scientific(old).items() if k!='model_order'}=={k:v for k,v in scientific(new).items() if k!='model_order'}:return old
 return new

def unit_key(u):return digest(list(u))[:18]
def source_geometry():
 from ila_selection import selected,done,conditions
 if not all(done(c) for c in conditions()):raise RuntimeError('Incomplete source diagnostics')
 path=ROOT/'selection_geometry/paired_units.json';existing=read(path)
 identities={str(p):sha(p) for p in (ROOT/'selection_gradients').glob('*/*.json')}
 if existing and existing['inputs']==identities:return existing
 from geometry import reference
 from scipy.stats import rankdata
 models=sorted({n for pool in POOLS.values() for n in pool[:3]});units=[];gradients=[];quality=[];cost=[];grids=[]
 rr=cohort('screen')['images']
 for n in models:
  folder=ROOT/'selection_gradients'/digest(n)[:12];metas=[read(folder/(digest(r['image_id'])[:18]+'.json')) for r in rr]
  if any(m is None for m in metas):raise RuntimeError('Missing paired source gradients')
  layers=metas[0]['layers']
  for m,r in zip(metas,rr):
   if m['layers']!=layers or m['image_id']!=r['image_id'] or m['DI_state']!=seed(r['image_id'],'DI') or not m['source_only']:raise RuntimeError('Gradient input alignment mismatch')
  arrays=[]
  for meta,r in zip(metas,rr):
   p=folder/(digest(r['image_id'])[:18]+'.npz')
   if sha(p)!=meta['sha256']:raise RuntimeError('Gradient cache corruption')
   with np.load(p) as d:arrays.append(d['gradients'].reshape(len(layers),-1))
  for ix,l in enumerate(layers):
   units.append((n,l));gradients.append(np.stack([a[ix] for a in arrays]));quality.append(float(np.mean([m['quality'][ix] for m in metas])))
   cost.append(float(np.mean([m['seconds'][ix] for m in metas])));grids.append(reference(n,l))
 gram=[];ranks=[]
 for z in grids:
  centered=z-z.mean(0);k=centered@centered.T;den=np.linalg.norm(k)
  if den<1e-14:raise RuntimeError('Degenerate reference CKA')
  gram.append(k/den);z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
  rv=rankdata((1-z@z.T)[np.triu_indices(len(z),1)]);rv=rv-rv.mean();rn=np.linalg.norm(rv)
  if rn<1e-14:raise RuntimeError('Degenerate reference RSA')
  ranks.append(rv/rn)
 pairs=[]
 for i,u in enumerate(units):
  for j in range(i+1,len(units)):
   v=units[j];a,b=gradients[i],gradients[j];den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
   if (den<1e-20).any():raise RuntimeError('Degenerate normalized selection gradient')
   cos=(a*b).sum(1)/den
   pairs.append(dict(left=list(u),right=list(v),CKA=float((gram[i]*gram[j]).sum()),RSA=float(ranks[i]@ranks[j]),
     gradient_cos=float(cos.mean()),gradient_abs_cos=float(np.abs(cos).mean()),negative_dot_fraction=float((cos<0).mean()),N=len(rr)))
 d=dict(policy=POLICY,inputs=identities,source_only=True,units=[dict(model=n,layer=l,quality=q,seconds=t) for (n,l),q,t in zip(units,quality,cost)],pairs=pairs)
 atomic(path,d);return d

def eligible_pool(fam,g):
 from ila_selection import selected
 from selection import layer_conditions,top,atoms
 pool=[]
 for n in POOLS[fam][:3]:
  own=[u for u in g['units'] if u['model']==n];quality=sorted(own,key=lambda u:(u['quality'],-u['seconds'],u['layer']),reverse=True)
  measured=top([c for c in layer_conditions(fam) if n in c['layers']])
  peak=selected(n)['latest_peak'];ls=([peak] if peak is not None else [])+[u['layer'] for u in quality[:2]]
  if measured:ls.append(atoms(measured[0])[0][1])
  pool.extend((n,l) for l in dict.fromkeys(ls))
 legal({n:sorted(l for nn,l in pool if nn==n) for n in sorted({n for n,_ in pool})},VALIDATORS[fam])
 return pool

def pair_record(g,u,v):
 return next(r for r in g['pairs'] if {tuple(r['left']),tuple(r['right'])}=={tuple(u),tuple(v)})
def prior(g,u,current):
 own=next(x for x in g['units'] if (x['model'],x['layer'])==u)
 same=[x['quality'] for x in g['units'] if x['model']==u[0]]
 quality=max(0,own['quality'])/max(max(same),1e-12)
 records=[pair_record(g,u,v) for v in current]
 novelty=float(np.mean([1-r['CKA'] for r in records])) if records else 0
 residual=float(np.mean([1-r['gradient_abs_cos'] for r in records])) if records else 0
 conflicts=float(np.mean([r['negative_dot_fraction'] for r in records])) if records else 0
 resident=[next(x for x in g['units'] if (x['model'],x['layer'])==v)['seconds'] for v in current if v[0]==u[0]]
 # Resident taps have a measured full-pass cost but only an estimated marginal cost.
 # A nonzero floor prevents a free-tap ratio singularity. Actual search cost is measured later.
 marginal=max(own['seconds']-(max(resident) if resident else 0),own['seconds']*.25,1e-6)
 return quality*(1+novelty)*(1+residual)*(1-conflicts)/marginal

def shortlist(g,pool,current):
 ranked=sorted([u for u in pool if u not in current],key=lambda u:(prior(g,u,current),u),reverse=True)
 resident={u[0] for u in current}
 taps=[u for u in ranked if u[0] in resident];models=[u for u in ranked if u[0] not in resident]
 # Explicitly test layer expansion versus model expansion whenever both are legal.
 return [taps[0],models[0]] if taps and models else ranked[:POLICY['shortlist_per_parent']]

def pareto_beam(candidates,k=2):
 from selection import score,rank
 rr={c['id']:c for c in candidates if eval_done(c)};front=[]
 def vector(c):
  s=score(c);return (s['minimum'],s['hits'],-s['backwards'],-s['seconds'],-s['peak'])
 for c in rr.values():
  v=vector(c)
  if not any(all(a>=b for a,b in zip(vector(other),v)) and any(a>b for a,b in zip(vector(other),v)) for other in rr.values() if other['id']!=c['id']):front.append(c)
 if len(front)<=k:return sorted(front,key=rank,reverse=True)
 strong=max(front,key=rank);cheap=min(front,key=lambda c:(score(c)['backwards'],score(c)['seconds'],score(c)['peak'],-score(c)['hits'],c['id']))
 result=[strong]
 if cheap['id']!=strong['id']:result.append(cheap)
 for c in sorted(front,key=rank,reverse=True):
  if len(result)>=k:break
  if c['id'] not in {x['id'] for x in result}:result.append(c)
 return result

def method_screen(fam,g):
 from ila_selection import selected
 from selection import atoms,score,top,uniform,layer_conditions,original_pair_controls
 pool=eligible_pool(fam,g)
 def condition(units,label):
  ls={}
  for n,l in sorted(set(units)):ls.setdefault(n,[]).append(l)
  return reuse_make(fam,ls,label)
 measured=top(layer_conditions(fam))
 if measured:seed_c=measured[0]
 else:
  seeds=[u for u in pool if selected(u[0])['latest_peak']==u[1]] or pool
  seed_c=condition([max(seeds,key=lambda u:(prior(g,u,[]),u))],'integrated_source_seed_fallback')
 allc=[seed_c];trace=[dict(size=1,condition=seed_c['id'],units=[list(u) for u in atoms(seed_c)],selection='measured held-out development singleton; ILA optional',policy_version=POLICY['version'])]
 if not eval_done(seed_c):return [seed_c],allc,trace
 pairs=[c for c in original_pair_controls() if c['family']==fam and set(atoms(c))<=set(pool)]
 pair_seed=top(pairs)
 beam=[seed_c]+pair_seed
 for c in pair_seed:
  allc.append(c);trace.append(dict(size=2,condition=c['id'],selection='measured real joint pair development seed; not singleton success union'))
 for round_index in range(1,POLICY['max_units']):
  proposed={}
  for base in beam:
   current=atoms(base)
   if len(current)>=POLICY['max_units']:continue
   options=shortlist(g,pool,current)
   for u in options:
    c=condition(current+[u],f'integrated_FDA_joint_{len(current)+1}')
    proposed[c['id']]=c
    trace.append(dict(size=len(current)+1,round=round_index,parent=base['id'],condition=c['id'],added=list(u),source_prior=prior(g,u,current),shortlist_only=True))
  items=list(proposed.values());allc+=items
  missing=[c for c in items if not eval_done(c)]
  if missing:return missing,allc,trace
  updated=pareto_beam(beam+items,2)
  trace.append(dict(size=max(len(atoms(c)) for c in updated),round=round_index,retained_beam=[c['id'] for c in updated],selection='actual held-out dev conditional gain and compute Pareto, parents retained; no union of singleton hits'))
  stable={c['id'] for c in updated}=={c['id'] for c in beam};beam=updated
  if stable:break
 return [],allc,trace

def plan_next(state):
 from ila_selection import conditions as diagnostics,done,smoke_done
 from selection import proxy_conditions,layer_conditions,score,top,uniform
 locked=read(ROOT/'method_lock.json')
 if locked:
  # Runtime source amendments cannot reselect or rewrite an already frozen method.
  if locked['stage2_cohort_sha256']!=sha(ROOT/'manifests/confirm.json'):raise RuntimeError('Frozen confirmation cohort changed')
  confirm=[]
  for fam in VALIDATORS:
   for key,label in [('primary','selected'),('uniform','uniform')]:
    layers=locked['families'][fam][key];legal(layers,VALIDATORS[fam])
    confirm.append(reuse_make(fam,layers,label,split='confirm'))
  missing=list({c['id']:c for c in confirm if not eval_done(c)}.values())
  return ('INTEGRATED_CONFIRM50',missing) if missing else ('STAGE1_COMPLETE',[])
 # Reuse completed matched proxy/layer results; do not schedule historical full ablation.
 bases=diagnostics();missing=[c for c in bases if not attack_done(c)]
 if missing:return 'ILA_BASELINES',missing
 smokes=[c for c in bases if not smoke_done(c)]
 if smokes:return 'ILA_DIAGNOSTIC_SMOKE',smokes
 diag=[c for c in bases if not done(c)]
 if diag:return 'ILA_SOURCE_DIAGNOSTIC',diag
 g=source_geometry();allc=[];traces={};screen={}
 for fam in VALIDATORS:
  pending,items,trace=method_screen(fam,g);allc+=pending;traces[fam]=trace;screen[fam]=items
 atomic(ROOT/'selection/integrated_search_trace.json',dict(policy=POLICY,families=traces,stage3_full_ablation='DEFERRED_BY_USER_2026-10-21',old_results_reused=True))
 if allc:return 'INTEGRATED_METHOD_PILOT',list({c['id']:c for c in allc}.values())
 explore=[];finalists={}
 for fam in VALIDATORS:
  baseline=reuse_make(fam,uniform(POOLS[fam][:3]),'uniform')
  if not eval_done(baseline):return 'UNIFORM_MATCHED_BASELINE',[baseline]
  # Only integrated candidates plus uniform advance; legacy controls remain diagnostic evidence.
  frontier=pareto_beam(screen[fam],2);finalists[fam]=[baseline]+[c for c in frontier if c['layers']!=baseline['layers']]
  explore.extend(reuse_make(fam,c['layers'],c['label'],split='explore') for c in finalists[fam])
 missing=[c for c in explore if not eval_done(c)]
 if missing:return 'INTEGRATED_EXPLORE30',list({c['id']:c for c in missing}.values())
 metadata={};confirm=[]
 for fam in VALIDATORS:
  items=[reuse_make(fam,c['layers'],c['label'],split='explore') for c in finalists[fam]];baseline=items[0];bs=score(baseline)
  eligible=[c for c in items if score(c)['hits']/30>=bs['hits']/30-.05 and score(c)['minimum']>=bs['minimum']-1]
  chosen=min(eligible,key=lambda c:(score(c)['backwards'],score(c)['seconds'],-score(c)['hits'],c['id']))
  metadata[fam]=dict(primary=chosen['layers'],uniform=baseline['layers'],selection_evidence=chosen['id'],tolerance_pp=5,
   policy=POLICY,caveat='Small dev selection, not statistical equivalence or proof of universal semantics; no confirmation/GPT feedback')
  confirm.extend(reuse_make(fam,c['layers'],label,split='confirm') for c,label in [(chosen,'selected'),(baseline,'uniform')])
 freeze(ROOT/'method_lock.json',dict(protocol='V8_integrated_ILA_FDA_geometry_joint_selection',families=metadata,study_lock_sha256=sha(ROOT/'study_lock.json'),
  sources=source_hashes(),commercial_family_rule='Predeclared qwen-dev primary; GPT feedback forbidden',stage2_primary_family='qwen',stage2_methods=['selected','uniform'],
  stage2_cohort_sha256=sha(ROOT/'manifests/confirm.json'),stage2_model='gpt-6.1-sol',effort='low',approval='user immediate start plus integrated-method-first override 2026-10-21'))
 missing=[c for c in confirm if not eval_done(c)]
 if missing:return 'INTEGRATED_CONFIRM50',list({c['id']:c for c in missing}.values())
 return 'STAGE1_COMPLETE',[]
