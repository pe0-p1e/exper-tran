"""Bounded conditional greedy and width-2 beam. Final-target scores forbidden."""
from common import *
import itertools

def uniform(models):return {n:native(n,[.25,.5,.75,1.]) for n in models}
def score(c):
 d=read(eval_path(c))
 if not eval_done(c):return None
 per=[sum(x['target_hit'] for x in d['images'] if x['transition']==t) for t in TIDS]
 receipts=[read(image_receipt(c,x['image_id'])) for x in cohort(c['split'])['images']]
 return dict(hits=d['hits'],N=d['N'],minimum=min(per),directions=per,seconds=sum(x['seconds'] for x in receipts),
  backwards=sum(x['model_backward_calls'] for x in receipts),peak=max(x['peak_reserved_gib'] for x in receipts))
def rank(c):
 s=score(c);return (s['minimum'],s['hits'],-s['backwards'],-s['seconds'],c['id'])
def top(conditions,k=1):return sorted([c for c in conditions if eval_done(c)],key=rank,reverse=True)[:k]
def atoms(c):return sorted((n,l) for n,ls in c['layers'].items() for l in ls)
def from_atoms(fam,units,label):
 ls={}
 for n,l in sorted(set(units)):ls.setdefault(n,[]).append(l)
 return make(fam,ls,label=label)
def proxy_conditions(fam):
 v,a,b,d=POOLS[fam]
 comps=[('vision_single',[v]),('VLM_A_single',[a]),('VLM_B_single',[b]),('vision_VLM_A',[v,a]),('cross_VLM',[a,b]),
  ('uniform',[v,a,b]),('vision_DINO_VLM',[v,d,a]),('four_saturation',[v,d,a,b])]
 scale={'qwen':('InternVL3.5-4B-HF','InternVL3.5-8B-HF'), 'internvl':('Gemma 4 E4B-it','Gemma 4 26B-A4B-it'), 'gemma4':('Qwen3.5-4B','Qwen3.5-9B')}[fam]
 for suffix,n in [('same_family_scale',scale[0]),('same_family_large',scale[1])]:
  if n in REG:comps.append((suffix,[a if fam!='internvl' else b,n]))
 return [make(fam,uniform(ns),label='proxy_'+label) for label,ns in comps]
def layer_conditions(fam):
 # Actual native blocks of each backbone, not a forced common layer identity.
 return [make(fam,{n:[l]},label='layer_single') for n in POOLS[fam][:3] for l in native(n,[.125,.25,.5,.625,.75,1.])]
def plan_next(state):
 # User 2026-10-21: method construction now, full component ablation in Stage3.
 from combo_selection import plan_next as integrated
 return integrated(state)

def original_pair_controls():
 # Finish the previously authorized 45 real pair controls alongside method pilots.
 allc=[]
 for fam in VALIDATORS:
  lc=layer_conditions(fam)
  if not all(eval_done(c) for c in lc):continue
  pool=[]
  from geometry import complement
  for n in POOLS[fam][:3]:
   singles=[c for c in lc if n in c['layers']];best=atoms(top(singles)[0])[0][1]
   pool.extend([(n,best),(n,complement(n,best,[atoms(c)[0][1] for c in singles]))])
  allc.extend(from_atoms(fam,p,'joint_pair') for p in itertools.combinations(pool,2))
 return allc

def plan_deferred_full_ablation(state):
 """Return stages in order. Reconstruct the same adaptive queue from immutable dev scores."""
 for phase in ['proxy','layers','pairs','joint3','joint4','layer_controls','explore','confirm']:
  allc=[];family_meta={}
  for fam in VALIDATORS:
   pc=proxy_conditions(fam);lc=layer_conditions(fam)
   if phase=='proxy':items=pc
   elif phase=='layers':items=lc
   else:
    if not all(eval_done(c) for c in pc+lc):return 'PREREQUISITE_BLOCKED',[]
    # Two strongest singleton blocks per backbone form a six-unit pool, preserving models.
    pool=[]
    for n in POOLS[fam][:3]:
     from geometry import complement
     singles=[c for c in lc if n in c['layers']];best=atoms(top(singles)[0])[0][1]
     diverse=complement(n,best,[atoms(c)[0][1] for c in singles]);pool.extend([(n,best),(n,diverse)])
    pair=[from_atoms(fam,p,'joint_pair') for p in itertools.combinations(pool,2)]
    best_single=top(lc)[0];best_atom=atoms(best_single)[0]
    conditional=[c for c in pair if best_atom in atoms(c)]
    if phase=='pairs':items=pair
    else:
     if not all(eval_done(c) for c in pair):return 'PREREQUISITE_BLOCKED',[]
     greedy_pair=top(conditional)[0];beam_pairs=top(pair,2)
     triples={}
     for base in [greedy_pair]+beam_pairs:
      for u in pool:
       if u not in atoms(base):
        c=from_atoms(fam,atoms(base)+[u],'joint_three');triples[c['id']]=c
     trip=list(triples.values())
     if phase=='joint3':items=trip
     else:
      if not all(eval_done(c) for c in trip):return 'PREREQUISITE_BLOCKED',[]
      greedy3=top([c for c in trip if set(atoms(greedy_pair))<=set(atoms(c))])[0]
      beam3=top(trip,2);quads={}
      for base in [greedy3]+beam3:
       for u in pool:
        if u not in atoms(base):
         c=from_atoms(fam,atoms(base)+[u],'joint_four');quads[c['id']]=c
      quad=list(quads.values())
      if phase=='joint4':items=quad
      else:
       if not all(eval_done(c) for c in quad):return 'PREREQUISITE_BLOCKED',[]
       baseline=next(c for c in pc if c['label']=='proxy_uniform')
       # FDA-inspired native-layer search on the strongest singleton backbone.
       base=top(lc)[0];n=next(iter(base['layers']));first=base['layers'][n][0]
       available=[c['layers'][n][0] for c in lc if n in c['layers']]
       cp=[make(fam,{n:sorted([first,l])},label='FDA_greedy2_candidate') for l in available if l!=first]
       static=[make(fam,{n:native(n,[.5,1.])},label='uniform2'),
               make(fam,{n:sorted([c['layers'][n][0] for c in top([x for x in lc if n in x['layers']],2)])},label='independent_top2'),
               make(fam,{n:sorted([first,complement(n,first,available)])},label='low_CKA2')]
       unfinished=[c for c in cp+static if not eval_done(c)]
       if unfinished:
        if phase=='layer_controls':items=unfinished;allc+=items;continue
        return 'PREREQUISITE_BLOCKED',[]
       g2=top(cp)[0];ct=[make(fam,{n:sorted(g2['layers'][n]+[l])},label='FDA_greedy3_candidate') for l in available if l not in g2['layers'][n]]
       unfinished=[c for c in ct if not eval_done(c)]
       if unfinished:
        if phase=='layer_controls':allc+=unfinished;continue
        return 'PREREQUISITE_BLOCKED',[]
       g3=top(ct)[0];cq=[make(fam,{n:sorted(g3['layers'][n]+[l])},label='FDA_greedy4_candidate') for l in available if l not in g3['layers'][n]]
       if phase=='layer_controls':items=cq;allc+=items;continue
       if not all(eval_done(c) for c in cq):return 'PREREQUISITE_BLOCKED',[]
       candidates=[baseline]+top(pc+lc+pair+trip+quad+cp+static+ct+cq,2)
       # Additional named controls use actual joint attacks; top independent atoms is a product control.
       independent=from_atoms(fam,[atoms(c)[0] for c in top(lc,4)],'independent_top4')
       controls=[independent]
       if not eval_done(independent):return 'independent_control',[independent]
       candidates=top([*candidates,*controls],3)
       if baseline not in candidates:candidates=candidates[:2]+[baseline]
       if phase=='explore':items=[make(fam,c['layers'],split='explore',label=c['label']) for c in candidates]
       else:
        explore=[make(fam,c['layers'],split='explore',label=c['label']) for c in candidates]
        if not all(eval_done(c) for c in explore):return 'PREREQUISITE_BLOCKED',[]
        base30=next(c for c in explore if c['layers']==baseline['layers']);eligible=[c for c in explore if score(c)['hits']/30>=score(base30)['hits']/30-.05 and score(c)['minimum']>=score(base30)['minimum']-1]
        selected=min(eligible,key=lambda c:(score(c)['backwards'],score(c)['seconds'],-score(c)['hits']))
        family_meta[fam]=dict(primary=selected['layers'],uniform=baseline['layers'],selection_evidence=selected['id'],tolerance_pp=5,
         caveat='Development tolerance, not proof of equivalence; no confirmation or GPT feedback selects settings')
        items=[make(fam,ls,split='confirm',label=label) for label,ls in [('selected',selected['layers']),('uniform',baseline['layers'])]]
   allc+=items
  if phase=='confirm':
   # Freeze before any confirmation predictions; stage2 consumes this lock unchanged.
   freeze(ROOT/'method_lock.json',dict(protocol='V8_family_disjoint_joint_selection',families=family_meta,study_lock_sha256=sha(ROOT/'study_lock.json'),
    commercial_family_rule='Use qwen-family development selection as predeclared generic primary; GPT feedback forbidden',
    stage2_primary_family='qwen',stage2_methods=['selected','uniform'],stage2_cohort_sha256=sha(ROOT/'manifests/confirm.json'),
    sources=source_hashes(),stage2_model='gpt-6.1-sol',effort='low',approval='explicit user start authorization 2026-10-20'))
  missing=[c for c in allc if not eval_done(c)]
  if missing:return phase,missing
 return 'STAGE1_COMPLETE',[]
