#!/usr/bin/env python3
"""CPU-only evidence for integrated method construction. No inference/selection."""
import csv,json,os
from pathlib import Path

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def table(path,rows):
 if not rows:return
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');fields=list(rows[0])
 with tmp.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def generate(root):
 out=root/'reports';out.mkdir(parents=True,exist_ok=True);propagation=[];audit=[]
 for folder in sorted((root/'ila').glob('*')):
  p=read(folder/'protocol.json')
  # Smoke-only folders may exist before the formal diagnostic starts.
  sp=read(folder/'smoke/protocol.json');name=(p or sp or {}).get('model')
  if not name:continue
  selection=read(folder/'selection.json');done=read(folder/'complete.json',{})
  receipts=[read(x) for x in folder.glob('*/*/state.json')]
  audit.append(dict(model=name,smoke_pass=(folder/'smoke/smoke_pass.json').exists(),formal_complete=bool(done.get('complete')),
   native_tap_images_completed=len(receipts),latest_peak=selection.get('latest_peak') if selection else None,
   peak_status=selection.get('status') if selection else 'PENDING',selection_uses_validator_outputs=False,
   extra_projection_backward_calls=sum(x.get('backwards',0) for x in receipts),projection_seconds=sum(x.get('seconds',0) for x in receipts)))
  if selection:
   for tap,row in enumerate(selection['mean_ratios']):
    for observed,value in enumerate(row):propagation.append(dict(model=name,attack_tap=tap,observed_tap=observed,mean_relative_displacement=value,N=selection['N'],selected_latest_peak=tap==selection['latest_peak']))
 table(out/'ILA_native_propagation.csv',propagation);table(out/'ILA_execution_audit.csv',audit)
 geometry=read(root/'selection_geometry/paired_units.json',{})
 pairs=[]
 for r in geometry.get('pairs',[]):
  pairs.append(dict(left_model=r['left'][0],left_layer=r['left'][1],right_model=r['right'][0],right_layer=r['right'][1],
   CKA=r['CKA'],RSA=r['RSA'],gradient_cos=r['gradient_cos'],gradient_abs_cos=r['gradient_abs_cos'],negative_dot_fraction=r['negative_dot_fraction'],N=r['N'],source_only=True))
 table(out/'integrated_source_geometry.csv',pairs)
 trace=read(root/'selection/integrated_search_trace.json',{});rows=[]
 for family,events in trace.get('families',{}).items():
  for event in events:
   cid=event.get('condition')
   if not cid:continue
   ev=read(root/'validation'/(cid+'.json'),{});parent=read(root/'validation'/(event.get('parent','NONE')+'.json'),{})
   complete=ev.get('complete',False);paired=complete and parent.get('complete',False)
   gain=loss=None
   if paired:
    a={x['image_id']:x for x in ev['images']};b={x['image_id']:x for x in parent['images']}
    if a.keys()!=b.keys():raise ValueError('Unpaired conditional search results')
    gain=sum(x['target_hit'] and not b[i]['target_hit'] for i,x in a.items());loss=sum(not x['target_hit'] and b[i]['target_hit'] for i,x in a.items())
   rows.append(dict(family=family,size=event['size'],condition=cid,parent=event.get('parent'),added=json.dumps(event.get('added')),source_prior=event.get('source_prior'),
    status='COMPLETE' if complete else 'PENDING',hits=ev.get('hits') if complete else None,N=ev.get('N') if complete else None,
    parent_hits=parent.get('hits') if paired else None,gained_images=gain,lost_images=loss))
 table(out/'integrated_conditional_search.csv',rows)
 text='''# Integrated method construction — Stage1

Purpose: construct one source-guided, compute-aware model–layer selection method. Full component ablations are deferred to Stage3. Existing matched experiments remain reusable evidence, with their original configuration and model order.

The primary image attack is unchanged:50-step1:1 cosine prototype Pull/Push,48 references/class, strict BF16 parameters, DI and equal L1-normalized proxy gradients. No language-head loss or final-target feedback.

ILA is a separately costed source-only diagnostic. Starting from clean, maximize projection along a frozen50-step PP baseline feature displacement for10 sign steps, then inspect relative displacement propagation through every native block on saved/reloaded PNGs. Use raw valid spatial tokens. Record the deepest interior self-peak, with explicit NO_PEAK rather than silently substituting a maximum. ViT preprocessing, source PP baseline and hyperparameters make this an adaptation, not a published benchmark reproduction. Extra diagnostic images are never scored as primary PP attacks.

Candidate generation retains ILA as an optional source proposal and includes already measured effective singletons/pairs; revision2 starts from those held-out development results rather than forcing an ILA peak. At identical random-start/DI states, compute image-gradient cosine/negative-dot fractions and CKA/RSA on aligned480 reference images. A bounded source heuristic screens TWO additions per beam parent: one resident-model layer and one new-model unit when both exist. CKA novelty, gradient residual novelty, conflict and predicted marginal cost guide screening; none alone proves useful complementarity.

FDA-inspired conditional search generates actual50-step joint PP attacks for these candidates and evaluates only held-out development validators. Retain parent states so a harmful addition is not compulsory; preserve the failed fixed-ILA-start results and revision1 trace. A two-state quality/compute Pareto beam grows model–layer subsets. Conditional gains count gained AND lost images relative to the parent. This is a bounded surrogate shortlist adaptation, not exhaustive FDA(N) or trained FDA auxiliary predictors. A nonzero marginal-cost floor prevents treating extra resident taps as free. Concurrency affects timings.

Uniform and at most two integrated finalists advance to independent30-image exploration. The selected and uniform methods freeze before50-image confirmation and subscription GPT testing. A5pp development tolerance is not equivalence evidence. Failed improvement and missing results remain explicit. Commercial responses never retune the frozen selection.

Sources: ILA algorithm2,§3.1,§4.2 https://arxiv.org/html/1907.10823v3 ; FDA(N) supplementaryB https://proceedings.neurips.cc/paper/2020/file/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Supplemental.pdf . The geometric shortlist and joint Pareto rule are our experimental heuristics, not theorems or faithful implementations claimed from these papers.
'''
 (out/'METHOD_CONSTRUCTION.md').write_text(text)
 if propagation:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  for folder in sorted((root/'ila').glob('*')):
   s=read(folder/'selection.json')
   if not s:continue
   matrix=np.array([[np.nan if v is None else v for v in row] for row in s['mean_ratios']]);fig,ax=plt.subplots(figsize=(7,6))
   im=ax.imshow(matrix,aspect='auto');ax.set(xlabel='Observed native block',ylabel='ILA-refined native block',title=s['model']+' — source-only relative displacement')
   fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/('ILA_propagation_'+folder.name+'.png'),dpi=140);plt.close(fig)
 return dict(source_models=audit,geometry_pairs=len(pairs),conditional_candidates_observed=len(rows),full_ablation='DEFERRED_STAGE3')

if __name__=='__main__':print(json.dumps(generate(Path('/home/ubuntu/exper-tran/outputs/a100_precommercial_v8_joint_selection24h')),indent=2))
