from common import *
import csv

def table(path,rr):
 if not rr:return
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 os.replace(tmp,path)
def generate():
 from selection import score
 conditions=[];images=[];directions=[]
 for p in sorted((ROOT/'conditions').glob('*.json')):
  c=read(p);legal(c['layers'],c['validator']);s=score(c);ev=read(eval_path(c),{})
  done=sum(valid_receipt(image_receipt(c,r['image_id']),c) for r in cohort(c['split'])['images'])
  conditions.append(dict(id=c['id'],label=c['label'],family=c['family'],split=c['split'],validator=c['validator'],models=json.dumps(c['layers']),
   status='COMPLETE' if s else 'PARTIAL' if done else 'QUEUED',attack_images=done,N=len(cohort(c['split'])['images']),hits=s['hits'] if s else None,
   ASR=s['hits']/s['N'] if s else None,minimum_direction_hits=s['minimum'] if s else None,attack_seconds=s['seconds'] if s else None,
   backwards=s['backwards'] if s else None,peak_reserved_gib=s['peak'] if s else None))
  for x in ev.get('images',[]):images.append(dict(condition=c['id'],family=c['family'],split=c['split'],**x))
  for t in TIDS:
   rr=[x for x in ev.get('images',[]) if x['transition']==t];N=sum(r['transition']==t for r in cohort(c['split'])['images'])
   directions.append(dict(condition=c['id'],transition=t,hits=sum(x['target_hit'] for x in rr) if len(rr)==N else None,N=N,observed=len(rr)))
 table(ROOT/'reports/conditions.csv',conditions);table(ROOT/'reports/per_image.csv',images);table(ROOT/'reports/directions.csv',directions)
 audit=read(ROOT/'audits/model_dtype_audit.json',[]);failures=read(ROOT/'scheduler.json',{}).get('failures',[])
 lock=read(ROOT/'method_lock.json');stage2=read(ROOT/'stage2/summary.json',{})
 text=['# V8 Stage 1–2 handoff','',f"Status: {read(ROOT/'status.json',{}).get('state','PREPARING')}. Completed evaluations: {sum(x['status']=='COMPLETE' for x in conditions)}/{len(conditions)} planned so far.",
  '', 'Algorithm: strict BF16, native spatial-token mean features,48 references/class, .5 cosine target pull + .5 source push,50 steps,L∞16/255,DI, equal per-image L1-normalized proxy gradients. Layer losses are averaged inside each model before proxy normalization. No output-head, patch or margin loss.',
  '', 'Family exclusion applies to each actual validator, including Qwen9/InternVL8/GemmaE4. Final open-weight target150-image cohorts remain sealed. Historical10-image screening is development data; exploratory30 and confirmation50 are separately frozen. Stage2 reuses confirmation images only after selection lock.',
  '',f'BF16 audit load events: {len(audit)}. Failed audit events: {sum(x.get("status")!="pass" for x in audit)}.',
  '',f'Failures retained: {len(failures)}. See scheduler.json and attempts/*/exit.json; missing results are not zeros.',
  '', '## Frozen methods',json.dumps(lock,indent=2) if lock else 'PENDING: no selected method or superiority claim.',
  '', '## GPT confirmation',json.dumps(stage2,indent=2) if stage2 else 'PENDING: no calls/results claimed.',
  '', '## Interpretation and Stage3 discussion','No claim of universal visual head, proven causal mechanism or first multi-model/multi-layer attack. CKA/RSA and gradient conflict are diagnostics, not success surrogates. An exact implementation combination alone is not a novelty claim.',
  '', 'Must discuss: held-out categories and COCO original benchmark protocols; faithful vs adapted FDA/FOA/FTM/AdaEA/SEA baselines; multi-seed42/43/44; JPEG/resizing robustness; ten-class targeted ASR vs semantic change; nested surrogate selection and model family seen during development; independent final targets; selection cost and amortization.',
  '', 'All observed results: reports/conditions.csv,per_image.csv,directions.csv. Exact methods and hypotheses remain pending until measurements complete. No paper conclusions are fabricated.']
 (ROOT/'reports/PAPER_HANDOFF.md').parent.mkdir(parents=True,exist_ok=True);(ROOT/'reports/PAPER_HANDOFF.md').write_text('\n'.join(text)+'\n')
 if any(r['ASR'] is not None for r in conditions):
  try:plots(conditions)
  except ImportError:atomic(ROOT/'reports/plot_blocker.json',dict(reason='matplotlib missing'))
 mechanisms()
 return conditions

def plots(rr):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 done=[r for r in rr if r['ASR'] is not None]
 fig,ax=plt.subplots()
 for family in VALIDATORS:
  r=[x for x in done if x['family']==family and x['split']=='screen']
  ax.scatter([x['attack_seconds']/x['N'] for x in r],[x['ASR'] for x in r],label=family)
 ax.set(xlabel='Measured attack seconds/image (concurrency-dependent)',ylabel='Held-out surrogate targeted ASR');ax.legend();fig.tight_layout();fig.savefig(ROOT/'reports/ASR_compute.png');plt.close(fig)
 fig,ax=plt.subplots()
 for family in VALIDATORS:
  r=[x for x in done if x['family']==family and x['split']=='screen']
  ax.scatter([x['peak_reserved_gib'] for x in r],[x['ASR'] for x in r],label=family)
 ax.set(xlabel='Measured peak reserved GiB',ylabel='Held-out surrogate targeted ASR');ax.legend();fig.tight_layout();fig.savefig(ROOT/'reports/ASR_VRAM.png');plt.close(fig)
 lock=read(ROOT/'method_lock.json')
 if lock:
  import numpy as np
  entries=[(fam,method,values['primary' if method=='selected' else 'uniform']) for fam,values in lock['families'].items() for method in ['selected','uniform']]
  units=sorted({(n,l) for _,_,ls in entries for n,layers in ls.items() for l in layers})
  arr=np.array([[int(n in ls and l in ls[n]) for n,l in units] for _,_,ls in entries])
  fig,ax=plt.subplots(figsize=(max(8,len(units)*.25),4));ax.imshow(arr,aspect='auto',vmin=0,vmax=1)
  ax.set_yticks(range(len(entries)),[f'{f}:{m}' for f,m,_ in entries]);ax.set_xticks(range(len(units)),[f'{n}:block{l}' for n,l in units],rotation=90,fontsize=6)
  fig.tight_layout();fig.savefig(ROOT/'reports/native_model_layer_selection.png');plt.close(fig)
  fig,ax=plt.subplots()
  for entry in done:
   if entry['split']!='confirm':continue
   ev=read(ROOT/'validation'/(entry['id']+'.json'));ys=[sum(i['target_hit'] for i in ev['images'] if i['transition']==t)/10 for t in TIDS]
   ax.plot(TIDS,ys,marker='o',label=entry['family']+':'+entry['label'])
  ax.set(ylabel='Confirmation targeted hits /10',ylim=(0,1.05));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'reports/per_direction_confirmation.png');plt.close(fig)

def mechanisms():
 semantic=[];gradients=[];cka=[]
 for p in (ROOT/'conditions').glob('*.json'):
  c=read(p)
  for r in cohort(c['split'])['images']:
   path=image_receipt(c,r['image_id'])
   if not valid_receipt(path,c):continue
   d=read(path)
   for x in d.get('semantic',[]):semantic.append(dict(condition=c['id'],image_id=r['image_id'],transition=r['transition'],**x))
   for step in d.get('trace',[]):
    for x in step.get('pairs',[]):gradients.append(dict(condition=c['id'],image_id=r['image_id'],step=step['step'],cancellation=step['cancellation'],**x))
 for p in (ROOT/'geometry').glob('*.json'):cka+=read(p).get('pairs',[])
 table(ROOT/'reports/representation_shift_variance.csv',semantic);table(ROOT/'reports/gradient_compatibility.csv',gradients);table(ROOT/'reports/CKA_RSA.csv',cka)
