"""CPU-only compiler for follow-up jobs. Never owns GPU admission or runs inference."""
import argparse,itertools,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common as C

LARGE={'qwen':'Qwen3.5-27B','internvl':'InternVL3.5-14B-HF','gemma4':'Gemma 4 31B-it'}
PROTOCOL=HERE/'protocol.yaml'
OUT=C.ROOT/'proxy_selection_followup'

def snapshot(n):
 s=C.REG[n]
 return Path(C.os.environ.get('HF_HOME',str(C.MAIN/'.hf-cache')))/'hub'/('models--'+s['repo_id'].replace('/','--'))/'snapshots'/s['revision']

def layer_template():
 return dict(schema='proxy_layer_lock_v1',complete=False,source_method_lock_sha256=None,
  families={f:dict(validator=C.VALIDATORS[f],models={n:dict(revision=C.REG[n]['revision'],layers=None,layer_evidence_conditions=[],rule='MUST_BE_FROZEN_AFTER_LAYER_STUDY') for n in C.POOLS[f][:3]}) for f in LARGE})

def validate_layer_lock(lock):
 if lock.get('schema')!='proxy_layer_lock_v1' or not lock.get('complete'):raise ValueError('Layer study must finish and freeze its per-model layer policy first')
 if not lock.get('source_method_lock_sha256'):raise ValueError('Missing frozen layer-study provenance')
 source=Path(lock.get('source_method_lock_path',str(C.ROOT/'method_lock.json')))
 if not source.exists() or C.sha(source)!=lock['source_method_lock_sha256']:raise ValueError('Changed/missing source layer-study freeze')
 for fam,target in LARGE.items():
  d=lock['families'][fam]
  if d['validator']!=C.VALIDATORS[fam]:raise ValueError('Changed development validator')
  if set(d['models'])!=set(C.POOLS[fam][:3]):raise ValueError('Frozen candidate pool changed; preregister extensions separately')
  for n in C.POOLS[fam][:3]:
   value=d['models'][n]
   if value['revision']!=C.REG[n]['revision']:raise ValueError('Changed proxy revision')
   C.legal({n:value['layers']},target)
   if not value.get('layer_evidence_conditions'):raise ValueError('Unjustified/unfrozen model layers')
   for cid in value['layer_evidence_conditions']:
    c=C.read(C.ROOT/'conditions'/(cid+'.json'))
    if not c or c['family']!=fam or n not in c['layers'] or c['layers'][n]!=value['layers']:raise ValueError('Layer evidence does not match native taps/family')
    if not C.eval_done(c):raise ValueError('Layer evidence is incomplete')
 return True

def reusable(fam,layers,split):
 proposed=C.make(fam,layers,split=split)
 for p in sorted((C.ROOT/'conditions').glob('*.json')):
  old=C.read(p)
  if old['layers']!=layers:continue
  if {k:v for k,v in C.scientific(old).items() if k!='model_order'}!={k:v for k,v in C.scientific(proposed).items() if k!='model_order'}:continue
  # Preserve exact prior model order; validate saved images instead of trusting labels.
  if C.attack_done(old):return dict(condition=old,reuse_attack=True,reuse_development_evaluation=C.eval_done(old))
 return dict(condition=proposed,reuse_attack=False,reuse_development_evaluation=False)

def compile_tasks(layer_lock):
 validate_layer_lock(layer_lock);tasks=[]
 for fam in LARGE:
  names=C.POOLS[fam][:3];model_layers=layer_lock['families'][fam]['models']
  for size in range(1,4):
   for combo in itertools.combinations(names,size):
    ls={n:model_layers[n]['layers'] for n in combo};entry=reusable(fam,ls,'screen');c=entry['condition']
    tasks.append(dict(task_id='PROXY_'+C.digest([C.sha(PROTOCOL),c['id']])[:18],family=fam,subset=list(combo),K=size,
      condition=c,role='DEVELOPMENT_SELECTION_ONLY',attack_reuse=entry['reuse_attack'],development_eval_reuse=entry['reuse_development_evaluation'],
      enqueue_into='existing_stage12_admission_owner',requires_frozen_layers=True,large_target_visible_to_attack=False))
 return tasks

def freeze_methods(layer_lock_path,methods):
 layer_lock=C.read(layer_lock_path);validate_layer_lock(layer_lock)
 families={}
 for fam in LARGE:
  cc=methods[fam]
  if not 1<=len(cc)<=2:raise ValueError('Freeze at most primary and control')
  for condition in cc:
   if condition['family']!=fam or condition['split']!='confirm':raise ValueError('Wrong confirmation family/cohort')
   expected=C.make(fam,condition['layers'],split='confirm')
   if C.scientific(expected)!=C.scientific(condition):raise ValueError('Attack settings changed')
   C.legal(condition['layers'],LARGE[fam])
   for n,layers in condition['layers'].items():
    if layer_lock['families'][fam]['models'][n]['layers']!=layers:raise ValueError('Proxy comparison changed frozen layers')
  families[fam]=dict(conditions=[c['id'] for c in cc],condition_science={c['id']:C.digest(C.scientific(c)) for c in cc})
 path=OUT/'method_lock.json'
 C.freeze(path,dict(protocol='proxy_followup_frozen_methods_v1',policy_sha256=C.sha(PROTOCOL),layer_lock_sha256=C.sha(layer_lock_path),families=families,large_target_feedback_for_selection=False))
 return path

def frozen_large_packet(condition,method_lock_path,method_lock_sha256):
 """Compile evaluation of existing PNGs; changing target never regenerates an attack."""
 if C.sha(method_lock_path)!=method_lock_sha256:raise ValueError('Changed proxy-method freeze')
 fam=condition['family'];target=LARGE[fam]
 lock=C.read(method_lock_path)
 if lock.get('protocol')!='proxy_followup_frozen_methods_v1' or lock.get('large_target_feedback_for_selection') is not False:raise ValueError('Invalid proxy-method freeze')
 if lock['families'][fam]['condition_science'].get(condition['id'])!=C.digest(C.scientific(condition)):raise ValueError('Unfrozen candidate')
 C.legal(condition['layers'],target)
 if condition['split']!='confirm' or not C.attack_done(condition):raise ValueError('All50 confirmation PNGs must be frozen before large evaluation')
 images=C.cohort('confirm')['images']
 if len(images)!=50 or any(sum(r['transition']==t for r in images)!=10 for t in C.TIDS):raise ValueError('Wrong large confirmation cohort')
 rows=[]
 for r in images:
  path=C.image_receipt(condition,r['image_id']);receipt=C.read(path)
  if not C.valid_receipt(path,condition):raise ValueError('Invalid frozen PNG')
  if any(receipt[k]!=r[k] for k in ['image_id','source','target','transition']):raise ValueError('Frozen image belongs to another cohort')
  rows.append(dict(image_id=r['image_id'],transition=r['transition'],source=r['source'],target=r['target'],clean=receipt['clean'],adv=receipt['adv'],png_hashes=receipt['png_hashes']))
 pilot=[r['image_id'] for t in C.TIDS for r in [x for x in rows if x['transition']==t][:2]]
 return dict(task='postfreeze_large_target_evaluation',condition_id=condition['id'],attack_science_sha256=C.digest(C.scientific(condition)),
  method_lock_sha256=method_lock_sha256,evaluation_model=target,checkpoint=C.REG[target],precision='bf16',quantized=False,
  target_feedback_for_selection=False,historical_target_exposure=True,primary_metric='targeted_hits/all_GT_selected_images',
  confirmation_N=50,pilot_N=10,pilot_ids=pilot,images=rows,group_by_checkpoint=True,exclusive_large_model_load=True,
  dispatch_requires='existing scheduler large-target evaluation handler; no separate GPU owner')

def prepare(layer_lock_path=None):
 OUT.mkdir(parents=True,exist_ok=True)
 expected=layer_template();C.atomic(OUT/'layer_lock_template.json',expected)
 inventory={f:dict(model=n,checkpoint=C.REG[n],snapshot_cached=snapshot(n).exists(),actual_BF16_audit='REQUIRED_AT_LOAD_NOT_CLAIMED_BY_CACHE_CHECK') for f,n in LARGE.items()}
 candidate_names=[n for n in C.REG if n not in LARGE.values()]
 catalog={f:[dict(model=n,checkpoint=C.REG[n],snapshot_cached=snapshot(n).exists(),layer_lock_required=True,enabled=n in C.POOLS[f][:3]) for n in candidate_names if C.REG[n]['family']!=f] for f in LARGE}
 C.atomic(OUT/'candidate_catalog.json',dict(families=catalog,extension_enabled=False,max_active_total=8,max_active_VLM=2,max_active_vision=6,max_active_CLIP=4,max_active_DINO=2,extension_requires_frozen_native_layer_policy=True,no_cartesian_sweep=True))
 tasks=[];blockers=[]
 if layer_lock_path and Path(layer_lock_path).exists():
  lock=C.read(layer_lock_path);tasks=compile_tasks(lock)
  layer_sha=C.sha(layer_lock_path)
 else:
  blockers=['WAITING_FOR_FROZEN_PER_MODEL_LAYER_POLICY'];layer_sha=None
 C.atomic(OUT/'prepared_tasks.json',dict(protocol_sha256=C.sha(PROTOCOL),layer_lock_sha256=layer_sha,tasks=tasks,status='PREPARED_NOT_ENQUEUED' if tasks else 'WAITING_FOR_LAYER_FREEZE'))
 report=dict(status='PREPARED_NOT_STARTED',preparation_only=True,current_GPU_campaign_changed=False,core_static_candidates=21,
  tasks_materialized=len(tasks),reuse_attacks=sum(t['attack_reuse'] for t in tasks),large_targets=inventory,blockers=blockers,
  max_frozen_methods_for_large_evaluation=2,confirmation_model_image_rows=300,pilot_model_image_rows=60,
  confirmation_unique_requests_without_clean_reuse=600,confirmation_requests_with_exact_clean_reuse=450,
  formal150_sealed=True,large_scores_not_used_to_select_proxies=True,full_ablation_stage=3,
  dispatch_status='Compiler ready; live-scheduler handoff waits for frozen layer policy. No follow-up launched.')
 C.atomic(OUT/'preparation_status.json',report);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--layer-lock',type=Path);p.add_argument('--status',action='store_true');a=p.parse_args()
 if a.status:print(json.dumps(C.read(OUT/'preparation_status.json',{'status':'NOT_PREPARED'}),indent=2))
 else:print(json.dumps(prepare(a.layer_lock),indent=2))
if __name__=='__main__':main()
