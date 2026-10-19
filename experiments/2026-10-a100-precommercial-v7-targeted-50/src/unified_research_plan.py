"""Predeclared surrogate-only follow-up; excludes the commercial ten-image set."""
from copy import deepcopy
import json
from pathlib import Path
from proxy_study_protocol import OUT, ROOT, EXP, fingerprint, sha, atomic_json

RESEARCH=ROOT/'surrogate_followup'
VALIDATORS=['Qwen3.5-9B','InternVL3.5-8B-HF','Gemma 4 E4B-it']


def conditions():
    anchor=['CLIP ViT-L/14','Qwen3.5-2B','InternVL3.5-2B-HF']
    variants=[('U00',{},'equal',anchor),
              ('U01',{'loss':'pull_push'},'equal',anchor),
              ('U02',{'patch_weight':0.},'equal',anchor),
              ('U03',{'aggregation':'soft_reference'},'equal',anchor),
              ('U04',{'loss':'combined','margin_weight':.5},'equal',anchor),
              ('U05',{},'pcgrad',anchor),
              ('U06',{'adaptive_temperature':.005},'weighted',anchor),
              ('U07',{'eot_samples':4},'equal',anchor),
              ('U08',{},'equal',['CLIP ViT-L/14','DINOv2-Large','InternVL3.5-2B-HF']),
              ('U09',{},'equal',anchor+['SigLIP2-So400m-patch14-384']),
              ('U10',{},'equal',anchor+['DINOv2-Large'])]
    return [dict(id=i,models=m,method='A3',fusion=f,options=dict(patch_weight=.35,research_dev_offset=2,**o) if 'patch_weight' not in o else dict(research_dev_offset=2,**o),
                 dev_offset=2,images_per_direction=2,steps=50,seed=42,restarts=1,
                 budget_class='expanded_forward_budget' if o.get('eot_samples',1)>1 else 'primary50_steps') for i,o,f,m in variants]


def condition_name(c):
    return c['id']+'__A3__'+c['fusion']+'__'+'+'.join(n.replace('/','_').replace(' ','_') for n in c['models'])+'__DI__s42__i50__r1'


def cohort():
    by={}
    commercial=json.loads((OUT/'commercial_pilot10/private/manifest.json').read_text())
    forbidden={c['image_id'] for c in commercial['cases']}
    for tid in ['T01','T03','T05','T07','T09']:
        rows=[json.loads(s) for s in (OUT/'manifests/dev'/f'{tid}.jsonl').read_text().splitlines() if s.strip()]
        by[tid]=[r['image_id'] for r in rows[2:4]]
        if len(by[tid])!=2 or set(by[tid])&forbidden:raise RuntimeError('Follow-up dev images overlap commercial ten')
    return by


def lock():
    from commercial_matrix_pack import freeze
    RESEARCH.mkdir(parents=True,exist_ok=True)
    value=dict(protocol='surrogate_followup_v1',conditions=conditions(),validation_models=VALIDATORS,
          dev_ids=cohort(),target_feedback_for_selection=False,
          cohort_note='Disjoint from commercial10/final150; historically seen surrogate development images, not an untouched holdout',
          selection='Minimum held-out-surrogate model-direction ASR, then pooled ASR, then measured compute; no commercial or final-target results',
          bf16=True,epsilon='16/255',steps=50,spatial_mean=True,reference_count_per_class=48,
          source_hashes={n:sha(EXP/'src'/n) for n in ['unified_research_worker.py','run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']})
    freeze(RESEARCH/'method_lock.json',value)
    for c in conditions():atomic_json(RESEARCH/'configs'/(c['id']+'.json'),c['options'])
    return value


def status(c):
    root=OUT/'attacks/dev'/condition_name(c);ids=cohort();done=set()
    for p in root.glob('T*/batch_*/state.json'):
        d=json.loads(p.read_text())
        if d['status']!='attack_complete' or d['steps']!=50 or d['target_access']:raise RuntimeError('Invalid follow-up attack state')
        if d['models']!=c['models'] or d['config'].get('research_dev_offset')!=2:raise RuntimeError('Different follow-up method')
        for rel,h in d['png_hashes'].items():
            if sha(OUT/rel)!=h:raise RuntimeError('Changed follow-up PNG')
        done.update((d['transition_id'],iid) for iid in d['image_ids'])
    expected={(t,i) for t,rows in ids.items() for i in rows}
    if not done<=expected:raise RuntimeError('Wrong follow-up cohort')
    pilot=fingerprint(ids)[:12]
    val={name:OUT/'focused_optimization/native_validation'/pilot/condition_name(c)/(name.replace('/','_')+'.json') for name in VALIDATORS}
    emb={name:RESEARCH/'embeddings'/c['id']/(name.replace('/','_')+'.json') for name in set(c['models']+VALIDATORS)}
    return dict(attack_complete=done==expected,attack_images=len(done),
          validations={n:p.exists() and json.loads(p.read_text()).get('denominator')==10 for n,p in val.items()},
          embeddings={n:p.exists() for n,p in emb.items()})
