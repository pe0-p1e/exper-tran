"""Bounded causal-component diagnostics; no full-matrix or commercial inference."""
import json
from pathlib import Path
from proxy_study_protocol import OUT, ROOT as PREVIOUS, EXP, sha, fingerprint, atomic_json, records, TIDS
from supplement_registry import registry

ROOT=OUT/'pull_push_four_layer'
VALIDATORS=['Qwen3.5-9B','InternVL3.5-8B-HF','Gemma 4 E4B-it']
RATIOS=[1.,.75,.5,.25,0.]
ANCHOR=['CLIP ViT-L/14','Qwen3.5-2B','InternVL3.5-2B-HF']
SECOND=['CLIP ViT-L/14','Qwen3.5-2B','Gemma 4 E2B-it']


def condition(models,pull=.75,layers=None,di=True,kind='pull_push',patch=0.,steps=50):
    opts=dict(loss=kind,layers=layers or [.25,.5,.75,1.],patch_weight=patch,
        aggregation='centroid',pull_weight=pull,push_weight=round(1-pull,8),
        projector_weight=0.,pp4_dev_offset=2,
        pp4_objective_sha256=sha(EXP/'src/pp4_objective.py'),
        pp4_worker_sha256=sha(EXP/'src/pp4_worker.py'))
    if steps!=50:opts['smoke_only']=True
    key=fingerprint([models,opts,di,steps])[:16]
    c=dict(id='PP4_'+key,models=list(models),method='PP4',fusion='equal',di=di,
           seed=42,steps=steps,restarts=1,split='dev',options=opts)
    c['name']=c['id']+'__PP4__equal__'+'+'.join(n.replace('/','_').replace(' ','_') for n in models)
    if di:c['name']+='__DI'
    c['name']+=f'__s42__i{steps}__r1'
    return c


def development_ids():
    ids={t:[r['image_id'] for r in records(OUT/'manifests/dev'/f'{t}.jsonl')[2:4]] for t in TIDS}
    forbidden={r['image_id'] for t in TIDS for r in records(OUT/'manifests/test'/f'{t}.jsonl')}
    forbidden|={c['image_id'] for c in json.loads((OUT/'commercial_pilot10/private/manifest.json').read_text())['cases']}
    refs={r['image_id'] for p in (OUT/'manifests/class_references').glob('*.jsonl') for r in records(p)}
    if any(len(v)!=2 or set(v)&(forbidden|refs) for v in ids.values()):raise RuntimeError('PP4 cohort leakage')
    return ids


def plan():
    rows=[]
    for group,models in [('A',ANCHOR),('B',SECOND)]:
        for label,c in [
            ('deep_PP_noDI',condition(models,layers=[1.],di=False)),
            ('four_PP_noDI',condition(models,di=False)),
            ('four_PP_DI',condition(models)),
            ('four_margin_DI',condition(models,kind='margin')),
            ('four_margin_patch_DI',condition(models,kind='margin',patch=.35))]:
            rows.append(dict(group=group,label=label,condition=c))
    weights=[dict(group='A',label=f'weight_{w:.2f}',condition=condition(ANCHOR,pull=w)) for w in RATIOS]
    return dict(ladder=rows,weights=weights,smoke=condition(ANCHOR,steps=2))


def freeze():
    from commercial_matrix_pack import freeze as immutable
    ROOT.mkdir(parents=True,exist_ok=True);p=plan()
    unique={r['condition']['id']:r['condition'] for r in p['ladder']+p['weights']}
    unique[p['smoke']['id']]=p['smoke']
    sources=['pp4_protocol.py','pp4_objective.py','pp4_worker.py','run_v7_attack.py',
             'representation_objectives.py','attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']
    immutable(ROOT/'protocol_lock.json',dict(protocol='pp4_bounded_diagnostics_v1',plan=p,
        development_ids=development_ids(),cohort_note='Same historical dev positions2:4 as U00/U02; not an untouched holdout',
        validators=VALIDATORS,parameter_precision='bf16',quantized=False,reference_count=48,
        epsilon='16/255',step_size='1/255',momentum=1,seed=42,steps=50,random_start=True,
        pooling='existing spatial-token mean',layer_reduction='uniform mean',
        interpretation='Paired exploratory dev comparisons;10 unique images evaluated on3 validators, not30 independent images',
        expansion='NO automatic full-matrix expansion, regardless of observed score',
        tuning='Surrogate-only minimum validator-direction ASR, then pooled ASR, then compute; no commercial or final-target feedback',
        commercial_testing='FORBIDDEN until user confirms exact proxy combination and settings',
        source_hashes={n:sha(EXP/'src'/n) for n in sources},model_revisions=registry()))
    for c in unique.values():atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    gate=ROOT/'commercial_approval.json'
    if not gate.exists():atomic_json(gate,dict(status='AWAITING_USER_CONFIRMATION',approved=False,proxy_models=None,settings=None))
    return p


def attack_status(c,deep=False):
    expected={(t,i) for t,ids in development_ids().items() for i in ids}
    if c['steps']!=50:expected={v for v in expected if v[0]=='T01'}
    found=set();seconds=0.;peak=0.
    for p in (OUT/'attacks/dev'/c['name']).glob('T*/batch_*/state.json'):
        d=json.loads(p.read_text())
        if d['config']!={**c['options'],'fusion':'equal','input_diversity':c['di'],'smoothing':False}:
            raise RuntimeError('Changed PP4 configuration')
        if d['target_access'] or d['steps']!=c['steps'] or d['seed']!=42 or d['models']!=c['models']:
            raise RuntimeError('Invalid attack provenance')
        if d['linf_png']>16/255+1e-7:raise RuntimeError('PNG bound violation')
        for iid in d['image_ids']:
            key=(d['transition_id'],iid)
            if key in found or key not in expected:raise RuntimeError('Wrong/duplicate PP4 image')
            found.add(key)
        if deep:
            for rel,h in d['png_hashes'].items():
                if sha(OUT/rel)!=h:raise RuntimeError('Changed PNG')
        seconds+=d['runtime_seconds'];peak=max(peak,d['peak_reserved_gib'])
    return dict(complete=found==expected,N=len(found),expected_N=len(expected),seconds=seconds,peak_reserved_gib=peak)


def validation_path(c,model):return ROOT/'validation'/c['id']/(model.replace('/','_')+'.json')
