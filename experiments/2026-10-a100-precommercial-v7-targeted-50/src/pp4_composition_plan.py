"""Paired ratio generalization across predeclared surrogate compositions."""
import json
from pp4_protocol import ROOT as OLD,PREVIOUS,OUT,EXP,VALIDATORS,condition,development_ids,attack_status,sha,atomic_json,fingerprint
from supplement_registry import registry

ROOT=OUT/'pull_push_composition_ablation'
C='CLIP ViT-L/14';Q='Qwen3.5-2B';I='InternVL3.5-2B-HF';G='Gemma 4 E2B-it';D='DINOv2-Large'
COMBINATIONS=[('S_CLIP',[C]),('S_Qwen',[Q]),('S_InternVL',[I]),('S_Gemma',[G]),('S_DINO',[D]),
    ('Qwen_InternVL',[Q,I]),('CLIP_Qwen_InternVL',[C,Q,I]),('CLIP_Qwen_Gemma',[C,Q,G]),
    ('CLIP_InternVL_Gemma',[C,I,G]),('CLIP_DINO_Qwen',[C,D,Q]),('CLIP_DINO_InternVL',[C,D,I]),
    ('CLIP_DINO_Gemma',[C,D,G]),('CLIP_DINO_Qwen_InternVL',[C,D,Q,I])]


def plan():
    rows=[dict(group=name,label='four_PP_DI',condition=condition(models,pull=pull))
          for name,models in COMBINATIONS for pull in [.5,.75]]
    from pp4_protocol import plan as old_plan
    return dict(ladder=rows,weights=[],smoke=old_plan()['smoke'])


def validation_path(c,model):return ROOT/'validation'/c['id']/(model.replace('/','_')+'.json')


def freeze():
    from commercial_matrix_pack import freeze as immutable
    ROOT.mkdir(parents=True,exist_ok=True);p=plan()
    for _,models in COMBINATIONS:
        if set(models)&set(VALIDATORS):raise RuntimeError('Validator overlap')
    sources=['pp4_composition_plan.py','pp4_composition_worker.py','pp4_protocol.py','pp4_objective.py','pp4_worker.py',
             'run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']
    immutable(ROOT/'protocol_lock.json',dict(protocol='pp4_composition_ratio_ablation_v1',plan=p,
        model_revisions=registry(),source_hashes={n:sha(EXP/'src'/n) for n in sources},
        development_ids=development_ids(),validators=VALIDATORS,ratios=[[.5,.5],[.75,.25]],
        layers=[.25,.5,.75,1.],patch_weight=0,DI=True,steps=50,epsilon='16/255',step_size='1/255',
        momentum=1,random_start=True,seed=42,pooling='spatial-token mean',fusion='equal per-image L1 normalized gradients',
        BF16=True,quantized=False,reference_count_per_class=48,
        scope='13 compositions x2 ratios; matched singles included; no final-target or commercial inference',
        hypothesis='Does the .5/.5 advantage depend on composition, proxy count or proxy paradigm?',
        inference='Exploratory same historical dev10;10 image clusters, not30 independent samples; no equivalence claim',
        no_target_feedback_for_selection=True,commercial_testing='AWAITING_USER_CONFIRMATION'))
    for row in p['ladder']:
        c=row['condition'];atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    if not (ROOT/'commercial_approval.json').exists():
        atomic_json(ROOT/'commercial_approval.json',dict(status='AWAITING_USER_CONFIRMATION',approved=False,proxy_models=None,settings=None))
    return p


def reuse_controls(p):
    for row in p['ladder']:
        c=row['condition'];paths=[OLD/'validation'/c['id']/(m.replace('/','_')+'.json') for m in VALIDATORS]
        if not all(f.exists() for f in paths):continue
        a=attack_status(c,deep=True)
        if not a['complete']:raise RuntimeError('Cached native result lacks matching frozen attacks')
        states=[dict(path=str(f),sha256=sha(f)) for f in sorted((OUT/'attacks/dev'/c['name']).glob('T*/batch_*/state.json'))]
        for model,path in zip(VALIDATORS,paths):
            d=json.loads(path.read_text())
            if d['denominator']!=10 or d['model']!=model:raise RuntimeError('Different cache model/sample count')
            if {r['image_id'] for r in d['images']}!={i for ids in development_ids().values() for i in ids}:raise RuntimeError('Different cache cohort')
            d.update(pp4_composition_reused_from=str(path),pp4_composition_reused_sha256=sha(path),new_inference=False)
            atomic_json(validation_path(c,model),d)
        atomic_json(ROOT/'reuse'/(c['id']+'.json'),dict(N=10,states=states,source=str(OLD),new_attack_generated=False))
