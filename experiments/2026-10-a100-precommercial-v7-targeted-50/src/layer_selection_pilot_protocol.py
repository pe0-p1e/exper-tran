"""Predeclared small layer/count/model controls; no final-target tuning."""
from pp4_protocol import OUT,PREVIOUS,EXP,condition,development_ids,attack_status,sha,atomic_json
from supplement_registry import registry
ROOT=OUT/'layer_selection_pilot'
VALIDATORS=['Qwen3.5-9B']
MODELS=['InternVL3.5-2B-HF','Gemma 4 E2B-it','CLIP ViT-L/14']

def plan():
    from family_protocol import require_excluded_family
    require_excluded_family(MODELS,'qwen',registry())
    arms=[('single37p5',[.375]),('single62p5',[.625]),('single87p5',[.875]),
          ('single75',[.75]),('two62p5_87p5',[.625,.875]),('four_uniform',[.25,.5,.75,1.])]
    rows=[dict(group='E11',label=label,condition=condition(MODELS,layers=layers,pull=.5)) for label,layers in arms]
    for label,models in [('Intern_single',[MODELS[0]]),('Gemma_single',[MODELS[1]]),('CLIP_single',[MODELS[2]]),('Intern_Gemma_pair',MODELS[:2])]:
        rows.append(dict(group='model_controls',label=label,condition=condition(models,pull=.5)))
    return dict(ladder=rows,weights=[],smoke=condition(MODELS,steps=2,pull=.5))

def validation_path(c,model):return ROOT/'validation'/c['id']/(model.replace('/','_')+'.json')

def freeze():
    from commercial_matrix_pack import freeze as immutable
    p=plan()
    names=['layer_selection_pilot_protocol.py','layer_selection_pilot_worker.py','pp4_protocol.py','pp4_objective.py','pp4_worker.py',
        'run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']
    immutable(ROOT/'protocol_lock.json',dict(protocol='layer_model_selection_pilot_v1',plan=p,
        source_hashes={n:sha(EXP/'src'/n) for n in names},model_revisions=registry(),development_ids=development_ids(),
        validators=VALIDATORS,steps=50,epsilon='16/255',alpha='1/255',momentum=1,seed=42,random_start=True,
        BF16=True,quantization=False,reference_count=48,CLS=0,patch=0,pull=.5,push=.5,DI=True,
        gradient_fusion='equal independently L1-normalized',final_target_models_loaded=False,commercial_testing=False,
        selection='Predeclared non-quartile taps and matched model-count controls; exploratory source/surrogate validation only. '
            'Not a faithful ILA or FDA reproduction, nor a claim that diagnostic ranking predicts transfer. '
            'Historical dev10, not independent confirmation. Uniform fractions resolve to native block indices per model.'))
    for row in p['ladder']+[dict(condition=p['smoke'])]:
        c=row['condition'];atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    return p

def reuse_controls(p):
    from pathlib import Path
    import json
    from commercial_matrix_pack import freeze as immutable
    old=OUT/'v6_v7_multi_bridge'
    for row in p['ladder']:
        c=row['condition'];path=old/'validation'/c['id']/'Qwen3.5-9B.json'
        if row['label']!='four_uniform' or not path.exists():continue
        lock=json.loads((old/'protocol_lock.json').read_text())
        if lock['development_ids']!=development_ids():raise RuntimeError('Old baseline cohort differs')
        for name,h in lock['source_hashes'].items():
            if sha(EXP/'src'/name)!=h:raise RuntimeError('Old baseline sources changed')
        if not attack_status(c,deep=True)['complete']:raise RuntimeError('Old baseline attacks invalid')
        d=json.loads(path.read_text());d.update(reused_from=str(path),reused_sha256=sha(path),new_inference=False)
        immutable(validation_path(c,'Qwen3.5-9B'),d)
