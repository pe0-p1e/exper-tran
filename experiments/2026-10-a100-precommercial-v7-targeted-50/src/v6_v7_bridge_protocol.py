"""Exact V6 E11 multi-proxy composition, small held-out cross-family bridge."""
from pp4_protocol import OUT,PREVIOUS,EXP,condition,development_ids,attack_status,sha,atomic_json
from supplement_registry import registry
ROOT=OUT/'v6_v7_multi_bridge'
VALIDATORS=['Qwen3.5-9B']
MODELS=['InternVL3.5-2B-HF','Gemma 4 E2B-it','CLIP ViT-L/14']

def plan():
    from family_protocol import require_excluded_family
    require_excluded_family(MODELS,'qwen',registry())
    arms=[('old_deep_noDI',[1.],False,.75),('single75_noDI',[.75],False,.75),
        ('deep_DI',[1.],True,.75),('single75_DI',[.75],True,.75),
        ('four_noDI',[.25,.5,.75,1.],False,.75),('four_DI',[.25,.5,.75,1.],True,.75),
        ('four_DI_balanced',[.25,.5,.75,1.],True,.5)]
    rows=[dict(group='E11',label=label,condition=condition(MODELS,layers=layers,di=di,pull=pull)) for label,layers,di,pull in arms]
    from pp4_protocol import plan as previous
    return dict(ladder=rows,weights=[],smoke=previous()['smoke'])

def validation_path(c,model):return ROOT/'validation'/c['id']/(model.replace('/','_')+'.json')

def freeze():
    from commercial_matrix_pack import freeze as immutable
    p=plan()
    names=['v6_v7_bridge_protocol.py','v6_v7_bridge_worker.py','pp4_protocol.py','pp4_objective.py','pp4_worker.py',
        'run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']
    immutable(ROOT/'protocol_lock.json',dict(protocol='V6_E11_exact_composition_surrogate_bridge_v1',plan=p,
        source_hashes={n:sha(EXP/'src'/n) for n in names},model_revisions=registry(),development_ids=development_ids(),
        validators=VALIDATORS,steps=50,epsilon='16/255',alpha='1/255',momentum=1,seed=42,random_start=True,
        BF16=True,quantization=False,reference_count=48,CLS=0,patch=0,gradient_fusion='equal independently L1-normalized',
        final_target_models_loaded=False,commercial_testing=False,
        caveat='Historical dev10; one Qwen9 cross-family validator, not Qwen27 formal score. '
            '75% is a predeclared representative tap, not a final-target-selected optimum. '
            'V6 settings replay in validated current runner; not full historical binary replay.'))
    for row in p['ladder']+[dict(condition=p['smoke'])]:
        c=row['condition'];atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    return p

def reuse_controls(p):pass
