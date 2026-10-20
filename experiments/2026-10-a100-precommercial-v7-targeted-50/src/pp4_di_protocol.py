"""DI removal only: two fixed compositions, ten historical dev images each."""
import json
from pp4_protocol import OUT,PREVIOUS,EXP,VALIDATORS,ANCHOR,SECOND,condition,development_ids,attack_status,sha,atomic_json
from supplement_registry import registry
ROOT=OUT/'pull_push_di_removal'


def plan():
    rows=[dict(group=g,label=label,condition=condition(models,pull=.5,di=di))
          for g,models in [('A',ANCHOR),('B',SECOND)]
          for label,di in [('DI_on',True),('DI_off',False)]]
    from pp4_protocol import plan as original
    return dict(ladder=rows,weights=[],smoke=original()['smoke'])


def validation_path(c,model):return ROOT/'validation'/c['id']/(model.replace('/','_')+'.json')


def freeze():
    from commercial_matrix_pack import freeze as immutable
    p=plan()
    sources=['pp4_di_protocol.py','pp4_di_worker.py','pp4_protocol.py','pp4_objective.py','pp4_worker.py',
        'run_v7_attack.py','representation_objectives.py','attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']
    immutable(ROOT/'protocol_lock.json',dict(protocol='pp4_DI_removal_dev10_v1',plan=p,
        source_hashes={n:sha(EXP/'src'/n) for n in sources},model_revisions=registry(),
        development_ids=development_ids(),validators=VALIDATORS,pull=.5,push=.5,layers=[.25,.5,.75,1.],
        patch=0,CLS=0,steps=50,epsilon='16/255',alpha='1/255',momentum=1,seed=42,random_start=True,
        BF16=True,quantization=False,full_matrix=False,final_target_access=False,commercial_testing=False,
        interpretation='Exploratory previously inspected ten images;30 correlated model-image observations; not independent confirmatory evidence'))
    for row in p['ladder']+[dict(condition=p['smoke'])]:
        c=row['condition'];atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    return p


def reuse_controls(p):
    old=OUT/'pull_push_composition_ablation'
    for row in p['ladder']:
        if row['label']!='DI_on':continue
        c=row['condition'];a=attack_status(c,deep=True)
        if not a['complete']:raise RuntimeError('Expected completed identical DI-on attacks')
        states=[dict(path=str(f),sha256=sha(f)) for f in sorted((OUT/'attacks/dev'/c['name']).glob('T*/batch_*/state.json'))]
        for model in VALIDATORS:
            path=old/'validation'/c['id']/(model.replace('/','_')+'.json')
            d=json.loads(path.read_text())
            if d['denominator']!=10 or d['model']!=model:raise RuntimeError('Cached evaluation differs')
            if {r['image_id'] for r in d['images']}!={i for ids in development_ids().values() for i in ids}:raise RuntimeError('Cached cohort differs')
            d.update(reused_from=str(path),reused_sha256=sha(path),new_inference=False)
            atomic_json(validation_path(c,model),d)
        atomic_json(ROOT/'reuse'/(c['id']+'.json'),dict(states=states,N=10,source=str(old),new_attack_generated=False))
