"""Small parallel mechanism pilots: gradient scaling and selected layer subsets."""
from pp4_di_protocol import *
ROOT=OUT/'pull_push_mechanism_pilots'


def plan():
    from pp4_protocol import fingerprint
    from family_protocol import condition_name
    rows=[]
    for group,models in [('A',ANCHOR),('B',SECOND)]:
        for label,layers,raw in [('reference',[.25,.5,.75,1.],False),('raw_gradient_mean',[.25,.5,.75,1.],True),
                ('single_75',[.75],False),('middle_pair',[.5,.75],False),('late_triple',[.5,.75,1.],False)]:
            c=condition(models,pull=.5,layers=layers)
            if raw:
                c['options'].update(gradient_normalization='none',raw_fusion_source_sha256=sha(EXP/'src/pp4_raw_fusion.py'))
                c['id']='PP4RAW_'+fingerprint([models,c['options']])[:16];c['name']=condition_name(c)
            rows.append(dict(group=group,label=label,condition=c))
    from pp4_protocol import plan as original
    return dict(ladder=rows,weights=[],smoke=original()['smoke'])


def validation_path(c,model):return ROOT/'validation'/c['id']/(model.replace('/','_')+'.json')


def freeze():
    from commercial_matrix_pack import freeze as immutable
    p=plan()
    sources=['pp4_mechanism_protocol.py','pp4_mechanism_worker.py','pp4_raw_fusion.py','pp4_di_protocol.py',
        'pp4_protocol.py','pp4_objective.py','pp4_worker.py','run_v7_attack.py','representation_objectives.py',
        'attack_optimizers.py','vision_features.py','evaluate_surrogate_outputs.py']
    immutable(ROOT/'protocol_lock.json',dict(protocol='pp4_small_mechanism_pilots_v1',plan=p,
        source_hashes={n:sha(EXP/'src'/n) for n in sources},model_revisions=registry(),
        development_ids=development_ids(),validators=VALIDATORS,steps=50,epsilon='16/255',alpha='1/255',
        seed=42,momentum=1,random_start=True,pull=.5,push=.5,DI=True,patch=0,CLS=0,BF16=True,quantization=False,
        final_target_access=False,commercial_testing=False,full_matrix=False,
        layer_rationale='75% is the best historical A3 single on this inspected surrogate dev cohort; '
            'test its PP behavior and middle pair/late triple marginal additions; not a global optimality claim',
        caveat='Exploratory previously inspected dev10; three checkpoint-held-out validators, not family-held-out; no confirmatory generalization claim',
        raw_gradient_diagnostics='Additional fusion_diagnostics retain raw L1 contribution fractions, raw-vs-normalized direction cosine; native cancellation trace is normalized geometry'))
    for row in p['ladder']+[dict(condition=p['smoke'])]:
        c=row['condition'];atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    return p


def reuse_controls(p):
    from pp4_di_protocol import reuse_controls as reuse
    reuse(dict(ladder=[dict(row,label='DI_on') for row in p['ladder'] if row['label']=='reference']))
    # The imported helper resolves its original module ROOT; transfer verified receipts.
    from pp4_di_protocol import ROOT as old
    import json
    for row in p['ladder']:
        if row['label']!='reference':continue
        c=row['condition']
        for m in VALIDATORS:
            path=old/'validation'/c['id']/(m.replace('/','_')+'.json')
            atomic_json(validation_path(c,m),json.loads(path.read_text()))
        path=old/'reuse'/(c['id']+'.json')
        atomic_json(ROOT/'reuse'/(c['id']+'.json'),json.loads(path.read_text()))
