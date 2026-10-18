"""Controlled, surrogate-only extensions to the original development queue."""
from copy import deepcopy
import json
import math


def extended_budget(previous, hours=20, now=0):
    """An explicitly requested total budget preserves the original start time."""
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('Budget hours must be finite and positive')
    if previous is None:
        return dict(started=now, deadline=now+hours*3600, hours=hours)
    result=deepcopy(previous)
    result['deadline']=max(previous['deadline'], previous['started']+hours*3600)
    result['hours']=(result['deadline']-result['started'])/3600
    return result


def layer_conditions(models):
    """One fixed anchor; every condition changes only depth or layer weights."""
    depths=[.25,.5,.75,1.]
    variants=[(f'single_{i}',[d],None) for i,d in enumerate(depths)]
    variants += [(f'without_{i}',[d for j,d in enumerate(depths) if j!=i],None) for i in range(4)]
    variants += [('late_pair',[.75,1.],None),('middle_pair',[.5,.75],None),('spread_pair',[.25,1.],None)]
    variants += [('early_weight',depths,[4,3,2,1]),('late_weight',depths,[1,2,3,4]),('middle_weight',depths,[1,3,3,1])]
    return [dict(id='T20_layer_'+name,models=list(models),method='A3',options=dict(layers=layers,**({'layer_weights':weights} if weights else {})),family='layer_mechanism') for name,layers,weights in variants]


def complementarity_conditions(anchor):
    """Freeze one selected ensemble; compare all individuals and leave-one-outs."""
    base=deepcopy(anchor)
    for key in ['id','n_per_direction','seed']:
        base.pop(key,None)
    result=[]
    for i,name in enumerate(anchor['models']):
        result.append(dict(base,id=f'T20_component_{i}',models=[name],family='proxy_complementarity'))
        if len(anchor['models'])>2:
            result.append(dict(base,id=f'T20_without_proxy_{i}',models=[n for j,n in enumerate(anchor['models']) if j!=i],family='proxy_complementarity'))
    result.append(dict(base,id='T20_complementarity_anchor',family='proxy_complementarity'))
    return result


def unique_candidates(ranked):
    """Do not spend the expanded confirmation budget on repeated controls."""
    seen=set();result=[]
    for item in ranked:
        c=item['config']
        identity=dict(models=c['models'],method=c.get('method','A3'),options=c.get('options',{}),fusion=c.get('fusion','equal'),di=c.get('di',True),ti=c.get('ti',False),seed=c.get('seed',42),steps=c.get('steps',50),restarts=c.get('restarts',1))
        key=json.dumps(identity,sort_keys=True)
        if key not in seen:result.append(item);seen.add(key)
    return result
