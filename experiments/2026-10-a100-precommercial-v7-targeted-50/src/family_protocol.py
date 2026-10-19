"""Fail-closed target-family isolation for the corrected cross-family study."""


def require_excluded_family(models, excluded_family, registry):
    if not models:
        raise ValueError('An attack must have at least one proxy')
    if not excluded_family:
        raise ValueError('Explicit excluded family required')
    collisions=[n for n in models if registry[n]['family']==excluded_family]
    if collisions:
        raise ValueError(f'Forbidden target-family proxies ({excluded_family}): {collisions}')


def require_condition_target(condition, target, models, lock, registry):
    if lock.get('protocol')!='strict_family_disjoint':
        return  # Historical checkpoint-disjoint studies remain unchanged.
    if lock['condition_targets'].get(condition)!=target:
        raise ValueError('Condition is not predeclared for this target')
    require_excluded_family(models,registry[target]['family'],registry)
    expected=lock['condition_models'].get(condition)
    if expected!=models:
        raise ValueError('Actual proxy list differs from predeclared family-isolation lock')


def corrected_conditions():
    """Fixed protocol correction; no post-target-feedback algorithm selection."""
    rows=[]
    for prefix,target,proxies in [
        ('CFQ','Qwen3.5-27B',['CLIP ViT-L/14','InternVL3.5-2B-HF','Gemma 4 E2B-it']),
        ('CFI','InternVL3.5-14B-HF',['CLIP ViT-L/14','Qwen3.5-2B','Gemma 4 E2B-it']),
    ]:
        def add(suffix,kind,models=None,method='A3',options=None,di=True,seed=42):
            rows.append(dict(id=prefix+'_'+suffix,target=target,models=list(models or proxies),method=method,options=options if options is not None else {'patch_weight':.35},di=di,seed=seed,steps=50,restarts=1,fusion='equal',kind=kind))
        add('main','primary_multi')
        for i,model in enumerate(proxies):add(f'single_{i}','matched_single',models=[model])
        for percent in [1,25,50,75,100]:add(f'layer_{percent}','layer',options={'patch_weight':.35,'layers':[percent/100]})
        add('no_patch','loss_ablation',options={})
        add('patch_only','loss_ablation',options={'patch_weight':1.})
        add('soft_references','loss_ablation',options={'patch_weight':.35,'aggregation':'soft_reference','reference_temperature':.05})
        add('original_pullpush','historical_algorithm_control',method='A0',options={},di=False)
        add('no_DI','input_ablation',di=False)
        for seed in [43,44]:add(f'seed_{seed}','primary_seed_check',seed=seed)
    return rows


def condition_name(c):
    name=c['id']+'__'+c['method']+'__'+c['fusion']+'__'+'+'.join(n.replace('/','_').replace(' ','_') for n in c['models'])
    if c['di']:name+='__DI'
    return name+f"__s{c['seed']}__i{c['steps']}__r{c['restarts']}"
