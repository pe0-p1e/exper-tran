"""Frozen FDA development30/confirmation50; no final-target method selection."""
import common as C
from stage1_methods.fixed_four import exact
from stage1_methods.selected_confirmation import large_done
OUT=C.ROOT/'fda_followup'

def prepare():
    old=C.read(OUT/'method_lock.json')
    if old:return old
    registry=C.read(C.ROOT/'stage1_methods/registry.json');families={}
    for fam in C.VALIDATORS:
        result=C.read(C.ROOT/'stage1_methods'/(fam+'_method_results.json'))
        if not result or not result.get('complete'):raise RuntimeError('Need finished FDA/proxy development selection: '+fam)
        source=registry[result['fixed_proxy_best']]['condition']
        evidence=C.read(C.eval_path(source))
        if not C.eval_done(source) or evidence['N']!=10:raise RuntimeError('Missing completed FDA selection evidence')
        policy=C.read(C.ROOT/'stage1_methods'/(fam+'_proxy_layer_policy.json'))
        baseline=C.read(C.ROOT/'method_lock.json')['families'][fam]['uniform']
        control={n:policy['layers'][n] for n in baseline}
        families[fam]=dict(source_condition=source['id'],selection_hits=evidence['hits'],selection_N=10,
          selection_evaluation_sha256=C.sha(C.eval_path(source)),
          main30=exact(fam,source['layers'],'FDA_selected_subset30','explore'),
          main50=exact(fam,source['layers'],'FDA_selected_subset50','confirm'),
          matched_control30=exact(fam,control,'FDA_fixed_three_proxies30','explore'))
    lock=dict(version='fda_confirmation_v1',families=families,selection='previously frozen development10 fixed_proxy_best; no30/50/GPT retuning',
      definition='FDA-inspired conditional two-layer selection per proxy, then development-selected proxy subset',
      control='same three Uniform proxy identities/order, each using its FDA-selected two native layers',
      manifests={split:C.sha(C.ROOT/'manifests'/(split+'.json')) for split in ['explore','confirm']},
      registry_sha256=C.digest(C.REG),final150_sealed=True,commercial_family='qwen',commercial_model='gpt-6.1-sol',effort='low')
    C.freeze(OUT/'method_lock.json',lock);return lock

def plan():
    lock=prepare();pending=[]
    for v in lock['families'].values():
        for key in ['main30','main50','matched_control30']:
            c=v[key]
            if not C.attack_done(c) or not C.eval_done(c):pending.append(c)
    return list({c['id']:c for c in pending}.values()),not pending

def contains(c):
    lock=C.read(OUT/'method_lock.json')
    return bool(lock and any(c['id']==v['main50']['id'] for v in lock['families'].values()) and
      lock['manifests']['confirm']==C.sha(C.ROOT/'manifests/confirm.json'))

def large_pending():
    lock=prepare();jobs=[]
    for fam,v in lock['families'].items():
        c=v['main50'];mode='evaluate_large_'+fam
        if C.attack_done(c) and not large_done(c,mode):jobs.append((c,mode))
    return jobs
