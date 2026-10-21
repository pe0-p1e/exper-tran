import itertools,random
from pathlib import Path
import common as C
import selection as S
from stage1_methods.fixed_four import exact
from stage1_methods import planner as P
OUT=C.ROOT/'paper_closeout'
METHODS=['uniform','selected','FDA']
MODELS=['gpt-6.1-sol','gpt-6-sol','gpt-5.6-sol']

def modify(c,**changes):
    c={**c,**changes};c.pop('id',None);label=c.pop('label','closeout')
    c['id']='V8_'+C.digest(c)[:18];c['label']=label
    C.legal(c['layers'],c['validator'])
    for p in (C.ROOT/'conditions').glob('*.json'):
        old=C.read(p)
        if C.scientific(old)==C.scientific(c):return old
    return c

def matched_layers(fam,order,layers,label,split):
    if set(order)!=set(layers):raise ValueError('Proxy identities changed')
    return exact(fam,{n:layers[n] for n in order},label,split)

def selections():
    """FDA3/4 is actual conditional growth; independent4 ranks existing singles."""
    from stage1_methods.fda_matched import prepare,grow
    matched=prepare();pending=[];conditions={}
    for fam,v in matched['families'].items():
        order=v['order'];top4={};fda3={};fda4={};evidence={}
        for n in order:
            singles=[c for c in S.layer_conditions(fam) if n in c['layers']]
            if not all(C.eval_done(c) for c in singles):raise RuntimeError('Singleton evidence missing')
            ranked=sorted(singles,key=S.rank,reverse=True)
            top4[n]=sorted(c['layers'][n][0] for c in ranked[:4])
            missing,chosen,trace=grow(fam,n);pending+=missing
            # Freeze three-layer parent only when every candidate at size3 completed.
            level3=[x for x in trace if x['size']==3]
            cs=[C.read(C.ROOT/'conditions'/(x['candidate']+'.json')) for x in level3]
            if cs and all(c and P.completed(c) for c in cs):fda3[n]=P.best(cs)[0]['layers'][n]
            if chosen:fda4[n]=chosen['layers'][n]
            evidence[n]=dict(singletons=[c['id'] for c in ranked],conditional_trace=trace)
        definitions={'selected4':top4}
        if len(fda3)==3:definitions['FDA3']=fda3
        if len(fda4)==3:definitions['FDA4']=fda4
        conditions[fam]={}
        for method,layers in definitions.items():
            C.freeze(OUT/'selection_locks'/(fam+'_'+method+'.json'),dict(layers=layers,order=order,
              rule='independent singleton top4' if method=='selected4' else 'conditional FDA development10',
              feedback='screen10 only',evidence=evidence if method!='selected4' else {n:evidence[n]['singletons'] for n in order}))
            conditions[fam][method]={}
            for split in ['explore','confirm']:
                c=matched_layers(fam,order,layers,method+'_'+split,split)
                conditions[fam][method][split]=c
                if not P.completed(c):pending.append(c)
    C.atomic(OUT/'followup_conditions.json',conditions)
    return list({c['id']:c for c in pending}.values()),conditions

def best(candidates):
    """Development30-only ranking, no commercial/large/final score access."""
    if not candidates or any(c['split']!='explore' or not C.eval_done(c) for c in candidates):raise ValueError('Require completed development30 candidates')
    return max(candidates,key=lambda c:(C.read(C.eval_path(c))['hits'],-sum(map(len,c['layers'].values())),c['id']))

def freeze_final(conditions):
    old=C.read(OUT/'final_method_lock.json')
    if old:return old
    matched=C.read(C.ROOT/'fda_matched/results.json',{})
    if not matched.get('complete') or any(len(v)!=3 for v in conditions.values()):return None
    commercial=C.read(OUT/'commercial50/summary.json',{})
    if not commercial.get('complete'):return None
    specs=commercial_specs(conditions)
    winners={}
    groups={'selected':['selected_prior','selected4'],'FDA':['FDA2','FDA3','FDA4'],'uniform':['uniform2','uniform4']}
    for method,names in groups.items():
        candidates=[r for r in commercial['methods'] if r['method'] in names]
        if len(candidates)!=len(names) or any(r['completed_pairs']!=50 for r in candidates):return None
        winners[method]=max(candidates,key=lambda r:(r['hits'],-sum(map(len,specs[r['method']]['qwen']['layers'].values())),r['method']))['method']
    families={}
    for fam in C.VALIDATORS:
        new=conditions[fam]
        if not all(C.eval_done(v['explore']) for v in new.values()):return None
        families[fam]={}
        for method in METHODS:
            chosen=winners[method];source=specs[chosen][fam]
            c=exact(fam,source['layers'],'final150_'+method,'final150')
            families[fam][method]=dict(condition=c,commercial50_selected_strategy=chosen,selection_condition=specs[chosen]['qwen']['id'])
    lock=dict(families=families,rule='User override: highest GPT6.1 Sol low development50 hits, fewer taps, stable method name',
      GPT50_used_for_selection=True,final150_feedback_for_selection=False,N=150,commercial_family='qwen',models=MODELS,effort='low',
      commercial50_summary_sha256=C.sha(OUT/'commercial50/summary.json'),winners=winners,
      final_cohort_sha256=C.sha(C.ROOT/'manifests/final150.json'),
      caveat='GPT50 is development, not untouched confirmation; final150 is a new holdout. A generic qwen-pool choice maps to predeclared legal family-specific configurations.')
    C.freeze(OUT/'final_method_lock.json',lock);return lock

def commercial_specs(conditions):
    original=C.read(C.ROOT/'selected_followup/method_lock.json');matched=C.read(C.ROOT/'fda_matched/protocol_lock.json')
    result={name:{} for name in ['selected_prior','selected4','FDA2','FDA3','FDA4','uniform2','uniform4']}
    for fam,v in original['families'].items():
        result['selected_prior'][fam]=v['selected'];result['uniform4'][fam]=v['uniform']
        base=matched['families'][fam]
        for label,key in [('FDA2','FDA2'),('uniform2','uniform2')]:
            result[label][fam]=matched_layers(fam,base['order'],base[key],label+'_confirm','confirm')
        for label in ['selected4','FDA3','FDA4']:
            if label in conditions[fam]:result[label][fam]=conditions[fam][label]['confirm']
    return result

def validate_large(c,mode):
    if not (OUT/'activation.json').exists():return False
    allowed=[]
    follow=C.read(OUT/'followup_conditions.json',{})
    for fam,methods in follow.items():
        if mode=='evaluate_large_'+fam:allowed.extend(v['confirm']['id'] for v in methods.values() if 'confirm' in v)
    final=C.read(OUT/'final_method_lock.json',{})
    for fam,methods in final.get('families',{}).items():
        if mode=='evaluate_large_'+fam:allowed.extend(v['condition']['id'] for v in methods.values() if 'condition' in v)
    fam=mode.replace('evaluate_large_','')
    allowed.extend(x['id'] for x in C.read(OUT/'pull_push_ablation_lock.json',{}).get(fam,[]))
    return c['id'] in allowed

def plan():
    from stage1_methods.selected_confirmation import large_done
    pending,conditions=selections();jobs=[]
    specs=commercial_specs(conditions)
    for name in ['FDA2','uniform2']:
        c=specs[name]['qwen']
        if not P.completed(c):pending.append(c)
    lock=freeze_final(conditions)
    if lock:
        for fam,v in lock['families'].items():
            mode='evaluate_large_'+fam
            for item in v.values():
                c=item['condition']
                if not C.attack_done(c):pending.append(c)
                elif not large_done(c,mode):jobs.append((c,mode))
    # Independent of final selection: fixed3-proxy Uniform4 ratio controls.
    ratios=prepare_ratios()
    for fam,values in ratios.items():
        for c in values:
            if not C.attack_done(c):pending.append(c)
            elif not large_done(c,'evaluate_large_'+fam):jobs.append((c,'evaluate_large_'+fam))
    C.atomic(OUT/'progress.json',dict(at=C.time.time(),final_frozen=bool(lock),pending_attack_or_dev=len({c['id'] for c in pending}),pending_large=len(jobs)))
    return list({c['id']:c for c in pending}.values()),jobs,lock

def prepare_ratios():
    old=C.read(OUT/'pull_push_ablation_lock.json')
    if old is not None:return old
    original=C.read(C.ROOT/'method_lock.json');out={}
    for fam in C.VALIDATORS:
        base=exact(fam,original['families'][fam]['uniform'],'ratio_uniform4','explore')
        out[fam]=[modify(base,pull=p,push=1-p,label='PP_ratio_'+str(p)) for p in [1.,.25,.5,.75,0.]]
    C.freeze(OUT/'pull_push_ablation_lock.json',out);return out
