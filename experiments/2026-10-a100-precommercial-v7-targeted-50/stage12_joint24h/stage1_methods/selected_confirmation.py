"""Additional frozen validation; never rewrite the original method/CLI locks."""
from pathlib import Path
import common as C
import selection as S
from stage1_methods.fixed_four import exact
OUT=C.ROOT/'selected_followup'

def choose(rows):
    valid=[r for r in rows if r['split']=='explore' and r['strategy']!='uniform4' and r['complete'] and r['N']==30]
    if not valid:raise RuntimeError('No complete development30 candidate')
    return max(valid,key=lambda r:(r['hits'],-sum(map(len,r['layers'].values())),r['condition']))

def prepare():
    path=OUT/'method_lock.json';locked=C.read(path)
    if locked:return locked
    data=C.read(C.ROOT/'reports/three_proxy_selected_layers.json')
    if not data or not data.get('complete'):raise RuntimeError('Selected-layer development study incomplete')
    families={}
    original=C.read(C.ROOT/'method_lock.json')
    for fam in C.VALIDATORS:
        win=choose([r for r in data['rows'] if r['family']==fam])
        c=exact(fam,win['layers'],'selected_followup50','confirm')
        base=exact(fam,original['families'][fam]['uniform'],'uniform','confirm')
        C.legal(c['layers'],C.LARGE_EVAL_MODES['evaluate_large_'+fam])
        families[fam]=dict(selected=c,uniform=base,development_condition=win['condition'],
          development_hits=win['hits'],development_N=30,evaluation_sha256=C.sha(C.eval_path(C.read(C.ROOT/'conditions'/(win['condition']+'.json')))))
    locked=dict(version='selected_followup_v1',families=families,cohort_sha256=C.sha(C.ROOT/'manifests/confirm.json'),
      selection_rule='development30_hits_then_fewer_native_taps_then_stable_condition_id',
      original_method_lock_sha256=C.sha(C.ROOT/'method_lock.json'),registry_sha256=C.digest(C.REG),
      selected_before_additional_confirmation=True,selection_feedback='development30_only',
      cohort_disclosure='Same confirmation50, original Uniform already evaluated; not a new untouched dataset',
      final150_sealed=True,original_commercial_unchanged=True,no_additional_commercial_calls=True)
    C.freeze(path,locked);return locked

def validate(c,mode):
    from paper_closeout.plan import validate_large
    if mode in C.LARGE_EVAL_MODES and validate_large(c,mode):
        C.legal(c['layers'],C.LARGE_EVAL_MODES[mode])
        if not C.attack_done(c):raise RuntimeError('Large target before PNG freeze')
        return
    lock=C.read(OUT/'method_lock.json')
    if not lock or mode not in C.LARGE_EVAL_MODES or c['split']!='confirm':raise RuntimeError('Large evaluation requires frozen confirmation method')
    if lock['cohort_sha256']!=C.sha(C.ROOT/'manifests/confirm.json'):raise RuntimeError('Frozen cohort changed')
    allowed=c['id'] in {v[k]['id'] for v in lock['families'].values() for k in ['selected','uniform']}
    if not allowed:
        from stage1_methods.fda_confirmation import contains
        allowed=contains(c)
    if not allowed:raise RuntimeError('Unfrozen large-target method')
    C.legal(c['layers'],C.LARGE_EVAL_MODES[mode])
    if not C.attack_done(c):raise RuntimeError('Large target accessed before PNG freeze')

def large_path(c,mode):return C.ROOT/'large_validation'/C.LARGE_EVAL_MODES[mode]/(c['id']+'.json')
def large_done(c,mode):
    d=C.read(large_path(c,mode))
    return bool(d and d.get('complete') and d.get('N')==len(C.cohort(c['split'])['images']) and d.get('model')==C.LARGE_EVAL_MODES[mode] and d.get('science')==C.digest(C.scientific(c)))

def plan():
    lock=prepare();pending=[]
    for fam,v in lock['families'].items():
        c=v['selected']
        if not C.attack_done(c) or not C.eval_done(c):pending.append(c)
    return pending,not pending

def large_pending():
    lock=prepare();pending=[]
    for fam,v in lock['families'].items():
        mode='evaluate_large_'+fam
        for k in ['uniform','selected']:
            c=v[k]
            if C.attack_done(c) and not large_done(c,mode):pending.append((c,mode))
    return list({(c['id'],m):(c,m) for c,m in pending}.values())

def report():
    lock=prepare();rows=[]
    for fam,v in lock['families'].items():
        for method in ['uniform','selected']:
            c=v[method]
            for mode in ['evaluate','evaluate_large_'+fam]:
                p=C.eval_path(c) if mode=='evaluate' else large_path(c,mode)
                d=C.read(p,{})
                rows.append(dict(family=fam,method=method,condition=c['id'],model=d.get('model',c['validator'] if mode=='evaluate' else C.LARGE_EVAL_MODES[mode]),
                  complete=d.get('complete',False),hits=d.get('hits') if d.get('complete') else None,N=50,
                  ASR=d.get('ASR') if d.get('complete') else None,role='paired_confirmation_not_selection'))
    C.atomic(OUT/'results.json',dict(rows=rows,complete=all(x['complete'] for x in rows),commercial_unchanged=True))
    return rows
