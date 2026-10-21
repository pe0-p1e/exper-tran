"""Same-three-proxy, same-tap-count controls; development10-only FDA growth."""
import common as C
import selection as S
from stage1_methods.fixed_four import exact, pair_outcomes
from stage1_methods import planner as P
OUT=C.ROOT/'fda_matched'

def matched(fam,order,layers,label):
    if set(layers)!=set(order):raise ValueError('Matched comparison changed proxies')
    counts={len(layers[n]) for n in order}
    if len(counts)!=1 or next(iter(counts)) not in [2,4]:raise ValueError('Unequal per-proxy tap counts')
    return exact(fam,{n:layers[n] for n in order},label,'explore')

def prepare():
    old=C.read(OUT/'protocol_lock.json')
    if old:return old
    original=C.read(C.ROOT/'method_lock.json');families={}
    for fam in C.VALIDATORS:
        base=original['families'][fam]['uniform'];order=list(base)
        policy=C.read(P.OUT/(fam+'_proxy_layer_policy.json'))
        families[fam]=dict(order=order,FDA2=policy['layers'],uniform4=base,
          uniform2={n:C.native(n,[.5,1.]) for n in order})
    lock=dict(version='FDA_matched_v1',families=families,
      screen_sha256=C.sha(C.ROOT/'manifests/screen.json'),explore_sha256=C.sha(C.ROOT/'manifests/explore.json'),
      rule='Same six native depth candidates and development10 S.rank as prior FDA; sequential conditional growth to4 for EVERY fixed proxy',
      score_role='development validators only; no large/commercial feedback',
      primary_contrasts=['FDA2 vs uniform2','FDA4 vs uniform4'],
      secondary='FDA2 vs uniform4 is layer-count/cost comparison, not equal-count selection evidence',
      N=30,max_new_screen_conditions=48,final150_sealed=True,commercial_locks_unchanged=True)
    C.freeze(OUT/'protocol_lock.json',lock);return lock

def grow(fam,n):
    # Reuse exact original FDA2 prefix, then score all remaining additions.
    available=C.native(n,[.125,.25,.5,.625,.75,1.])
    singles=[P.condition(fam,[(n,l)],'FDA_single') for l in available]
    if any(not P.completed(c) for c in singles):raise RuntimeError('FDA singleton evidence incomplete')
    parent=P.best(singles)[0];trace=[]
    for size in [2,3,4]:
        candidates=[P.condition(fam,P.units(parent)+[(n,l)],'FDA_conditional_'+str(size))
          for l in available if l not in parent['layers'][n]]
        trace.extend(dict(parent=parent['id'],candidate=c['id'],size=size) for c in candidates)
        missing=[c for c in candidates if not P.completed(c)]
        if missing:return missing,None,trace
        parent=P.best(candidates)[0]
    return [],parent,trace

def plan():
    lock=prepare();pending=[];rows=[];traces={};new={}
    for fam,v in lock['families'].items():
        for label,key in [('uniform2','uniform2'),('FDA2','FDA2'),('uniform4','uniform4')]:
            c=matched(fam,v['order'],v[key],'matched_'+label+'30')
            if not P.completed(c):pending.append(c)
            rows.append(dict(family=fam,method=label,condition=c,complete=P.completed(c)))
        layers={};family_trace={}
        for n in v['order']:
            missing,chosen,trace=grow(fam,n);pending+=missing;family_trace[n]=trace
            for c in missing:
                if not (C.ROOT/'conditions'/(c['id']+'.json')).exists():new[c['id']]=c
            if chosen:layers[n]=chosen['layers'][n]
        traces[fam]=family_trace
        if len(layers)==len(v['order']):
            selected=dict(layers=layers,evidence=family_trace,selection_split='screen',no_target_retuning=True)
            C.freeze(OUT/(fam+'_FDA4_selection.json'),selected)
            c=matched(fam,v['order'],layers,'matched_FDA4_30')
            if not P.completed(c):pending.append(c)
            rows.append(dict(family=fam,method='FDA4',condition=c,complete=P.completed(c)))
    known=C.read(OUT/'new_search_conditions.json',{});known.update(new)
    if len(known)>lock['max_new_screen_conditions']:raise RuntimeError('Matched FDA bounded search cap exceeded')
    C.atomic(OUT/'new_search_conditions.json',known)
    for row in rows:
        c=row.pop('condition');row.update(condition=c['id'],layers=c['layers'],proxy_count=3,
          taps_per_proxy=len(next(iter(c['layers'].values()))),N=30)
        d=C.read(C.eval_path(c),{});row['hits']=d.get('hits') if row['complete'] else None
    pairs=[]
    for fam in C.VALIDATORS:
        by={r['method']:r for r in rows if r['family']==fam}
        for left,right in [('uniform2','FDA2'),('uniform4','FDA4')]:
            if right not in by:continue
            a,b=by[left],by[right]
            if a['complete'] and b['complete']:
                ca=C.read(C.ROOT/'conditions'/(a['condition']+'.json'));cb=C.read(C.ROOT/'conditions'/(b['condition']+'.json'))
                pairs.append(dict(family=fam,baseline=left,selected=right,**pair_outcomes(ca,cb)))
    ready=not pending and len(rows)==12 and all(r['complete'] for r in rows)
    C.atomic(OUT/'results.json',dict(complete=ready,rows=rows,paired=pairs,search_traces=traces,
      pending=len({c['id'] for c in pending}),selection_uses_screen_only=True))
    return list({c['id']:c for c in pending}.values()),ready
