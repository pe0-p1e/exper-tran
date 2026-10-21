"""Keep all THREE proxies; select TWO/THREE native layers independently per proxy.

Independent singleton ranking is a control; geometry proposes complementary taps.
Actual attacks still differentiate mean layer losses, then normalize proxy gradients.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C
import combo_selection as J
import selection as S
from stage1_methods.fixed_four import exact,pair_outcomes
OUT=C.ROOT/'stage1_methods/three_proxy_layers'

def mapping(units,order):return {n:sorted(l for nn,l in units if n==nn) for n in order}

def source_conditional(pool,seeds,g,order):
    """Pairwise source heuristic, not gradient of a mean of normalized tap gradients."""
    current=list(seeds);result={}
    for cardinality in [2,3]:
        while any(sum(n==model for n,_ in current)<cardinality for model in order):
            candidates=[u for u in pool if u not in current and sum(n==u[0] for n,_ in current)<cardinality]
            if not candidates:raise ValueError('Insufficient distinct native layers per model')
            chosen=max(candidates,key=lambda u:(J.prior(g,u,current),u))
            current.append(chosen)
        result[cardinality]=mapping(current,order)
    return result

def select(fam,g):
    base=J.reuse_make(fam,S.uniform(C.POOLS[fam][:3]),'three_proxy_uniform','screen')
    order=base['model_order'];all_singles=S.layer_conditions(fam);ranked={};evidence={};pool=[];seeds=[]
    for n in order:
        candidates=[c for c in all_singles if n in c['layers']]
        if len(candidates)<3 or not all(C.attack_done(c) and C.eval_done(c) for c in candidates):raise RuntimeError('Need completed per-model singleton evidence '+n)
        candidates=sorted(candidates,key=S.rank,reverse=True)
        ranked[n]=[c['layers'][n][0] for c in candidates]
        evidence[n]=[dict(condition=c['id'],evaluation_sha256=C.sha(C.eval_path(c))) for c in candidates]
        pool.extend((n,l) for l in ranked[n]);seeds.append((n,ranked[n][0]))
    geometry=source_conditional(pool,seeds,g,order)
    selected={}
    for k in [2,3]:
        selected[f'independent_top{k}']={n:sorted(ranked[n][:k]) for n in order}
        selected[f'geometry_selected{k}']=geometry[k]
    result=dict(order=order,selected=selected,evidence=evidence,geometry_inputs=g['inputs'],
      source_only_geometry=True,development_singletons_used=True,target_feedback=False)
    C.freeze(OUT/(fam+'_layer_selection.json'),result)
    return result

def make_conditions(fam,selection,split):
    base=J.reuse_make(fam,S.uniform(C.POOLS[fam][:3]),'three_proxy_uniform',split)
    out={}
    for strategy,ls in selection['selected'].items():
        # Preserve the actual baseline proxy order for this cohort.
        ls={n:ls[n] for n in base['model_order']}
        c=exact(fam,ls,'three_proxy_'+strategy,split)
        a,b=C.scientific(base),C.scientific(c)
        if {k:v for k,v in a.items() if k!='layers'}!={k:v for k,v in b.items() if k!='layers'}:raise RuntimeError('Non-layer settings changed')
        k=2 if strategy.endswith('2') else 3
        if len(ls)!=3 or any(len(taps)!=k for taps in ls.values()):raise RuntimeError('Require exactly3 proxies and k layers/model')
        out[strategy]=c
    return base,out

def plan(g):
    C.freeze(OUT/'protocol_lock.json',dict(version='three_proxy_2_3_layers_v1',source_sha256=C.sha(__file__),
      cohorts={s:C.sha(C.ROOT/'manifests'/(s+'.json')) for s in ['screen','explore']},
      strategies=['independent_top2','geometry_selected2','independent_top3','geometry_selected3'],
      precision='bfloat16',fusion='mean_layer_losses_before_per_proxy_L1_then_equal_mean',
      development_only=True,commercial_method_unchanged=True,active_proxies=3))
    pending=[];rows=[]
    for fam in C.VALIDATORS:
        choice=select(fam,g);base,cc=make_conditions(fam,choice,'screen')
        jobs=[('uniform4',base)]+list(cc.items())
        missing=[c for c in cc.values() if not C.attack_done(c) or not C.eval_done(c)]
        pending+=missing
        if not missing:
            winners={}
            for k in [2,3]:
                options=[(label,c) for label,c in cc.items() if label.endswith(str(k))]
                label,c=max(options,key=lambda item:S.rank(item[1]));winners[str(k)]=dict(strategy=label,condition=c['id'])
            C.freeze(OUT/(fam+'_promotion.json'),dict(rule='best_development_screen_candidate_at_each_layer_count',winners=winners,feedback='screen10_only'))
            base30,cs30=make_conditions(fam,choice,'explore');jobs.append(('uniform4',base30))
            if not C.attack_done(base30) or not C.eval_done(base30):pending.append(base30)
            for winner in winners.values():
                c=cs30[winner['strategy']];jobs.append((winner['strategy'],c))
                if not C.attack_done(c) or not C.eval_done(c):pending.append(c)
        for strategy,c in jobs:
            done=C.attack_done(c) and C.eval_done(c);metric=S.score(c) if done else None
            ref=base if c['split']=='screen' else base30
            rows.append(dict(family=fam,split=c['split'],strategy=strategy,condition=c['id'],layers=c['layers'],proxy_count=3,
              layers_per_proxy=4 if strategy=='uniform4' else 2 if strategy.endswith('2') else 3,
              complete=done,hits=metric['hits'] if metric else None,N=10 if c['split']=='screen' else 30,
              paired_vs_uniform=pair_outcomes(ref,c) if done else None,compute=metric))
    pending=list({c['id']:c for c in pending}.values())
    result=dict(complete=not pending,pending=len(pending),rows=rows,not_a_2proxy_or_3total_tap_experiment=True,
       independent_ranking_not_assumed_joint_optimal=True,geometry_is_prescreen_not_mathematical_guarantee=True)
    C.atomic(OUT/'results.json',result);C.atomic(C.ROOT/'reports/three_proxy_selected_layers.json',result)
    return pending,not pending
