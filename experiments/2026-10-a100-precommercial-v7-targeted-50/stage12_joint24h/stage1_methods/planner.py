"""Bounded Stage1 method trials. Called by the existing admission owner only.

No GPU inference, process launch or final-target access occurs in this planner.
"""
import argparse, itertools, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common as C
import selection as S
import combo_selection as J
import numpy as np
OUT=C.ROOT/'stage1_methods'
POLICY=HERE/'protocol.yaml'

def dedup(items):return list({c['id']:c for c in items}.values())
def condition(fam,units,label,split='screen'):
    ls={}
    for n,l in sorted(set(units)):ls.setdefault(n,[]).append(l)
    C.legal(ls,C.VALIDATORS[fam])
    return J.reuse_make(fam,ls,label,split)
def units(c):return S.atoms(c)
def completed(c):return C.attack_done(c) and C.eval_done(c)
def best(cs,k=1):return sorted([c for c in dedup(cs) if completed(c)],key=S.rank,reverse=True)[:k]
def snapshot_identity():
    return dict(protocol=C.sha(POLICY),sources={p.name:C.sha(p) for p in HERE.glob('*.py')},
      base_study=C.sha(C.ROOT/'study_lock.json'),cohort=C.sha(C.ROOT/'manifests/screen.json'))
def register(cs,method):
    """Persist evidence/receipts, preserve exact prior condition/order when reusing."""
    old=C.read(OUT/'registry.json',{})
    for c in dedup(cs):
        C.legal(c['layers'],c['validator'])
        key=c['id'];entry=old.setdefault(key,dict(condition=c,methods=[],preexisting=(C.ROOT/'conditions'/(key+'.json')).exists()))
        if C.scientific(entry['condition'])!=C.scientific(c):raise RuntimeError('Changed registered condition')
        if method not in entry['methods']:entry['methods'].append(method)
    C.atomic(OUT/'registry.json',old)
    fresh=sum(not e['preexisting'] for e in old.values())
    if fresh>96:raise RuntimeError('Stage1 method cap reached; preserve partial state, no silent skip')
    return [c for c in dedup(cs) if not completed(c)]

def fda_model(fam,n,goal=2):
    available=C.native(n,[.125,.25,.5,.625,.75,1.])
    singles=[condition(fam,[(n,l)],'FDA_single') for l in available]
    missing=register(singles,'FDA_single');history=[]
    if missing:return missing,{},history
    parent=best(singles)[0];chosen={1:parent}
    for size in range(2,goal+1):
        cs=[condition(fam,units(parent)+[(n,l)],f'FDA_conditional_{size}') for l in available if (n,l) not in units(parent)]
        missing=register(cs,f'FDA_conditional_{size}')
        history.extend(dict(parent=parent['id'],candidate=c['id'],size=size) for c in cs)
        if missing:return missing,chosen,history
        parent=best(cs)[0];chosen[size]=parent
    return [],chosen,history

def proxy_fixed(fam,layer_policy):
    ns=C.POOLS[fam][:3]
    return [condition(fam,[(n,l) for n in subset for l in layer_policy[n]],'proxy_fixed_FDA2')
            for k in [1,2,3] for subset in itertools.combinations(ns,k)]

def gradient_matrix(n):
    records=[];ids=[];layers=None
    for r in C.cohort('screen')['images']:
        p=C.ROOT/'selection_gradients'/C.digest(n)[:12]/(C.digest(r['image_id'])[:18]+'.json')
        m=C.read(p);npz=p.with_suffix('.npz')
        if not m or not m.get('source_only') or m['model']!=n or m['image_id']!=r['image_id']:raise RuntimeError('Unaligned source gradient cache')
        if m['DI_state']!=C.seed(r['image_id'],'DI') or m['random_start_seed']!=C.seed(r['image_id'],'start'):raise RuntimeError('Mismatched image/DI state')
        if C.sha(npz)!=m['sha256']:raise RuntimeError('Corrupt gradient array')
        if layers is not None and layers!=m['layers']:raise RuntimeError('Different layer ordering')
        layers=m['layers']
        with np.load(npz) as a:records.append(a['gradients'].reshape(len(layers),-1))
        ids.append(dict(path=str(p),metadata_sha=C.sha(p),npz_sha=m['sha256']))
    return layers,np.stack(records),ids

def alignment_scores(data,depths):
    """PRAF-inspired PP control, NOT original clean/CLS/target-image AILS."""
    anchors=[]
    for n,(ls,a) in data.items():
        if depths[n]-1 not in ls:raise ValueError('Missing actual deepest native gradient')
        if a.ndim!=3 or not np.isfinite(a).all():raise ValueError('Invalid gradient tensor')
        anchors.append(a[:,ls.index(depths[n]-1)])
    if len({a.shape for a in anchors})!=1:raise ValueError('Cross-model image gradients must align')
    anchor=np.mean(anchors,axis=0);an=np.linalg.norm(anchor,axis=1)
    if (an<1e-20).any():raise ValueError('Global gradient cancellation; no invented rank')
    scores=[]
    for n,(ls,a) in data.items():
        for j,l in enumerate(ls):
            g=a[:,j];den=an*np.linalg.norm(g,axis=1)
            if (den<1e-20).any():raise ValueError('Zero selection gradient')
            scores.append((float(np.mean((g*anchor).sum(1)/den)),(n,l)))
    return sorted(scores,key=lambda x:(-x[0],x[1]))
def spaced_ranks(scores,ranks=(1,3,5)):
    spaced=[]
    for value,u in scores:
        if all(u[0]!=v[0] or abs(u[1]-v[1])>1 for _,v in spaced):spaced.append((value,u))
    if len(spaced)<max(ranks):raise ValueError('Insufficient nonadjacent source candidates')
    return [spaced[k-1][1] for k in ranks]

def neighborhood(current,pool,max_units=4):
    current=set(current);pool=set(pool);out=set()
    if len(current)<max_units:
        out.update(tuple(sorted(current|{u})) for u in pool-current)
    if len(current)>1:out.update(tuple(sorted(current-{v})) for v in current)
    out.update(tuple(sorted((current-{v})|{u})) for v in current for u in pool-current)
    return sorted(x for x in out if x and len({n for n,_ in x})<=3)
def proposals(fam,base,pool,g):
    neighbors=neighborhood(units(base),pool);resident={n for n,_ in units(base)}
    def priority(us):
        new=set(us)-set(units(base));return sum(J.prior(g,u,list(set(us)-{u})) for u in new)
    # Bounded distinct add/replace/remove proposals; weak candidates not hard-filtered by singleton ASR.
    chosen=[]
    for kind in ['replace','remove','add']:
        eligible=[u for u in neighbors if (len(u)==len(units(base)) if kind=='replace' else len(u)<len(units(base)) if kind=='remove' else len(u)>len(units(base)))]
        if eligible:chosen.append(max(eligible,key=lambda u:(priority(u),u)))
    return [condition(fam,u,'joint_local_move') for u in chosen]

def family_plan(fam,g):
    pending=[];policies={};fda=[];trace={}
    for n in C.POOLS[fam][:3]:
        todo,chosen,history=fda_model(fam,n);pending+=todo;trace[n]=history
        if 2 in chosen:policies[n]=chosen[2]['layers'][n]
        fda+=list(chosen.values())
    if pending:return 'FDA_CONDITIONAL',dedup(pending),[],trace
    # Method pilot, not nine full FDA4 ablations: deepen one predeclared best FDA2 backbone/family.
    representative=next(iter(best([c for c in fda if len(units(c))==2])[0]['layers']))
    todo,chosen,history=fda_model(fam,representative,goal=4)
    trace[representative]=history;fda+=list(chosen.values())
    if todo:return 'FDA_CONDITIONAL4',todo,[],trace
    # This development-only layer lock precedes proxy comparisons; it is NOT the final method lock.
    C.freeze(OUT/(fam+'_proxy_layer_policy.json'),dict(protocol=C.sha(POLICY),layers=policies,
        evidence=[c['id'] for c in fda if len(units(c))==2],role='fixed_layer_proxy_control_only'))
    pc=proxy_fixed(fam,policies);pending=register(pc,'proxy_fixed_layers')
    if pending:return 'PROXY_FIXED_LAYER',pending,[],trace
    data={};inputs=[]
    for n in C.POOLS[fam][:3]:
        ls,a,ii=gradient_matrix(n);data[n]=(ls,a);inputs+=ii
    ranks=alignment_scores(data,{n:C.REG[n]['expected_vision_depth'] for n in data})
    praf=condition(fam,spaced_ranks(ranks),'PRAF_inspired_PP_alignment')
    C.atomic(OUT/(fam+'_PRAF_selection.json'),dict(adaptation=True,original_PRAF_reproduction=False,
        state='cached_random_start_DI',objective='PP_spatial_prototype',anchor='equal_normalized_deepest_PP',
        inputs=inputs,ranked=ranks,selected=units(praf),condition=praf['id']))
    chosen_proxy=best(pc)[0];ind=[]
    for n in chosen_proxy['layers']:
        singles=[c for c in S.layer_conditions(fam) if n in c['layers']]
        ind.extend(u for c in best(singles,2) for u in units(c))
    independent=condition(fam,ind,'independent_proxy_then_layer')
    pending=register([praf,independent],'PRAF_and_independent')
    if pending:return 'PRAF_AND_INDEPENDENT',pending,[],trace
    pool=sorted((n,l) for n in data for l in data[n][0]) # no singleton-ASR filter
    singles=[c for c in S.layer_conditions(fam) if completed(c)]
    # A weak-but-source-diverse pair can start a separate beam path, bypassing strongest-single greed.
    weakest=units(best(singles,len(singles))[-1])[0]
    lowpair=min((uv for uv in itertools.combinations(pool,2) if weakest in uv),key=lambda uv:(J.pair_record(g,*uv)['CKA'],uv))
    weak=condition(fam,lowpair,'weak_pair_complementarity_control')
    pending=register([weak],'weak_pair_control')
    if pending:return 'WEAK_PAIR_CONTROL',pending,[],trace
    seeds=dedup(best(singles)+[weak,praf]+([independent] if len(units(independent))<=4 else []))
    greedy=best(singles)[0];beam=J.pareto_beam(seeds,2);allc=seeds[:];moves=[]
    for turn in range(2):
        candidates=dedup([c for b in dedup([greedy]+beam) for c in proposals(fam,b,pool,g)])
        pending=register(candidates,'joint_greedy_beam')
        moves.extend(dict(round=turn,parent=b['id'],candidate=c['id']) for b in dedup([greedy]+beam) for c in proposals(fam,b,pool,g))
        if pending:return 'JOINT_LOCAL_SEARCH',pending,[],dict(FDA=trace,joint_moves=moves)
        greedy=best([greedy]+proposals(fam,greedy,pool,g))[0]
        beam=J.pareto_beam(beam+candidates,2);allc+=candidates
    nominees=dedup([greedy]+beam+[independent,praf])
    C.atomic(OUT/(fam+'_method_results.json'),dict(complete=True,greedy=greedy['id'],beam=[c['id'] for c in beam],
        fixed_proxy_best=chosen_proxy['id'],independent=independent['id'],PRAF=praf['id'],
        nominees=[c['id'] for c in nominees],moves=moves))
    return 'READY',[],nominees,trace

def plan_next(g):
    identity=snapshot_identity();C.freeze(OUT/'source_lock.json',identity)
    pending=[];nominees={};phases={}
    for fam in C.VALIDATORS:
        phase,cc,nn,trace=family_plan(fam,g);pending+=cc;nominees[fam]=nn;phases[fam]=phase
        C.atomic(OUT/(fam+'_trace.json'),trace)
    C.atomic(OUT/'status.json',dict(complete=not pending,phases=phases,pending=[c['id'] for c in dedup(pending)],
        nominees={f:[c['id'] for c in cs] for f,cs in nominees.items()},final_target_feedback=False))
    return ('READY' if not pending else 'STAGE1_LITERATURE_METHODS'),dedup(pending)
def nominees(fam):
    result=C.read(OUT/(fam+'_method_results.json'),{})
    if not result.get('complete'):raise RuntimeError('Stage1 literature methods incomplete; no freeze/Stage2')
    registry=C.read(OUT/'registry.json',{})
    return [registry[cid]['condition'] for cid in result['nominees'] if cid in registry]

def report():
    registry=C.read(OUT/'registry.json',{});rows=[]
    for cid,e in registry.items():
        c=e['condition'];s=S.score(c) if completed(c) else None
        rows.append(dict(id=cid,family=c['family'],methods=e['methods'],models=len(c['layers']),taps=len(units(c)),
          status='COMPLETE' if s else 'PENDING',score=s))
    paired=[]
    for fam in C.VALIDATORS:
        trace=C.read(OUT/(fam+'_trace.json'),{})
        moves=trace.get('joint_moves',[]) if isinstance(trace,dict) else []
        result=C.read(OUT/(fam+'_method_results.json'),{})
        moves+=result.get('moves',[])
        for a,b in sorted({(m['parent'],m['candidate']) for m in moves}):
            if a not in registry or b not in registry:continue
            ca,cb=registry[a]['condition'],registry[b]['condition']
            if not completed(ca) or not completed(cb):continue
            aa={x['image_id']:x for x in C.read(C.eval_path(ca))['images']}
            bb={x['image_id']:x for x in C.read(C.eval_path(cb))['images']}
            if aa.keys()!=bb.keys():raise RuntimeError('Unpaired joint comparison')
            paired.append(dict(family=fam,parent=a,candidate=b,N=len(aa),
                gained=sum(bb[i]['target_hit'] and not aa[i]['target_hit'] for i in aa),
                lost=sum(aa[i]['target_hit'] and not bb[i]['target_hit'] for i in aa)))
    C.atomic(OUT/'results.json',dict(rows=rows,paired=paired,missing_are_zero=False,full_ablation_stage=3))
    return dict(conditions=len(rows),complete=sum(x['status']=='COMPLETE' for x in rows),status=C.read(OUT/'status.json',{}))
def integration_text(source):
    before=" if allc:return 'INTEGRATED_METHOD_PILOT',list({c['id']:c for c in allc}.values())\n"
    after=before+" from stage1_methods.planner import plan_next as literature_plan, nominees as literature_nominees\n literature_phase,literature_tasks=literature_plan(g)\n if literature_phase!='READY':return literature_phase,literature_tasks\n"
    frontier="frontier=pareto_beam(screen[fam],2)"
    if source.count(before)!=1 or source.count(frontier)!=1:raise RuntimeError('Owner source changed; inspect integration rather than blind patch')
    return source.replace(before,after).replace(frontier,"frontier=pareto_beam(screen[fam]+literature_nominees(fam),2)")
def prepare():
    import difflib
    source=(C.HERE/'combo_selection.py').read_text();patched=integration_text(source);compile(patched,'prepared_owner','exec')
    patch=''.join(difflib.unified_diff(source.splitlines(True),patched.splitlines(True),fromfile='a/'+str((C.HERE/'combo_selection.py').relative_to(C.REPO)),tofile='b/'+str((C.HERE/'combo_selection.py').relative_to(C.REPO))))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'owner_integration.patch').write_text(patch)
    C.atomic(OUT/'preparation.json',dict(status='PREPARED_NOT_ACTIVATED',protocol=C.sha(POLICY),
       active_source_unchanged=True,patch=str(OUT/'owner_integration.patch'),
       admission_owner='existing_stage12_scheduler',requires_reviewed_source_lock_amendment=True,
       no_new_GPU_jobs=True,no_commercial_calls=True,methods=['FDA_conditional2_4','proxy_fixed_layer7','PRAF_inspired_PP','independent','joint_greedy_beam_local_moves']))
    return C.read(OUT/'preparation.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','plan','status','report'],default='prepare',nargs='?');a=ap.parse_args()
    if a.mode=='prepare':result=prepare()
    elif a.mode=='status':result=C.read(OUT/'status.json',C.read(OUT/'preparation.json',{}))
    elif a.mode=='report':result=report()
    else:
        C.check_lock();g=J.source_geometry();phase,cc=plan_next(g);result=dict(phase=phase,pending=len(cc),no_jobs_launched=True)
    print(C.json.dumps(result,indent=2))
