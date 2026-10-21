"""Matched four-tap position controls; never mutate the frozen primary method."""
import csv,itertools,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C
import selection as S
import combo_selection as J
OUT=C.ROOT/'stage1_methods/fixed_four'
SCHEMES={'uniform_quartiles':[.25,.5,.75,1.],
         'shifted_quartiles':[.125,.375,.625,.875],
         'interior_swap':[.25,.5,.625,1.]}

def exact(fam,layers,label,split):
    c=C.make(fam,layers,label=label,split=split)
    for p in sorted((C.ROOT/'conditions').glob('*.json')):
        old=C.read(p)
        if C.scientific(old)==C.scientific(c):return old
    return c

def controls(fam,split='screen'):
    # Reuse actual baseline and preserve its order; no permutation cache equivalence.
    base=J.reuse_make(fam,S.uniform(C.POOLS[fam][:3]),'fixed4_uniform',split)
    out={'uniform_quartiles':base}
    for label,depths in SCHEMES.items():
        if label=='uniform_quartiles':continue
        layers={n:C.native(n,depths) for n in base['model_order']}
        if any(len(v)!=4 for v in layers.values()):raise ValueError('Need four distinct native taps/model')
        c=exact(fam,layers,'fixed4_'+label,split)
        a,b=C.scientific(base),C.scientific(c)
        if {k:v for k,v in a.items() if k!='layers'}!={k:v for k,v in b.items() if k!='layers'}:raise RuntimeError('Position control changed other science')
        out[label]=c
    return out

def completed(c):return C.attack_done(c) and C.eval_done(c)

def pair_outcomes(base,other):
    if not completed(base) or not completed(other):return None
    aa={x['image_id']:x for x in C.read(C.eval_path(base))['images']}
    bb={x['image_id']:x for x in C.read(C.eval_path(other))['images']}
    if aa.keys()!=bb.keys():raise RuntimeError('Unpaired fixed-four image cohort')
    for i in aa:
        if any(aa[i][k]!=bb[i][k] for k in ['source','target','transition','clean_label']):raise RuntimeError('Changed paired labels/clean prediction')
    gained=sum(bb[i]['target_hit'] and not aa[i]['target_hit'] for i in aa)
    lost=sum(aa[i]['target_hit'] and not bb[i]['target_hit'] for i in aa)
    return dict(gained=gained,lost=lost,net=gained-lost)

def plan():
    identity=dict(version='fixed_four_positions_v1',schemes=SCHEMES,source_sha256=C.sha(__file__),
        registry={n:C.REG[n] for fam in C.VALIDATORS for n in C.POOLS[fam][:3]},
        cohorts={s:C.sha(C.ROOT/'manifests'/(s+'.json')) for s in ['screen','explore']},
        method_selection='screen10_select_one_alternative_then_matched_explore30',primary_method_changes=False)
    C.freeze(OUT/'protocol_lock.json',identity)
    pending=[];rows=[]
    for fam in C.VALIDATORS:
        screen=controls(fam);jobs=[(label,c) for label,c in screen.items()]
        missing=[c for c in screen.values() if not completed(c)]
        pending+=missing
        if not missing:
            alternatives=[(label,c) for label,c in screen.items() if label!='uniform_quartiles']
            label,chosen=max(alternatives,key=lambda item:S.rank(item[1]))
            C.freeze(OUT/(fam+'_selection.json'),dict(rule='minimum_direction_hits_then_hits_then_cost',
                chosen_scheme=label,evidence=chosen['id'],selection_split='screen',not_commercial_feedback=True))
            explore=controls(fam,'explore')
            for key in ['uniform_quartiles',label]:
                c=explore[key];jobs.append((key,c))
                if not completed(c):pending.append(c)
        for label,c in jobs:
            done=completed(c);metric=S.score(c) if done else None
            baseline=screen['uniform_quartiles'] if c['split']=='screen' else controls(fam,'explore')['uniform_quartiles']
            pair=pair_outcomes(baseline,c) if done else None
            rows.append(dict(family=fam,split=c['split'],scheme=label,condition=c['id'],layers=c['layers'],
                proxy_count=3,taps_per_proxy=4,total_taps=12,complete=done,hits=metric['hits'] if metric else None,
                N=10 if c['split']=='screen' else 30,ASR=metric['hits']/metric['N'] if metric else None,
                directions=metric['directions'] if metric else None,paired=pair,
                seconds=metric['seconds'] if metric else None,backwards=metric['backwards'] if metric else None,
                peak_GiB=metric['peak'] if metric else None))
    pending=list({c['id']:c for c in pending}.values())
    result=dict(complete=not pending,pending=len(pending),rows=rows,scope='fixed_proxy_identity_and_four_taps_per_model_only_positions_change',
        original_main_and_commercial_freeze_unchanged=True,screen_selection_bias_disclosed=True)
    C.atomic(OUT/'results.json',result);C.atomic(C.ROOT/'reports/four_layer_positions.json',result)
    dest=C.ROOT/'reports/four_layer_positions.csv';tmp=dest.with_suffix('.csv.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    C.os.replace(tmp,dest)
    return pending,not pending

if __name__=='__main__':
    cc,done=plan();print(C.json.dumps(dict(pending=len(cc),complete=done,no_GPU_jobs_launched_by_planner=True),indent=2))
