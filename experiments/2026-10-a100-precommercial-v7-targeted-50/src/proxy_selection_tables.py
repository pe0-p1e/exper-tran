"""CPU-only seven-subset matrix and provenance-backed proxy ablation tables."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from v7_common import OUT,EXP,REGISTRY,TIDS,fingerprint,atomic_json,sha
from family_protocol import corrected_conditions,condition_name,require_excluded_family

ROOT=OUT/'proxy_selection'


def matrix():
    framework=yaml.safe_load((EXP/'latest_experiment_framework.yaml').read_text())['proxy_selection']
    if len(framework['target_slots'])!=3:raise ValueError('This study requires exactly three predeclared open-weight targets')
    core=corrected_conditions();old=json.loads((OUT/'ten_hour_research/selected_method_lock.json').read_text())
    rows=[]
    for prefix,(target,slots) in zip(['PSQ','PSI','PSG'],framework['target_slots'].items()):
        for subset in framework['subset_order']:
            models=[slots[s] for s in subset]
            require_excluded_family(models,REGISTRY[target]['family'],REGISTRY)
            c=dict(id=prefix+'_'+''.join(subset),target=target,models=models,method='A3',options={'patch_weight':.35},di=True,fusion='equal',steps=50,seed=42,restarts=1,kind='proxy_selection')
            reuse=next((x for x in core if x['target']==target and x['models']==models and x['kind'] in ('primary_multi','matched_single')),None)
            if reuse:
                source='corrected_family_disjoint';measurement=condition_name(reuse)
                csvpath=OUT/'family_disjoint_recovery/results/target_direction_results.csv'
            elif target=='Gemma 4 31B-it' and (models==old['configs'][0]['models'] or models==old['configs'][1]['models']):
                index=0 if models==old['configs'][0]['models'] else 1
                source='historical_valid_Gemma_cross_family';measurement=old['allowed_test_conditions'][index]
                csvpath=OUT/'ten_hour_research/final_target_results.csv'
            else:
                source='new_subset_measurement';measurement=condition_name(c)
                csvpath=ROOT/'results/target_direction_results.csv'
            rows.append(dict(config=c,slots='+'.join(subset),measurement_condition=measurement,source=source,source_csv=str(csvpath.relative_to(OUT))))
    assert len(rows)==21
    return rows


def writecsv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    tmp.replace(path)


def complete_cell(row):
    n=int(row.get('denominator') or 0);hits=int(row.get('hits') or 0)
    if not 0<=hits<=n<=30:raise ValueError('Invalid cell numerator/denominator')
    return row.get('status')=='complete' and n==30


def feature_similarity(x,y):
    """Linear CKA via sample Gram matrices; cosine RSA on matched rows."""
    from scipy.stats import spearmanr
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)!=len(y) or len(x)<3:raise ValueError('Matched embedding rows required')
    xc=x-x.mean(0);yc=y-y.mean(0);a=xc@xc.T;b=yc@yc.T
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    cka=float(np.sum(a*b)/denom) if denom>1e-20 else None
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
    ii=np.triu_indices(len(x),1);u=(x@x.T)[ii];v=(y@y.T)[ii]
    rsa=float(spearmanr(u,v).statistic) if np.std(u)>1e-12 and np.std(v)>1e-12 else None
    return cka,rsa


def add_clean_geometry(table,plan):
    """Reuse only content/revision/layer-matched caches; never run inference here."""
    anchors={r['config']['target']:r for r in plan if len(r['config']['models'])==3}
    loaded={}
    for row in table:
        if row['K']==1:row['feature_status']='not_applicable_single';continue
        target=row['target'];anchor=anchors[target]
        pairs=[]
        for p in sorted((OUT/'attacks/test'/anchor['measurement_condition']).glob('T*/batch_*/state.json')):
            d=json.loads(p.read_text());pairs+=list(zip(d['image_ids'],d['clean_paths']))
        pairs.sort()
        if len(pairs)!=150 or len({x[0] for x in pairs})!=150:continue
        banks=[]
        for name in anchor['config']['models']:
            if name not in row['proxies'].split('+'):continue
            cachekey=(target,name)
            if cachekey not in loaded:
                vectors=[]
                for _,rel in pairs:
                    depth=REGISTRY[name]['expected_vision_depth']
                    key=fingerprint([sha(OUT/rel),REGISTRY[name],[depth-1],'V6_spatial_mean'])
                    candidates=[root/'embeddings'/name.replace('/','_')/(key+'.npy') for root in [ROOT/'analysis',OUT/'family_disjoint_recovery/analysis',OUT/'ten_hour_research/analysis']]
                    file=next((p for p in candidates if p.exists()),None)
                    if file is None:break
                    vector=np.load(file,allow_pickle=False)
                    if not np.isfinite(vector).all():raise ValueError('Nonfinite cached embedding')
                    vectors.append(vector)
                loaded[cachekey]=np.stack(vectors) if len(vectors)==150 else None
            banks.append(loaded[cachekey])
        if len(banks)!=row['K'] or any(x is None for x in banks):continue
        scores=[feature_similarity(banks[i],banks[j]) for i in range(len(banks)) for j in range(i+1,len(banks))]
        ckas=[x for x,_ in scores if x is not None];rsas=[x for _,x in scores if x is not None]
        row['mean_clean_CKA']=float(np.mean(ckas)) if ckas else None
        row['mean_clean_RSA']=float(np.mean(rsas)) if rsas else None
        row['feature_status']='complete' if len(ckas)==len(scores) and len(rsas)==len(scores) else 'degenerate_features'


def collect(plan):
    table=[];directions=[];sources={}
    for item in plan:
        c=item['config'];path=OUT/item['source_csv']
        source=list(csv.DictReader(path.open())) if path.exists() else []
        if path.exists():sources[str(path.relative_to(OUT))]=sha(path)
        cells=[r for r in source if r['condition']==item['measurement_condition'] and r['target']==c['target']]
        if len({r['transition'] for r in cells})!=len(cells):raise ValueError('Duplicate directed measurement cell')
        per={r['transition']:r for r in cells}
        if not set(per)<=set(TIDS):raise ValueError('Unexpected direction in subset measurements')
        for tid in TIDS:
            r=per.get(tid,{});n=int(r.get('denominator') or 0);h=int(r.get('hits') or 0)
            complete_cell(r)
            directions.append(dict(id=c['id'],target=c['target'],slots=item['slots'],transition=tid,hits=h,denominator=n,ASR=h/n if n else None,status='complete' if complete_cell(r) else 'pending_or_partial',measurement_condition=item['measurement_condition'],source=item['source']))
        measured=sum(int(r.get('denominator') or 0) for r in cells)
        hits=sum(int(r.get('hits') or 0) for r in cells)
        complete=set(per)==set(TIDS) and all(complete_cell(r) for r in cells)
        diagnostics=[]
        for file in (OUT/'attacks/test'/item['measurement_condition']).glob('T*/batch_*/state.json'):
            d=json.loads(file.read_text());require_excluded_family(d['models'],REGISTRY[c['target']]['family'],REGISTRY)
            if d['models']!=c['models']:raise ValueError('Reused attack proxy identity mismatch')
            if d['steps']!=50 or d['seed']!=42 or d['restarts']!=1 or d['target_access']:raise ValueError('Incompatible attack budget/isolation')
            if d['config'].get('patch_weight')!=.35 or not d['config'].get('input_diversity'):raise ValueError('Incompatible reused loss/DI')
            diagnostics.append(d)
        cos=[];cancellation=[]
        for d in diagnostics:
            for history in d['trace']:
                for step in history:
                    cos.extend(np.asarray(step['pairwise_cosines']).ravel().tolist())
                    cancellation.extend(step['cancellation_ratio'])
        table.append(dict(id=c['id'],target=c['target'],slots=item['slots'],proxies='+'.join(c['models']),K=len(c['models']),hits=hits,denominator=measured,ASR=hits/150 if complete else None,status='complete' if complete else 'partial' if measured else 'pending',gain_vs_best_member_single=None,gain_vs_best_pool_single=None,mean_pairwise_gradient_cosine=float(np.mean(cos)) if cos else None,negative_pairwise_cosine_fraction=float(np.mean(np.asarray(cos)<0)) if cos else None,mean_equal_mean_cancellation=float(np.mean(cancellation)) if cancellation else None,diagnostic_images=sum(len(d['image_ids']) for d in diagnostics),attack_seconds=sum(d['runtime_seconds'] for d in diagnostics),peak_reserved_gib=max((d['peak_reserved_gib'] for d in diagnostics),default=None),mean_clean_CKA=None,mean_clean_RSA=None,feature_status='not_yet_measured',measurement_condition=item['measurement_condition'],source=item['source']))
    for row in table:
        singles=[x for x in table if x['target']==row['target'] and x['K']==1]
        members=[x for x in singles if x['slots'] in row['slots'].split('+')]
        if row['status']=='complete':
            if len(members)==row['K'] and all(x['status']=='complete' for x in members):row['gain_vs_best_member_single']=row['ASR']-max(x['ASR'] for x in members)
            if len(singles)==3 and all(x['status']=='complete' for x in singles):row['gain_vs_best_pool_single']=row['ASR']-max(x['ASR'] for x in singles)
    add_clean_geometry(table,plan)
    return table,directions,sources


def paired_interval(groups,seed=42,draws=2000):
    if len(groups)!=5 or any(len(g)!=30 for g in groups):raise ValueError('Five matched30-image groups required')
    rng=np.random.default_rng(seed)
    groups=[np.asarray(g,dtype=float) for g in groups]
    boot=np.stack([g[rng.integers(0,30,(draws,30))].mean(1) for g in groups]).mean(0)
    return float(np.concatenate(groups).mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def paired_comparisons(plan,table):
    masks={};result=[]
    for item in plan:
        c=item['config']
        for tid in TIDS:
            rows=[]
            for file in (OUT/'evaluations'/item['measurement_condition']/c['target'].replace('/','_')/tid).glob('batch_*/metrics.json'):
                rows+=json.loads(file.read_text())['images']
            if len(rows)==30 and len({r['image_id'] for r in rows})==30:masks[c['id'],tid]={r['image_id']:int(r['target_hit']) for r in rows}
    for row in table:
        if row['K']<2 or row['status']!='complete':continue
        members=[r for r in table if r['target']==row['target'] and r['K']==1 and r['slots'] in row['slots'].split('+')]
        if len(members)!=row['K'] or any(r['status']!='complete' for r in members):continue
        best=max(members,key=lambda r:(r['ASR'],r['id']))
        groups=[]
        for tid in TIDS:
            a=masks.get((row['id'],tid),{});b=masks.get((best['id'],tid),{})
            if len(a)!=30 or set(a)!=set(b):break
            groups.append([a[i]-b[i] for i in sorted(a)])
        if len(groups)==5:
            mean,lo,hi=paired_interval(groups)
            result.append(dict(target=row['target'],ensemble=row['id'],best_observed_member_single=best['id'],N=150,ASR_difference=mean,CI_low=lo,CI_high=hi,interpretation='descriptive paired contrast; not target-guided method selection'))
    return result


def prepare(plan=None):
    plan=matrix() if plan is None else plan;ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'planned_matrix.json',dict(status='predeclared_not_a_new_measurement',rows=plan,conditions=21,cells=105,image_evaluations=3150,loss_fixed=True,target_ranking_descriptive_only=True))
    table,directions,sources=collect(plan)
    writecsv(ROOT/'tables/proxy_selection.csv',table);writecsv(ROOT/'tables/directed_results.csv',directions)
    writecsv(ROOT/'tables/paired_comparisons.csv',paired_comparisons(plan,table))
    atomic_json(ROOT/'tables/source_hashes.json',sources)
    lines=['# Multi-proxy selection ablation','','V/A/B use target-specific family-disjoint proxy pools in LATEST_EXPERIMENT_FRAMEWORK.md. Loss, DI, layers and attack budget are fixed. Pending/partial cells are not zero.','', '| Target | Subset | K | Hits/N | ASR | Status |','|---|---|---|---|---|---|']
    for row in table:
        score=f"{row['ASR']:.1%}" if row['ASR'] is not None else 'pending'
        lines.append(f"| {row['target']} | {row['slots']} | {row['K']} | {row['hits']}/{row['denominator']} | {score} | {row['status']} |")
    lines+=['','Gradient metrics may cover only completed attack batches; diagnostic_images records that scope. Negative pairwise cosine fraction is not pixelwise sign disagreement. CKA/RSA are pending until matching clean embeddings are measured. Reused historical results retain their source fingerprint. Raw scores can describe proxy selection but cannot tune another attack on the same evaluated target.']
    (ROOT/'tables/proxy_selection.md').write_text('\n'.join(lines)+'\n')
    return table


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();prepare()
