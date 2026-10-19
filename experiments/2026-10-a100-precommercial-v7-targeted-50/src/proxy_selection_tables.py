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
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def complete_cell(row):
    n=int(row.get('denominator') or 0);hits=int(row.get('hits') or 0)
    if not 0<=hits<=n<=30:raise ValueError('Invalid cell numerator/denominator')
    return row.get('status')=='complete' and n==30


def collect(plan):
    table=[];directions=[];sources={}
    for item in plan:
        c=item['config'];path=OUT/item['source_csv']
        source=list(csv.DictReader(path.open())) if path.exists() else []
        if path.exists():sources[str(path.relative_to(OUT))]=sha(path)
        cells=[r for r in source if r['condition']==item['measurement_condition'] and r['target']==c['target']]
        if len({r['transition'] for r in cells})!=len(cells):raise ValueError('Duplicate directed measurement cell')
        per={r['transition']:r for r in cells}
        for tid in TIDS:
            r=per.get(tid,{});n=int(r.get('denominator') or 0);h=int(r.get('hits') or 0)
            complete_cell(r)
            directions.append(dict(id=c['id'],target=c['target'],slots=item['slots'],transition=tid,hits=h,denominator=n,ASR=h/n if n else None,status='complete' if complete_cell(r) else 'pending_or_partial',measurement_condition=item['measurement_condition'],source=item['source']))
        measured=sum(int(r.get('denominator') or 0) for r in cells)
        hits=sum(int(r.get('hits') or 0) for r in cells)
        complete=len(cells)==5 and all(complete_cell(r) for r in cells)
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
    return table,directions,sources


def prepare():
    plan=matrix();ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'planned_matrix.json',dict(status='predeclared_not_a_new_measurement',rows=plan,conditions=21,cells=105,image_evaluations=3150,loss_fixed=True,target_ranking_descriptive_only=True))
    table,directions,sources=collect(plan)
    writecsv(ROOT/'tables/proxy_selection.csv',table);writecsv(ROOT/'tables/directed_results.csv',directions)
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
