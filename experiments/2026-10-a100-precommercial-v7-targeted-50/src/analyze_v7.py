#!/usr/bin/env python3
"""Post-freeze geometry. These features never enter method selection."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from v7_common import OUT, TARGETS, REGISTRY, TIDS, fingerprint, sha, atomic_json
from vision_features import VisionProxy
from finalize_v7 import writecsv


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def cka(x,y):
    a=x-x.mean(0);b=y-y.mean(0)
    return float(np.linalg.norm(a.T@b)**2/max(np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b),1e-20))


def rsa(x,y):
    idx=np.triu_indices(len(x),1);a=(unit(x)@unit(x).T)[idx];b=(unit(y)@unit(y).T)[idx]
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(spearmanr(a,b).statistic)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method-lock',type=Path,default=OUT/'audits/method_lock.json');parser.add_argument('--analysis-dir',type=Path,default=OUT/'analysis');parser.add_argument('--results-csv',type=Path,default=OUT/'reports/target_direction_results.csv');args=parser.parse_args()
    A=args.analysis_dir;A.mkdir(parents=True,exist_ok=True)
    lock=json.loads(args.method_lock.read_text())
    states=[]
    for c in lock['allowed_test_conditions']:
        for p in sorted((OUT/'attacks/test'/c).glob('T*/batch_*/state.json')):
            d=json.loads(p.read_text());states.append((c,d))
    # Reuse frozen image features by content hash, model revision and pooling.
    proxy_names=sorted(set(n for c in lock.get('test_conditions',lock.get('configs',[])) for n in c['models']))
    all_names=list(TARGETS)+proxy_names
    summaries=[];image_rows=[];variance=[];features={}
    for name in all_names:
        m=VisionProxy(name,role='target_posthoc' if name in TARGETS else 'proxy_posthoc');m.to('cuda');m.prepare_bank()
        try:
            refs=m.bank[m.layers[0]][0].cpu().numpy();centers=unit(unit(refs).mean(1))
            for label,r in enumerate(refs,1):
                r=unit(r);center=unit(r.mean(0,keepdims=True))[0];sing=np.linalg.svd(r-r.mean(0),compute_uv=False);eig=sing**2/max(1,len(r)-1);p=eig/eig.sum() if eig.sum()>1e-20 else np.zeros_like(eig);nz=p>0
                variance.append(dict(model=name,label=label,dispersion=float((1-r@center).mean()),covariance_trace=float(eig.sum()),effective_rank=float(np.exp(-(p[nz]*np.log(p[nz])).sum())) if nz.any() else 0.))
            def embed(paths):
                from primary_ml_cka.artifacts.png import load_png_tensor
                zs=[]
                for path in paths:
                    key=fingerprint([sha(OUT/path),REGISTRY[name],m.layers,'V6_spatial_mean'])
                    cache=A/'embeddings'/name.replace('/','_')/f'{key}.npy'
                    if cache.exists():z=np.load(cache)
                    else:
                        with torch.inference_mode():z=m.features(load_png_tensor(OUT/path)[None].cuda())[m.layers[0]][0][0].float().cpu().numpy()
                        cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_suffix('.tmp.npy');np.save(tmp,z);tmp.replace(cache)
                    zs.append(z)
                return np.stack(zs)
            for condition,state in states:
                clean=embed(state['clean_paths']);adv=embed(state['adv_paths'])
                for iid,c,a in zip(state['image_ids'],clean,adv):features[condition,state['transition_id'],iid,name]=(c,a)
            for condition in lock['allowed_test_conditions']:
                for tid in TIDS:
                    items=[(iid,ca) for (c,t,iid,n),ca in features.items() if c==condition and t==tid and n==name]
                    items.sort(key=lambda x:x[0])
                    if len(items)!=30:continue
                    source=int(tid[1:]);target=source+1
                    clean=unit(np.stack([ca[0] for _,ca in items]));adv=unit(np.stack([ca[1] for _,ca in items]))
                    cs=clean@centers.T;ads=adv@centers.T
                    dp=ads[:,target-1]-cs[:,target-1];dpush=cs[:,source-1]-ads[:,source-1]
                    clean_margin=cs[:,target-1]-cs[:,source-1];adv_margin=ads[:,target-1]-ads[:,source-1]
                    summary=dict(condition=condition,model=name,transition=tid,N=30,delta_pull=float(dp.mean()),delta_push=float(dpush.mean()),DeltaR=float((dp+dpush).mean()),median_DeltaR=float(np.median(dp+dpush)),clean_margin=float(clean_margin.mean()),margin_change=float((adv_margin-clean_margin).mean()),prototype_distance=float(1-centers[source-1]@centers[target-1]),clean_adv_CKA=cka(clean,adv),clean_adv_RSA=rsa(clean,adv))
                    summaries.append(summary)
                    for i,(iid,_) in enumerate(items):image_rows.append(dict(condition=condition,model=name,transition=tid,image_id=iid,delta_pull=float(dp[i]),delta_push=float(dpush[i]),DeltaR=float(dp[i]+dpush[i]),clean_margin=float(clean_margin[i]),adversarial_margin=float(adv_margin[i])))
                    # Joint reference + clean + adversarial set: one fit per plot.
                    # This forward-only protocol has no B->A cohort: do not invent it.
                    if name in TARGETS and condition==lock['allowed_test_conditions'][0]:
                        groups=[unit(refs[source-1]),unit(refs[target-1]),clean,adv]
                        X=np.concatenate(groups);labels=np.repeat(['source refs','target refs','clean','adversarial'],list(map(len,groups)))
                        pca=PCA(n_components=2,random_state=42);xy=pca.fit_transform(X)
                        reduced=PCA(n_components=min(40,len(X)-1,X.shape[1]),random_state=42).fit_transform(X)
                        tsne=TSNE(n_components=2,perplexity=30,random_state=42,init='pca',learning_rate='auto').fit_transform(reduced)
                        for mode,coords in [('pca',xy),('tsne',tsne)]:
                            dst=OUT/'analysis'/mode/name.replace('/','_')/tid;dst.mkdir(parents=True,exist_ok=True)
                            writecsv(dst/'coordinates.csv',[dict(group=str(g),x=float(v[0]),y=float(v[1])) for g,v in zip(labels,coords)])
                            fig,ax=plt.subplots(figsize=(6,5))
                            for g in dict.fromkeys(labels):ax.scatter(*coords[labels==g].T,s=13,alpha=.7,label=g)
                            if mode=='pca':
                                offset=len(groups[0])+len(groups[1])
                                for i in range(30):ax.annotate('',coords[offset+30+i],coords[offset+i],arrowprops=dict(arrowstyle='->',alpha=.25,lw=.5))
                            ax.legend(fontsize=8);ax.set_title(f'{name} {tid}: joint {mode.upper()}');fig.tight_layout();fig.savefig(dst/'plot.png',dpi=160);plt.close(fig)
            writecsv(A/'representation_shift/summary.csv',summaries)
            writecsv(A/'representation_shift/per_image.csv',image_rows)
            writecsv(A/'variance/class_variance.csv',variance)
        finally:m.close()
    cross=[]
    for c in lock['allowed_test_conditions']:
        for target in TARGETS:
            for tid in TIDS:
                for proxy in proxy_names:
                    ids=sorted(iid for (cond,t,iid,n) in features if cond==c and t==tid and n==target and (c,tid,iid,proxy) in features)
                    if len(ids)!=30:continue
                    for kind,index in [('clean',0),('adversarial',1)]:
                        x=np.stack([features[c,tid,i,proxy][index] for i in ids]);y=np.stack([features[c,tid,i,target][index] for i in ids])
                        cross.append(dict(condition=c,target=target,proxy=proxy,transition=tid,kind=kind,CKA=cka(x,y),RSA=rsa(x,y)))
    writecsv(A/'cross_model/cka_rsa.csv',cross)
    # Correlations are exploratory aggregate associations, not a tuning criterion.
    import pandas as pd
    table=pd.read_csv(args.results_csv)
    geom=pd.DataFrame([r for r in summaries if r['model'] in TARGETS]).rename(columns={'model':'target'})
    joined=table.merge(geom,on=['condition','target','transition'],validate='one_to_one')
    joined.to_csv(A/'measurement_table.csv',index=False)
    cols=['ASR','DeltaR','delta_pull','delta_push','prototype_distance','clean_margin','margin_change','clean_adv_CKA','clean_adv_RSA']
    corr=joined[cols].corr(method='spearman');dst=A/'correlations';dst.mkdir(parents=True,exist_ok=True);corr.to_csv(dst/'spearman.csv')
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(cols)),cols,rotation=65,ha='right');ax.set_yticks(range(len(cols)),cols);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(dst/'heatmap.png',dpi=160);plt.close(fig)
    atomic_json(A/'posthoc_analysis.json',dict(status='complete',method_lock_sha256=sha(args.method_lock),target_feedback_for_attack=False,geometry_rows=len(summaries),cross_model_rows=len(cross),asymmetry='not_estimable_without_reverse_direction_runs',tsne='qualitative; distances not quantitative evidence'))

    if A==OUT/'analysis':atomic_json(OUT/'audits/posthoc_analysis.json',json.loads((A/'posthoc_analysis.json').read_text()))

if __name__=='__main__':main()
