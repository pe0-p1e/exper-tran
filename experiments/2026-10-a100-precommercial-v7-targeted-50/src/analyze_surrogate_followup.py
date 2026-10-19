"""CPU post-hoc DeltaR, geometry and joint projections from frozen feature caches."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from proxy_study_protocol import OUT,sha,atomic_json,fingerprint
from unified_research_plan import RESEARCH,conditions,VALIDATORS,condition_name,cohort
from cached_reference_geometry import unit,gram,cka_from_grams
from finalize_v7 import writecsv


def rsa(a,b):
    ix=np.triu_indices(len(a),1)
    x=(1-unit(a)@unit(a).T)[ix];y=(1-unit(b)@unit(b).T)[ix]
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    value=spearmanr(x,y).statistic
    return float(value) if np.isfinite(value) else None


def main():
    dest=RESEARCH/'analysis';dest.mkdir(parents=True,exist_ok=True)
    measures=[];cross=[];image_rows=[];sources=[]
    pilot=fingerprint(cohort())[:12]
    for c in conditions():
        models={}
        for path in sorted((RESEARCH/'embeddings'/c['id']).glob('*.json')):
            meta=json.loads(path.read_text());npz=Path(meta['npz'])
            if sha(npz)!=meta['npz_sha256']:raise RuntimeError('Changed frozen feature cache')
            with np.load(npz,allow_pickle=False) as a:data={k:a[k] for k in a.files}
            if len(set(data['image_ids']))!=10:raise RuntimeError('Expected10 unique development images')
            models[meta['model']]=(meta,data);sources.append(dict(condition=c['id'],model=meta['model'],metadata_sha256=sha(path),npz_sha256=meta['npz_sha256']))
            native=OUT/'focused_optimization/native_validation'/pilot/condition_name(c)/(meta['model'].replace('/','_')+'.json')
            native_rows=json.loads(native.read_text())['images'] if native.exists() else []
            hits={(r['transition_id'],r['image_id']):r['target_hit'] for r in native_rows}
            for l in meta['layers']:
                clean=unit(data[f'clean_{l}']);adv=unit(data[f'adv_{l}']);refs=unit(data[f'refs_{l}']);centers=unit(refs.mean(1))
                s=data['source_labels']-1;t=data['target_labels']-1
                cs=clean@centers.T;ads=adv@centers.T;ix=np.arange(10)
                dp=ads[ix,t]-cs[ix,t];push=cs[ix,s]-ads[ix,s]
                margin0=cs[ix,t]-cs[ix,s];margin1=ads[ix,t]-ads[ix,s]
                for i,iid in enumerate(data['image_ids']):
                    image_rows.append(dict(condition=c['id'],model=meta['model'],layer=l,image_id=str(iid),transition=str(data['transitions'][i]),
                        delta_pull=float(dp[i]),delta_push=float(push[i]),DeltaR=float(dp[i]+push[i]),clean_margin=float(margin0[i]),adv_margin=float(margin1[i])))
                for tid in sorted(set(data['transitions'])):
                    mask=data['transitions']==tid;indices=np.flatnonzero(mask);source=int(data['source_labels'][indices[0]])-1;target=source+1
                    sourcevar=float((1-refs[source]@centers[source]).mean());targetvar=float((1-refs[target]@centers[target]).mean())
                    keys=[(str(tid),str(data['image_ids'][i])) for i in indices]
                    observed=all(k in hits for k in keys)
                    measures.append(dict(condition=c['id'],model=meta['model'],role='held_out_surrogate' if meta['model'] in VALIDATORS else 'attack_proxy',
                        layer=l,normalized_depth=(meta['layers'].index(l)+1)/len(meta['layers']),transition=str(tid),N=int(mask.sum()),
                        native_hits=sum(hits[k] for k in keys) if observed else None,native_ASR=sum(hits[k] for k in keys)/len(keys) if observed else None,
                        delta_pull=float(dp[mask].mean()),delta_push=float(push[mask].mean()),DeltaR=float((dp+push)[mask].mean()),
                        clean_margin=float(margin0[mask].mean()),margin_change=float((margin1-margin0)[mask].mean()),
                        prototype_distance=float(1-centers[source]@centers[target]),source_variance=sourcevar,target_variance=targetvar,
                        clean_adv_CKA=cka_from_grams(gram(clean[mask]),gram(adv[mask])),clean_adv_RSA=rsa(clean[mask],adv[mask])))
                    # With N=2, per-direction CKA is degenerate and RSA undefined.
                    # Keep these fields explicitly unavailable for inference.
                    measures[-1]['clean_adv_CKA']=None;measures[-1]['clean_adv_RSA']=None
                    if c['id']=='U00' and l==meta['layers'][-1]:
                        folder=dest/'joint'/meta['model'].replace('/','_')/str(tid)
                        signature=fingerprint([meta['npz_sha256'],int(l),str(tid),'joint-v1'])
                        if (folder/'fit.json').exists() and json.loads((folder/'fit.json').read_text()).get('signature')==signature:continue
                        groups=[refs[source],refs[target],clean[mask],adv[mask]]
                        x=np.concatenate(groups);labels=np.repeat(['source references','target references','clean','adversarial'],[len(g) for g in groups])
                        pca=PCA(n_components=min(40,len(x)-1,x.shape[1]),svd_solver='randomized',random_state=42);reduced=pca.fit_transform(x)
                        coords={'pca':reduced[:,:2],'tsne':TSNE(n_components=2,perplexity=30,init='pca',random_state=42,learning_rate='auto').fit_transform(reduced)}
                        folder.mkdir(parents=True,exist_ok=True)
                        for kind,xy in coords.items():
                            writecsv(folder/(kind+'_coordinates.csv'),[dict(group=str(g),x=float(v[0]),y=float(v[1])) for g,v in zip(labels,xy)])
                            fig,ax=plt.subplots(figsize=(6,5))
                            for group in dict.fromkeys(labels):
                                idx=labels==group;ax.scatter(xy[idx,0],xy[idx,1],label=group,s=24 if group in ['clean','adversarial'] else 12,alpha=.75)
                            if kind=='pca':
                                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})');ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                                offset=96
                                for i in range(2):ax.annotate('',xy[offset+2+i],xy[offset+i],arrowprops=dict(arrowstyle='->',alpha=.5))
                            ax.set_title(f"U00 {meta['model']} {tid}: joint {kind.upper()}");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/(kind+'.png'),dpi=150);plt.close(fig)
                        atomic_json(folder/'fit.json',dict(signature=signature,one_fit_per_method=True,groups=[len(g) for g in groups],seed=42,reverse_direction_absent=True))
        for left,right in itertools.combinations(models,2):
            lm,ld=models[left];rm,rd=models[right]
            li={str(v):i for i,v in enumerate(ld['image_ids'])};ri={str(v):i for i,v in enumerate(rd['image_ids'])}
            if li.keys()!=ri.keys():raise RuntimeError('Cross-model image mismatch')
            ids=sorted(li)
            for side in ['clean','adv']:
                for pos,(ll,rl) in enumerate(zip(lm['layers'],rm['layers'])):
                    x=ld[f'{side}_{ll}'][[li[i] for i in ids]];y=rd[f'{side}_{rl}'][[ri[i] for i in ids]]
                    cross.append(dict(condition=c['id'],left_model=left,right_model=right,normalized_depth=(pos+1)/4,kind=side,N=10,
                                      CKA=cka_from_grams(gram(unit(x)),gram(unit(y))),RSA=rsa(x,y)))
    writecsv(dest/'representation_shift.csv',measures);writecsv(dest/'per_image.csv',image_rows);writecsv(dest/'cross_model_cka_rsa.csv',cross)
    import pandas as pd
    frame=pd.DataFrame([r for r in measures if r['role']=='held_out_surrogate' and r['native_ASR'] is not None])
    if not frame.empty:
        cols=['native_ASR','DeltaR','source_variance','target_variance','prototype_distance','clean_margin','margin_change']
        # Collapse layers to prevent four copies of one outcome per observation.
        table=frame.groupby(['condition','model','transition'],as_index=False)[cols].mean()
        table.to_csv(dest/'correlation_input.csv',index=False)
        corr=table[cols].corr(method='spearman');corr.to_csv(dest/'spearman.csv')
        fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(cols)),cols,rotation=65,ha='right');ax.set_yticks(range(len(cols)),cols);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(dest/'spearman.png',dpi=150);plt.close(fig)
    atomic_json(dest/'status.json',dict(status='PARTIAL' if len({s['condition'] for s in sources})<11 else 'COMPLETE_CACHE_ANALYSIS',
          embedding_sources=sources,measurement_rows=len(measures),cross_model_rows=len(cross),model_inference_calls=0,
          per_direction_N=2,per_direction_CKA_RSA='N/A: N2 is degenerate/undefined; cross-model metrics use all10 aligned images',
          tsne='qualitative; original-dimensional cosine geometry supplies quantitative measurements',
          correlations='exploratory dependent repeated-image observations; no causal or significance claims',
          asymmetry='not estimable: no reverse-direction cohort',commercial_or_final_target_feedback=False))
    print('Analyzed',len(sources),'model-condition caches;',len(measures),'geometry rows.',flush=True)


if __name__=='__main__':main()
