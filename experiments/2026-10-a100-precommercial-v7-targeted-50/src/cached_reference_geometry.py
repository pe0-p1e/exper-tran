"""CPU-only geometry of exact existing reference caches; never model inference."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proxy_study_protocol import OUT, ROOT, atomic_json, sha, fingerprint
from supplement_registry import registry
from finalize_v7 import writecsv

DEST=ROOT/'mechanism_reference'


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def gram(x):
    a=np.asarray(x,dtype=np.float64);a=a-a.mean(0,keepdims=True)
    return a@a.T


def cka_from_grams(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/d) if d>1e-20 else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--no-plots',action='store_true');args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    refs=[OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl' for i in range(1,11)]
    rows=[[json.loads(s) for s in p.read_text().splitlines() if s.strip()] for p in refs]
    ids=[r['image_id'] for group in rows for r in group]
    if any(len(group)!=48 for group in rows) or len(set(ids))!=480:raise RuntimeError('Invalid shared reference cohort')
    hashes=[sha(p) for p in refs];specs=registry();models={};sources=[];variance=[];layer_rows=[];missing=[]
    for name,spec in specs.items():
        depth=spec['expected_vision_depth'];layer_sets=[[depth-1],sorted({max(0,math.ceil(f*depth)-1) for f in [.25,.5,.75,1.]})]
        found={}
        for layers in layer_sets:
            key=fingerprint(dict(model=spec,refs=hashes,layers=layers,k=8,version='v7_features_2_v6_pooling'))
            path=OUT/'cache/reference_features'/f'{key}.npz'
            if not path.exists():continue
            with np.load(path,allow_pickle=False) as data:
                for layer in layers:
                    z=np.asarray(data[f'z_{layer}'],dtype=np.float64)
                    if z.shape[:2]!=(10,48) or not np.isfinite(z).all():raise RuntimeError('Invalid cached reference bank')
                    if layer in found and not np.allclose(found[layer],z,rtol=1e-5,atol=1e-6):
                        raise RuntimeError('Same-block reference caches disagree')
                    found[layer]=z
            sources.append(dict(model=name,revision=spec['revision'],layers=layers,path=str(path),sha256=sha(path)))
        if depth-1 not in found:missing.append(name);continue
        print('[reference geometry]',name,sorted(found),flush=True)
        deep=unit(found[depth-1]);flat=deep.reshape(480,-1)
        models[name]=dict(x=flat,gram=gram(flat),residual_gram=gram((deep-deep.mean(1,keepdims=True)).reshape(480,-1)))
        layer_grams={l:gram(unit(z).reshape(480,-1)) for l,z in found.items()}
        for l,r in itertools.combinations(sorted(found),2):
            layer_rows.append(dict(model=name,left_layer=l,right_layer=r,reference_N=480,
                                  CKA=cka_from_grams(layer_grams[l],layer_grams[r]),scope='same-image reference layers only'))
        for label,z in enumerate(deep,1):
            center=unit(z.mean(0,keepdims=True))[0];centered=z-z.mean(0,keepdims=True)
            eigen=np.linalg.eigvalsh(centered@centered.T/47).clip(0,None)
            total=eigen.sum();probs=eigen[eigen>max(total*1e-12,1e-15)]/total if total>1e-20 else np.array([])
            variance.append(dict(model=name,layer=depth-1,class_label=label,N=48,
                cosine_dispersion=float((1-z@center).mean()),covariance_trace=float(total),
                effective_rank=float(np.exp(-(probs*np.log(probs)).sum())) if len(probs) else 0.))
        if not args.no_plots:
            labels=np.repeat(np.arange(1,11),48)
            pca=PCA(n_components=min(40,flat.shape[1],479),svd_solver='randomized',random_state=42)
            reduced=pca.fit_transform(flat)
            coords={'pca':reduced[:,:2], 'tsne':TSNE(n_components=2,perplexity=30,init='pca',random_state=42,learning_rate='auto').fit_transform(reduced)}
            for kind,xy in coords.items():
                dest=DEST/kind/name.replace('/','_');dest.mkdir(parents=True,exist_ok=True)
                writecsv(dest/'coordinates.csv',[dict(image_id=iid,class_label=int(label),x=float(v[0]),y=float(v[1])) for iid,label,v in zip(ids,labels,xy)])
                fig,ax=plt.subplots(figsize=(7,5))
                for label in range(1,11):
                    part=xy[labels==label];ax.scatter(part[:,0],part[:,1],s=10,alpha=.65,label=str(label))
                ax.set_title(f'{name}: joint reference {kind.upper()} (no attacks)')
                ax.legend(title='Class',ncol=2,fontsize=7);fig.tight_layout();fig.savefig(dest/'plot.png',dpi=150);plt.close(fig)
    pairs=[];idx=np.triu_indices(480,1)
    rdms={n:(1-v['x']@v['x'].T)[idx] for n,v in models.items()}
    for left,right in itertools.combinations(models,2):
        a,b=models[left],models[right]
        correlation=spearmanr(rdms[left],rdms[right]).statistic
        pairs.append(dict(left_model=left,right_model=right,N=480,
            reference_CKA=cka_from_grams(a['gram'],b['gram']),
            within_class_residual_CKA=cka_from_grams(a['residual_gram'],b['residual_gram']),
            reference_RSA=float(correlation) if np.isfinite(correlation) else None,
            scope='paired same-image reference geometry; no adversarial features or causal claim'))
    writecsv(DEST/'cross_model_cka_rsa.csv',pairs);writecsv(DEST/'layer_cka.csv',layer_rows)
    writecsv(DEST/'class_variance.csv',variance)
    atomic_json(DEST/'provenance.json',dict(status='COMPLETE_WITH_MISSING_CACHES' if missing else 'COMPLETE',
        models=len(models),reference_images=480,reference_manifest_hashes=hashes,
        sources=sources,missing_cached_models=missing,model_inference_calls=0,gpu_model_loads=0,
        joint_fit='one shared ten-class set per model; no separate class fits',
        tsne='qualitative only; no quantitative distances from2D coordinates',
        limitations='Reference geometry only. Adversarial DeltaR and proxy/target alignment require separate PNG embedding extraction. No reverse-direction asymmetry is inferred.'))
    (DEST/'README.md').write_text('# Cached reference geometry\n\nCPU-only paired geometry of480 fixed reference images. No model inference.\n\n'+
        f'Models with exact pinned caches: {len(models)}. Missing models: {", ".join(missing)}.\n\n'+
        'CKA/RSA compare identical image IDs/order across models. Within-class residual CKA controls shared class-center structure. Layer CKA uses actual cached block IDs.\n\n'+
        'Joint PCA/t-SNE use all ten reference classes in one fit per model. These plots contain no attacked images. t-SNE distances do not support quantitative claims. Reference associations alone do not establish why an ensemble transfers.\n')
    print('Saved',DEST,'models',len(models),'cross-model pairs',len(pairs),flush=True)


if __name__=='__main__':main()
