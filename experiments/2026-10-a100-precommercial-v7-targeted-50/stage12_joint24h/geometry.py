"""Aligned reference CKA/RSA, never raw cross-dimensional feature cosine."""
from common import *
import numpy as np,itertools

def reference(n,l):
 old=DATA/'all_layer_selection/models'/n.replace('/','_')
 receipt=read(old/'reference_features.json')
 if receipt and sha(old/'reference_features.npz')==receipt['sha256']:
  with np.load(old/'reference_features.npz') as z:return z[f'z_{l}'].reshape(-1,z[f'z_{l}'].shape[-1]).astype('float64')
 for p in (ROOT/'cache/refs').glob('*.json'):
  d=read(p)
  if d.get('model')==n and d.get('revision')==REG[n]['revision']:
   path=p.with_suffix('.npz')
   if sha(path)!=d['sha256']:raise RuntimeError('Reference checksum mismatch')
   with np.load(path) as z:return z[f'z_{l}'].reshape(-1,z[f'z_{l}'].shape[-1]).astype('float64')
 raise RuntimeError('Missing aligned references '+n)
def cka(x,y):
 if len(x)!=len(y):raise ValueError('Unaligned sample count')
 x=x-x.mean(0);y=y-y.mean(0);kx=x@x.T;ky=y@y.T
 den=np.linalg.norm(kx)*np.linalg.norm(ky)
 return float((kx*ky).sum()/den) if den>1e-14 else None
def rsa(x,y):
 from scipy.stats import spearmanr
 x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
 ids=np.triu_indices(len(x),1);value=float(spearmanr((1-x@x.T)[ids],(1-y@y.T)[ids]).statistic)
 return value if np.isfinite(value) else None
def complement(n,best,candidates):
 p=ROOT/'geometry'/(digest([n,best,candidates])[:18]+'.json');d=read(p)
 if not d:
  x=reference(n,best);rr=[]
  for l in candidates:
   if l==best:continue
   y=reference(n,l);rr.append(dict(model=n,left=best,right=l,CKA=cka(x,y),RSA=rsa(x,y),N=480))
  d=dict(pairs=rr,reference_ids=[r['image_id'] for label in range(1,11) for r in rows(DATA/'manifests/class_references'/f'class_references_{label:02d}.jsonl')]);atomic(p,d)
 values=[r for r in d['pairs'] if r['CKA'] is not None]
 if not values:raise RuntimeError('All candidate reference geometry degenerate')
 return min(values,key=lambda r:(r['CKA'],r['right']))['right']
