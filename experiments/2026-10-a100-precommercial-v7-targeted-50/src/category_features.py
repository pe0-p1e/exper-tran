"""Class-wise reference caches for the unchanged visual loss at arbitrary C."""
import numpy as np
from pathlib import Path
import torch
import transformers
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from vision_features import VisionProxy
from v7_common import REPO,fingerprint,sha
from primary_ml_cka.artifacts.png import load_png_tensor


def image_batch(root,rows):
    return torch.stack([load_png_tensor(root/r['path']) for r in rows]).cuda()


class CategoryVisionProxy(VisionProxy):
    def prepare_category_bank(self,data,data_root,cache_root):
        # Class-wise caching shares unchanged48-image banks across10/20/50.
        by_layer={l:[] for l in self.layers};patch_layer={l:[] for l in self.layers}
        for c in data['classes']:
            key=fingerprint(dict(model=self.spec,class_id=c['id'],references=c['references'],layers=self.layers,k=self.patch_k,feature_code=sha(Path(__file__).with_name('vision_features.py')),preprocess_code=sha(REPO/'src/primary_ml_cka/models/proxies/visual.py'),stack=[torch.__version__,transformers.__version__],bank_code=sha(__file__),version='category_banks_v1'))
            path=cache_root/(key+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as d:
                    for l in self.layers:
                        z=d[f'z_{l}'];patch=d[f'p_{l}']
                        if len(z)!=48 or len(patch)!=self.patch_k or not np.isfinite(z).all() or not np.isfinite(patch).all():raise RuntimeError('Invalid cached new-class bank')
                        by_layer[l].append(z);patch_layer[l].append(patch)
                continue
            zs={l:[] for l in self.layers};tokens={l:[] for l in self.layers}
            with torch.inference_mode():
                for r in c['references']:
                    fs=self.features(image_batch(data_root,[r]))
                    for l,(z,t,m) in fs.items():
                        zs[l].append(z.cpu().numpy());real=t[0][m[0]].float()
                        ix=torch.linspace(0,len(real)-1,min(32,len(real)),device=real.device).long()
                        tokens[l].append(F.normalize(real[ix],dim=-1).cpu().numpy())
            payload={}
            for l in self.layers:
                z=np.concatenate(zs[l]);pts=np.concatenate(tokens[l]);clusters=MiniBatchKMeans(n_clusters=self.patch_k,random_state=42,n_init=3,batch_size=256).fit(pts).cluster_centers_
                if not np.isfinite(z).all() or not np.isfinite(clusters).all():raise RuntimeError('Nonfinite new-class reference bank')
                payload[f'z_{l}']=z;payload[f'p_{l}']=clusters
                by_layer[l].append(z);patch_layer[l].append(clusters)
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(path)
        self.bank={l:(torch.from_numpy(np.stack(by_layer[l])).cuda(),torch.from_numpy(np.stack(patch_layer[l])).cuda()) for l in self.layers}
