"""Fill missing clean proxy embeddings after the main correction finishes."""
import argparse
import json
import numpy as np
import torch
from v7_common import OUT,REGISTRY,sha,fingerprint
from proxy_selection_tables import ROOT
from vision_features import VisionProxy
from primary_ml_cka.artifacts.png import load_png_tensor


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.load(open(a.plan))['rows'];work={}
    for row in plan:
        if len(row['config']['models'])!=3:continue
        for file in (OUT/'attacks/test'/row['measurement_condition']).glob('T*/batch_*/state.json'):
            state=json.loads(file.read_text())
            for rel in state['clean_paths']:
                for name in row['config']['models']:work[name,sha(OUT/rel)]=rel
    for name in sorted({n for n,_ in work}):
        missing=[]
        for (n,h),rel in work.items():
            if n!=name:continue
            key=fingerprint([h,REGISTRY[name],[REGISTRY[name]['expected_vision_depth']-1],'V6_spatial_mean'])
            roots=[ROOT/'analysis',OUT/'family_disjoint_recovery/analysis',OUT/'ten_hour_research/analysis']
            if any((r/'embeddings'/name.replace('/','_')/(key+'.npy')).exists() for r in roots):continue
            missing.append((rel,ROOT/'analysis/embeddings'/name.replace('/','_')/(key+'.npy')))
        if not missing:continue
        m=VisionProxy(name,role='proxy_selection_clean_analysis');m.to('cuda')
        try:
            for rel,dest in missing:
                with torch.inference_mode():z=m.features(load_png_tensor(OUT/rel)[None].cuda())[m.layers[0]][0][0].float().cpu().numpy()
                if not np.isfinite(z).all():raise RuntimeError('Invalid clean proxy feature')
                dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.npy');np.save(tmp,z);tmp.replace(dest)
        finally:m.close()


if __name__=='__main__':main()
