#!/usr/bin/env python3
"""Feature, input-gradient and BF16 numeric audit before method development."""
import argparse
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from v7_common import OUT, BASE, TIDS, records, images, atomic_json, prepare_data, fingerprint
from vision_features import VisionProxy
from representation_objectives import original_pull_push_loss, class_scores, all_class_margin_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);a=p.parse_args();prepare_data()
    reports=[]
    for name in a.models:
        m=VisionProxy(name,(.01,.25,.5,.75,1.))
        try:
            # Real disjoint development examples; no final target call.
            batch=[records(OUT/'manifests/dev'/f'{tid}.jsonl')[0] for tid in TIDS]
            x=images(batch[:2]).requires_grad_(True)
            features=m.features(x)
            comparisons=[]
            for layer,(z,t,mask) in features.items():
                with torch.no_grad():old=m.proxy.image_embeddings(x.detach(),representation_type='vision_encoder',layer=layer,pooling='mean')
                err=float((z.detach()-old.embeddings.float()).abs().max())
                if not torch.allclose(z.detach(),old.embeddings.float(),atol=2e-6,rtol=2e-5):raise RuntimeError(f'Feature semantics differ {name}/{layer}: {err}')
                g=torch.autograd.grad(z[:,0].sum(),x,retain_graph=True)[0]
                norms=g.abs().flatten(1).sum(1)
                if not torch.isfinite(g).all() or (norms==0).any():raise RuntimeError(f'No raw-image gradient {name}/{layer}')
                comparisons.append(dict(layer=layer,token_shape=list(t.shape),valid_tokens=mask.sum(1).tolist(),max_difference_from_v6=err,gradient_l1=norms.tolist(),embedding_std=z.std(1).tolist(),norm=z.norm(dim=-1).tolist()))
            deepest=max(features)
            with torch.no_grad():
                z=features[deepest][0].float()
                gram=F.normalize(z,dim=-1)@F.normalize(z,dim=-1).T
            record=dict(model=name,features=comparisons,clean_pair_cosine=gram.tolist(),parameter_dtype='bfloat16',quantized=False,revision=m.spec['revision'])
            reports.append(record);atomic_json(OUT/'audits'/f'feature_diagnostic_{fingerprint(name)[:12]}.json',record)
        finally:m.close()
    # Read historical frozen predictions for diagnostic summaries only. These do not
    # enter the surrogate-selection score or attack runner.
    confusion={}
    for path in (BASE/'multiple_proxy/evaluations').glob('*/*json'):
        d=json.loads(path.read_text())
        if isinstance(d,dict) and d.get('transition_id') in ('T03','T05'):
            key=d.get('target','?')+'/'+d['transition_id'];hist=confusion.setdefault(key,{})
            for r in d.get('images',[]):
                label=str(r.get('adv_label'));hist[label]=hist.get(label,0)+1
    atomic_json(OUT/'audits/historical_failure_confusion.json',confusion)
    (OUT/'reports').mkdir(parents=True,exist_ok=True)
    lines=['# Initial V7 diagnostics','','Feature semantics compared against V6 at all selected layers. BF16 parameter audit precedes every inference. Raw 224px image gradients are finite and nonzero.','', 'Historical confusion is diagnostic and excluded from method-selection inputs.','']
    for r in reports:lines.append(f"- {r['model']}: {len(r['features'])} genuine blocks checked, exact revision {r['revision']}.")
    lines+=['','Near-constant embeddings, layer efficacy and gradient conflicts require further measured interpretation; successful gradient smoke is not transfer evidence.']
    (OUT/'reports/diagnostics.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
