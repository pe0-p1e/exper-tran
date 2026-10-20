"""Same weights/input/layer comparison of the V6 proxy API and V7 feature path.

Diagnostic only. No attack, target model, model output labels or target feedback.
"""
import fcntl
import hashlib
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pp4_worker as base
from pp4_protocol import OUT,PREVIOUS,EXP,records,sha,atomic_json
from pp4_campaign import table
from vision_features import VisionProxy
from representation_objectives import original_pull_push_loss

ROOT=OUT/'v6_v7_setting_audit'


def main():
    with (PREVIOUS/'scheduler.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.ROOT=ROOT;base.install_audit()
        from v7_common import images
        base.shared.verify_existing_data()
        refs=[records(OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl') for i in range(1,11)]
        digest=hashlib.sha256(''.join(r['image_id'] for bank in refs for r in bank).encode()).hexdigest()
        rows=[];caches=[];began=time.time()
        for name in ['Qwen3.5-2B','InternVL3.5-2B-HF','Gemma 4 E2B-it']:
            model=VisionProxy(name,[.25,.5,.75,1.],role='V6_V7_parity_proxy')
            try:
                model.prepare_bank()
                for layer in [model.layers[-2],model.layers[-1]]:
                    x=images(records(OUT/'manifests/dev/T03.jsonl')[2:4]).detach().requires_grad_(True)
                    old=model.proxy.image_embeddings(x,representation_type='vision_encoder',layer=layer,pooling='mean').embeddings.float()
                    bank=model.bank[layer][0]
                    centers=F.normalize(F.normalize(bank.float(),dim=-1).mean(1),dim=-1)
                    old_loss=.75*(1-(F.normalize(old,dim=-1)*centers[3]).sum(-1))+.25*(1+(F.normalize(old,dim=-1)*centers[2]).sum(-1))
                    old_g=torch.autograd.grad(old_loss.sum(),x)[0].detach().float()
                    old_z=old.detach();del old,old_loss
                    new=model.features(x)[layer][0]
                    new_loss=original_pull_push_loss(new,bank,3,4,reduction='none')
                    new_g=torch.autograd.grad(new_loss.sum(),x)[0].detach().float()
                    if not torch.isfinite(old_g).all() or not torch.isfinite(new_g).all():raise RuntimeError('Nonfinite parity gradient')
                    cos=F.cosine_similarity(old_g.flatten(1),new_g.flatten(1))
                    rows.append(dict(model=name,layer=layer,N_images=2,
                        embedding_max_abs_difference=float((old_z-new.detach()).abs().max()),
                        embedding_cosine_mean=float(F.cosine_similarity(old_z,new.detach()).mean()),
                        gradient_cosine_min=float(cos.min()),gradient_cosine_mean=float(cos.mean()),
                        gradient_norm_ratio=float((new_g.flatten(1).norm(dim=1)/old_g.flatten(1).norm(dim=1).clamp_min(1e-12)).mean()),
                        sign_agreement=float((old_g.sign()==new_g.sign()).float().mean()),
                        note='Current checkout of original V6 API versus V7 path; both losses .75/.25; not a historical binary/environment replay'))
                    table(ROOT/'reports/feature_gradient_parity.csv',rows)
                    print('PARITY',name,layer,rows[-1],flush=True)
                    del x,new,new_loss,new_g,old_g;torch.cuda.empty_cache()
                spec=model.spec;deep=model.layers[-1]
                key=hashlib.sha256(f"{spec['repo_id']}|{spec['revision']}|{digest}|{deep}|vision_encoder|mean".encode()).hexdigest()
                path=OUT.parent/'a100_precommercial_v6_multi_repr_recovery/cache/reference_embeddings'/f'{key}.npz'
                result=dict(model=name,layer=deep,historical_cache=str(path),present=path.exists())
                if path.exists():
                    with np.load(path) as data:old=torch.from_numpy(np.stack([data[f'class_{i}'] for i in range(10)])).float()
                    new=model.bank[deep][0].detach().float().cpu()
                    result.update(N_reference_vectors=int(old.shape[0]*old.shape[1]),
                        cosine_min=float(F.cosine_similarity(old,new,dim=-1).min()),max_abs_difference=float((old-new).abs().max()),
                        old_cache_sha256=sha(path),note='Historical stored feature vectors versus current cached bank, all480 references')
                caches.append(result);atomic_json(ROOT/'reports/reference_cache_parity.json',caches)
            finally:model.close()
        atomic_json(ROOT/'parity_complete.json',dict(status='MEASURED',seconds=time.time()-began,
            settings_source_sha256=sha(__file__),rows=len(rows),target_models_loaded=0,
            caveat='Current software path parity plus historical cached values; not proof that every historical runtime dependency is identical'))

if __name__=='__main__':main()
