import sys
import pp4_worker as worker
from pp4_mechanism_protocol import ROOT,plan,validation_path,atomic_json

def load_condition(cid):
    p=plan()
    return next(c for c in [r['condition'] for r in p['ladder']]+[p['smoke']] if c['id']==cid)

worker.ROOT=ROOT
worker.validation_path=validation_path
worker.load_condition=load_condition
if __name__=='__main__':
    cid=sys.argv[sys.argv.index('--condition')+1];c=load_condition(cid);diagnostics=[]
    if sys.argv[1]=='attack' and c['options'].get('gradient_normalization')=='none':
        import run_v7_attack as runner
        import torch
        from attack_optimizers import normalize_pixel_gradients
        from pp4_raw_fusion import raw_equal_mean
        def fused(g,**kwargs):
            raw=raw_equal_mean(g,**kwargs);norm=normalize_pixel_gradients(g).mean(0)
            sizes=g.float().abs().flatten(2).sum(2)
            diagnostics.append(dict(call=len(diagnostics)+1,gradient_l1=sizes.cpu().tolist(),
                raw_contribution_fraction=(sizes/sizes.sum(0).clamp_min(1e-12)).cpu().tolist(),
                raw_vs_normalized_cosine=torch.nn.functional.cosine_similarity(raw.flatten(1),norm.flatten(1)).cpu().tolist()))
            return raw
        runner.fuse_proxy_gradients=fused
    try:worker.main()
    finally:
        if diagnostics:
            import os
            atomic_json(ROOT/'fusion_diagnostics'/f'{cid}_{os.getpid()}.json',dict(condition=cid,
                process_calls=len(diagnostics),partial_or_resumed_process=True,calls=diagnostics))
