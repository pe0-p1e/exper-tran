"""BF16 isolated child, serialized audits/reference cache; frozen attack implementation."""
import fcntl
import json
import sys
import time

from proxy_study_protocol import ROOT, OUT, atomic_json
import strict_model_audit
original_audit = strict_model_audit.audit_loaded_model
from supplement_registry import install, memory_cap, verify_existing_data
install()
memory_cap()
import vision_features


def audit(model, **kwargs):
    kwargs['output_dir'] = ROOT
    path = ROOT / 'audits/writer.lock'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return original_audit(model, **kwargs)


vision_features.audit_loaded_model = audit
strict_model_audit.audit_loaded_model = audit
old_prepare = vision_features.VisionProxy.prepare_bank


def bank_locked(self):
    # New study workers must not race another study writer on the same .tmp.npz.
    path = ROOT / 'audits/reference_bank.lock'
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return old_prepare(self)


vision_features.VisionProxy.prepare_bank = bank_locked


def calibrate(names):
    import torch
    from v7_common import REGISTRY, records, images
    from vision_features import VisionProxy
    from run_v7_attack import objective, METHODS
    for name in names:
        began = time.time(); model = None
        try:
            model = VisionProxy(name, [.25, .5, .75, 1.], role='study18_proxy_calibration')
            model.to('cuda'); model.prepare_bank()
            results = []
            for size in [1, 2]:
                torch.cuda.reset_peak_memory_stats()
                x = images(records(OUT / 'manifests/dev/T03.jsonl')[:size]).detach().requires_grad_(True)
                torch.cuda.synchronize(); start = time.perf_counter()
                loss, margin = objective(model, x, 3, 4, {**METHODS['A3'], 'patch_weight': .35})
                g = torch.autograd.grad(loss.sum(), x)[0]
                torch.cuda.synchronize()
                if not torch.isfinite(g).all() or (g.abs().flatten(1).sum(1) == 0).any():
                    raise RuntimeError('Nonfinite or disconnected gradient')
                results.append(dict(batch=size, seconds=time.perf_counter()-start,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                    gradient_l1=g.abs().flatten(1).sum(1).tolist()))
                del x, loss, margin, g
                torch.cuda.empty_cache()
            atomic_json(ROOT / 'calibration' / (name.replace('/', '_') + '.json'),
                        dict(passed=True, spec=REGISTRY[name], measurements=results,
                             layers=model.layers, elapsed_seconds=time.time()-began))
        finally:
            if model is not None:
                model.close()


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'calibration':
        calibrate(sys.argv[1:])
    elif mode == 'attack':
        import run_v7_attack
        run_v7_attack.prepare_data = verify_existing_data
        run_v7_attack.main()
    elif mode == 'evaluation':
        from proxy_study_protocol import evaluator_hashes
        lockarg = sys.argv[sys.argv.index('--method-lock')+1]
        locked = json.loads(__import__('pathlib').Path(lockarg).read_text())
        if locked.get('evaluator_hashes') and locked['evaluator_hashes'] != evaluator_hashes():
            raise RuntimeError('Evaluator/prompt/parser changed')
        import evaluate_v7
        evaluate_v7.main()
    else:
        raise ValueError(mode)
