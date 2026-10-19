"""Real four-tap BF16 gradients and safe memory calibration, GPU owner only."""
from supplement_registry import install,memory_cap
install()
memory_cap()
import argparse
import json
import time
import torch
from v7_common import OUT,REGISTRY,records,images,atomic_json,sha
from vision_features import VisionProxy
from run_v7_attack import objective,METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);a=p.parse_args()
    root=OUT/'multiproxy_8h_supplement';dest=root/'audits/model_gradients.json'
    audit=json.loads(dest.read_text()) if dest.exists() else {}
    for name in a.models:
        if name in audit and audit[name].get('passed') and audit[name]['spec']==REGISTRY[name]:continue
        model=None;began=time.time()
        try:
            torch.cuda.reset_peak_memory_stats();model=VisionProxy(name,[.25,.5,.75,1.],role='supplement_proxy_calibration')
            model.prepare_bank();model.to('cuda')
            bank_seconds=time.time()-began;measurements=[];backwards=0
            rows=records(OUT/'manifests/dev/T03.jsonl')[:4]
            for size in [1,2,4]:
                try:
                    torch.cuda.reset_peak_memory_stats();x=images(rows[:size]).detach().requires_grad_(True)
                    torch.cuda.synchronize();start=time.perf_counter();loss,margin=objective(model,x,3,4,{**METHODS['A3'],'patch_weight':.35})
                    grad=torch.autograd.grad(loss.sum(),x)[0];backwards+=1;torch.cuda.synchronize();seconds=time.perf_counter()-start
                    if not torch.isfinite(grad).all() or (grad.flatten(1).abs().sum(1)<=0).any():raise RuntimeError('Invalid224px input gradient')
                    free,total=torch.cuda.mem_get_info();peak=torch.cuda.max_memory_reserved()/2**30
                    measurements.append(dict(batch=size,seconds=seconds,seconds_per_image_step=seconds/size,peak_reserved_gib=peak,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,free_gib=free/2**30,safe=peak<=75 and free/2**30>=8))
                    del x,loss,margin,grad;torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    measurements.append(dict(batch=size,status='OOM',safe=False));torch.cuda.empty_cache()
                    break
            # Independently verify every selected true block reaches original pixels.
            x=images(rows[:1]).detach().requires_grad_(True);fs=model.features(x);taps=[]
            for layer,(z,tokens,mask) in fs.items():
                refs=model.bank[layer][0][3].mean(0);term=-(z.float()*refs.float()).sum()
                g=torch.autograd.grad(term,x,retain_graph=True)[0];backwards+=1
                if not torch.isfinite(g).all() or not g.abs().sum()>0:raise RuntimeError('Disconnected visual layer')
                taps.append(dict(layer=layer,token_shape=list(tokens.shape),valid_tokens=int(mask.sum()),gradient_l1=float(g.abs().sum())))
            safe=[r for r in measurements if r.get('safe') and r['batch']<=2]
            if not safe:raise RuntimeError('No safe BF16 calibration batch')
            chosen=max(safe,key=lambda r:r['batch'])
            audit[name]=dict(passed=True,spec=REGISTRY[name],layers=model.layers,taps=taps,measurements=measurements,reference_prepare_seconds=bank_seconds,chosen_batch=chosen['batch'],seconds_per_image_step=chosen['seconds_per_image_step'],elapsed_seconds=time.time()-began,model_calls=dict(vision=model.forward_calls,backward=backwards,image_forwards=model.image_forwards))
            del fs,x,g
        except Exception as exc:audit[name]=dict(passed=False,spec=REGISTRY[name],error=str(exc),elapsed_seconds=time.time()-began)
        finally:
            if model is not None:model.close()
        atomic_json(dest,audit);print(name,audit[name]['passed'],flush=True)
        if not audit[name]['passed']:raise RuntimeError(f'Proxy smoke failed: {name}: {audit[name]["error"]}')


if __name__=='__main__':main()
