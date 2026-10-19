"""Real-checkpoint CPU preflight while A100 is owned; never replaces GPU smoke."""
from supplement_registry import install,PIN_FILE
install()
import gc
import json
import os
import time
import torch
from v7_common import OUT,REGISTRY,images,records,atomic_json,sha
from vision_features import VisionProxy,layer_indices,load_proxy
from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from strict_model_audit import audit_loaded_model


def main():
    if torch.cuda.is_available():raise RuntimeError('CPU preflight requires CUDA_VISIBLE_DEVICES empty')
    root=OUT/'multiproxy_8h_supplement/cpu_preflight';results=[]
    for name in json.loads(PIN_FILE.read_text()):
        began=time.time();spec=REGISTRY[name];proxy=None
        try:
            MODEL_REVISIONS[spec['repo_id']]=spec['revision']
            proxy=load_proxy(spec['repo_id'],__import__('pathlib').Path(os.environ['HF_HOME']),torch.device('cpu'),AttackConfig(generative_precision='bf16'))
            audit=audit_loaded_model(proxy.model,model_name=name,revision=spec['revision'],role='proxy_CPU_preflight_only',repo_id=spec['repo_id'],output_dir=root,vision_depth=spec['expected_vision_depth'])
            cfg=getattr(proxy.model.config,'vision_config',proxy.model.config)
            depth=getattr(cfg,'num_hidden_layers',None)
            if depth!=spec['expected_vision_depth']:raise RuntimeError('Loaded visual depth mismatch')
            visual=object.__new__(VisionProxy);visual.model=proxy.model;visual.proxy=proxy;visual.spec=spec;visual.layers=layer_indices(depth,[.25,.5,.75,1.])
            refs=images(records(OUT/'manifests/class_references/class_references_04.jsonl')[:2],device='cpu')
            with torch.inference_mode():bank=visual.features(refs)
            x=images(records(OUT/'manifests/dev/T03.jsonl')[:1],device='cpu').requires_grad_(True)
            features=visual.features(x);taps=[]
            for layer,(z,tokens,mask) in features.items():
                reference=bank[layer][0].mean(0)
                gradient=torch.autograd.grad(-(z*reference).sum(),x,retain_graph=True)[0]
                if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:raise RuntimeError('Disconnected/invalid CPU BF16 pixel gradient')
                taps.append(dict(layer=layer,tokens=list(tokens.shape),valid_tokens=int(mask.sum()),gradient_l1=float(gradient.abs().sum())))
            results.append(dict(model=name,revision=spec['revision'],status='CPU_VALIDATED_GPU_PENDING',BF16_parameters=audit['all_floating_parameters_bf16'],depth=depth,layers=taps,reference_smoke_images=2,actual_input_size=spec['native_image_size'],GPU_smoke_completed=False,elapsed_seconds=time.time()-began))
            del bank,features,x,gradient,visual
        except Exception as exc:results.append(dict(model=name,status='BLOCKED',error=str(exc),GPU_smoke_completed=False))
        finally:
            if proxy is not None:del proxy
            gc.collect()
        atomic_json(root/'results.json',results);print(results[-1],flush=True)


if __name__=='__main__':main()
