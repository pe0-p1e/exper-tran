"""Isolated child-process registry. Never edit frozen registry/loader files."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from v7_common import EXP, REGISTRY as BASE_REGISTRY, sha

PIN_FILE = EXP/'supplement_model_revisions.json'


def registry():
    extra=json.loads(PIN_FILE.read_text())
    if set(extra)&set(BASE_REGISTRY):raise RuntimeError('Extension cannot replace an existing model')
    for spec in extra.values():
        if len(spec['revision'])!=40:raise ValueError('Full checkpoint SHA required')
    return {**BASE_REGISTRY,**{n:{**s,'extension_loader_sha256':sha(__file__)} for n,s in extra.items()}}


def native_clip_preprocess(images,size,mean,std):
    from primary_ml_cka.data.preprocessing import resize_crop_normalize
    import torch
    return resize_crop_normalize(images,size=size,mean=tuple(mean),std=tuple(std)).to(torch.bfloat16)


def install():
    """Call before importing the frozen attack runner in a NEW Python process."""
    import sys
    if 'vision_features' in sys.modules:raise RuntimeError('Install extension before importing vision_features')
    import v7_common
    v7_common.REGISTRY=registry()
    import vision_features
    # The frozen auditor appends atomically but has no cross-process writer
    # lock. Keep supplementary GPU load events in a separate audit stream.
    import strict_model_audit
    audit_original=strict_model_audit.audit_loaded_model
    def audit_isolated(model,**kwargs):
        if 'CPU' not in kwargs.get('role',''):
            kwargs['output_dir']=v7_common.OUT/'multiproxy_8h_supplement'
        return audit_original(model,**kwargs)
    vision_features.audit_loaded_model=audit_isolated
    strict_model_audit.audit_loaded_model=audit_isolated
    original=vision_features.load_proxy
    extras=json.loads(PIN_FILE.read_text())

    def load(model_id,hf_home,device,attack_config=None):
        matches=[s for s in extras.values() if s['repo_id']==model_id]
        if not matches:return original(model_id,hf_home,device,attack_config)
        import torch
        from transformers import AutoModel,AutoImageProcessor
        from primary_ml_cka.models.common.loading import local_snapshot,freeze_module
        from primary_ml_cka.models.common.precision import assert_bf16_unquantized
        if os.environ.get('PRIMARY_ML_CKA_STRICT_BF16')!='1' or attack_config.generative_precision!='bf16':raise RuntimeError('BF16 required')
        spec=matches[0];snap=local_snapshot(Path(hf_home),model_id,spec['revision'])
        model=AutoModel.from_pretrained(snap,local_files_only=True,torch_dtype=torch.bfloat16).to(device=device,dtype=torch.bfloat16)
        assert_bf16_unquantized(model,model_name=model_id)
        freeze_module(model)
        processor=AutoImageProcessor.from_pretrained(snap,local_files_only=True)
        if spec['family']=='clip':
            size=int(model.config.vision_config.image_size)
            if size!=spec['native_image_size']:raise RuntimeError('Native CLIP resolution changed')
            return SimpleNamespace(model=model,image_preprocess=lambda x:native_clip_preprocess(x,size,processor.image_mean,processor.image_std))
        size=int(processor.size.get('height',processor.size.get('shortest_edge',518)))
        if size!=spec['native_image_size']:raise RuntimeError('Native DINO resolution changed')
        return SimpleNamespace(model=model,image_size=size,mean=torch.tensor(processor.image_mean,device=device).view(1,3,1,1),std=torch.tensor(processor.image_std,device=device).view(1,3,1,1))

    vision_features.load_proxy=load
    return v7_common.REGISTRY


def memory_cap():
    """A child allocation ceiling, independent of the existing owner's limit."""
    value=os.environ.get('SUPPLEMENT_MEMORY_CAP_GIB')
    if value:
        import torch
        total=torch.cuda.get_device_properties(0).total_memory/2**30
        cap=float(value)
        if not 0<cap<total:raise ValueError('Invalid supplementary VRAM cap')
        torch.cuda.set_per_process_memory_fraction(cap/total)


def verify_existing_data():
    """Read existing cohorts; do not overwrite the active global data audits."""
    from v7_common import OUT,records,TIDS
    references=[r for i in range(1,11) for r in records(OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl')]
    if len(references)!=480:raise RuntimeError('Reference bank incomplete')
    refs={r['image_id'] for r in references}
    for split,count in [('test',30),('dev',10)]:
        for tid in TIDS:
            rows=records(OUT/'manifests'/split/f'{tid}.jsonl')
            if len(rows)!=count or refs&{r['image_id'] for r in rows}:raise RuntimeError('Changed cohort or reference leakage')
            if any(not (OUT/'canonical_images'/r['relative_path']).is_file() for r in rows):raise RuntimeError('Missing canonical cohort PNG')
