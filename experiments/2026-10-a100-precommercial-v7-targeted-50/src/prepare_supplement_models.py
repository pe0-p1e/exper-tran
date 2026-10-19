"""CPU/network only: download pinned extension weights without GPU allocation."""
import json
import os
import time
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
from supplement_registry import PIN_FILE
from v7_common import OUT,atomic_json


def main():
    root=OUT/'multiproxy_8h_supplement';events=[]
    cache=Path(os.environ.get('HF_HOME','/ephemeral/a100_precommercial_v6/model-cache/hf-cache'))/'hub'
    for name,s in json.loads(PIN_FILE.read_text()).items():
        try:
            api=HfApi();info=api.model_info(s['repo_id'],revision=s['revision'])
            if info.sha!=s['revision']:raise RuntimeError('Resolved checkpoint SHA changed')
            config=json.loads(Path(hf_hub_download(s['repo_id'],'config.json',revision=s['revision'],cache_dir=cache)).read_text())
            vision=config.get('vision_config',config)
            if vision['num_hidden_layers']!=s['expected_vision_depth'] or vision['image_size']!=s.get('config_image_size',s['native_image_size']):raise RuntimeError('Checkpoint config differs from pinned audit')
            files=[x.rfilename for x in info.siblings]
            weights='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
            path=snapshot_download(s['repo_id'],revision=s['revision'],cache_dir=cache,allow_patterns=[weights,'*.json','*.txt','*.model'],max_workers=2)
            events.append(dict(model=name,status='downloaded',revision=info.sha,snapshot=path,weight_format=weights,depth=vision['num_hidden_layers'],config_image_size=vision['image_size'],actual_input_size=s['native_image_size'],preprocessing=s.get('preprocessing','V5-compatible CLIP native square resize'),at=time.time(),BF16_runtime_audit='pending_GPU_load'))
        except Exception as exc:events.append(dict(model=name,status='blocked',error=str(exc),at=time.time()))
        atomic_json(root/'audits/extension_downloads.json',events)
        print(events[-1],flush=True)


if __name__=='__main__':main()
