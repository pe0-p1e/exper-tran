"""Compact, authorized result upload; never publish model/image/feature corpora."""
import json
from pathlib import Path
import subprocess
import time
from v7_common import OUT,REPO,REGISTRY,atomic_json


def model_audits(names,dest):
    events=json.loads((OUT/'audits/model_dtype_audit.json').read_text())
    latest={e['model_name']:e for e in events if e['model_name'] in names}
    passed=set(latest)==set(names) and all(e['all_floating_parameters_bf16'] and e['status']=='pass' and e['revision']==REGISTRY[n]['revision'] for n,e in latest.items())
    atomic_json(dest,dict(all_expected_models_verified=passed,expected_models=sorted(names),latest_model_audits=latest))
    return passed


def upload_compact(root):
    root=Path(root).resolve()
    if not root.is_relative_to(OUT.resolve()):raise ValueError('Upload root outside V7 outputs')
    excluded={'attacks','evaluations','embeddings','reference_cache','canonical_images','logs','data'}
    files=[p for p in root.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(root).parts) and p.suffix in ('.json','.csv','.md','.yaml') and p.name!='upload_status.json']
    rel=[str(p.relative_to(REPO)) for p in files]
    try:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
        if branch!='experiment/a100-precommercial-v7-targeted-50':raise RuntimeError('Wrong branch; preserve local results without uploading')
        if rel:
            subprocess.run(['git','add','-f','--',*rel],cwd=REPO,check=True)
            if subprocess.run(['git','diff','--cached','--quiet','--',*rel],cwd=REPO).returncode:
                subprocess.run(['git','commit','--only','-m','Archive '+root.name+' measurements and validation','--',*rel],cwd=REPO,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:'+branch],cwd=REPO,check=True,timeout=120)
        atomic_json(root/'upload_status.json',dict(status='pushed',commit=commit,at=time.time()))
    except Exception as exc:atomic_json(root/'upload_status.json',dict(status='failed',error=str(exc),local_results_retained=True))
