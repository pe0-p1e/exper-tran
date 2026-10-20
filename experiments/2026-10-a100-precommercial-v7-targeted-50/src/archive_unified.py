"""Compact publication on ten-image/full/GPU milestones, with no credentials or PNG dumps."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
from proxy_study_protocol import ROOT,EXP,atomic_json


def archive(push=False):
    repo=EXP.parents[1];target=EXP/'reports/unified_campaign';target.mkdir(parents=True,exist_ok=True)
    sources={
        'commercial_model_versions':['access/*.json','ten_to_full_gate.json','campaign_completion.json','full_interface_verified_models.json',
            'model_matrix_lock.json','model_locks/*.json','parallel_orchestration_addendum.json','lanes/*.json',
            'reports/*.json','reports/*.md','stages/*/reports/*.json','stages/*/reports/*.csv','stages/*/reports/*.md','stages/*/cohort_lock.json'],
        'mechanism_reference':['*.csv','*.json','*.md'],
        'mechanism_gradients':['*.csv','*.json'],
        'surrogate_followup':['method_lock.json','progress.json','analysis/*.csv','analysis/status.json',
            'analysis/spearman.png','analysis/joint/*/*/*_coordinates.csv','analysis/joint/*/*/fit.json','embeddings/*/*.json'],
        'unified_scheduler':['adoption.json','completion.json','resource_status.json']}
    for folder,patterns in sources.items():
        for pattern in patterns:
            for p in (ROOT/folder).glob(pattern):
                rel=p.relative_to(ROOT/folder)
                if folder=='commercial_model_versions' and rel.parts[0]=='stages' and not (ROOT/folder/'stages'/rel.parts[1]/'cohort_lock.json').exists():
                    continue  # Preflight previews are not executed experiment stages.
                dest=target/folder/p.relative_to(ROOT/folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    # Two representative reference figures; all numerical reference measurements above.
    for kind in ['pca','tsne']:
        p=ROOT/'mechanism_reference'/kind/'CLIP ViT-L_14/plot.png'
        if p.exists():
            dest=target/'mechanism_reference/examples'/(kind+'.png');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    # Compact per-request provenance instead of raw CLI conversations/media/OAuth logs.
    requests=[]
    for p in (ROOT/'commercial_model_versions/stages').glob('*/commercial_cli/*/*/*/result.json'):
        d=json.loads(p.read_text());parts=p.relative_to(ROOT/'commercial_model_versions/stages').parts
        requests.append(dict(stage=parts[0],provider=parts[2],model=parts[3],request_id=parts[4],
            identity=d['identity'],raw_output=d['raw_output'],parsed_label=d['parsed_label'],session_id=d.get('session_id'),
            execution_settings=d.get('execution_settings',{'effort':'low' if parts[2]=='codex' else 'unknown'}),
            evidence=d.get('evidence',d.get('model_identity_evidence')),reused=(p.parent/'matrix_reuse.json').exists()))
    atomic_json(target/'commercial_request_provenance.json',requests)
    if push:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
        if branch!='experiment/a100-precommercial-v7-proxy-study-18h':raise RuntimeError('Refuse publication from wrong branch')
        with (ROOT/'publication.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            subprocess.run(['git','add',str(target.relative_to(repo))],cwd=repo,check=True)
            changed=subprocess.run(['git','diff','--cached','--quiet','--',str(target.relative_to(repo))],cwd=repo).returncode
            if changed:subprocess.run(['git','commit','-m','Publish unified commercial and mechanism experiment measurements'],cwd=repo,check=True)
            subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:experiment/a100-precommercial-v7-proxy-study-18h'],cwd=repo,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--push',action='store_true');a=p.parse_args()
    if not a.watch:archive(a.push)
    else:
        markers=[ROOT/'commercial_model_versions/ten_to_full_gate.json',ROOT/'commercial_model_versions/campaign_completion.json',ROOT/'unified_scheduler/completion.json']
        handled=set()
        while len(handled)<len(markers):
            ready=[str(m) for m in markers if m.exists() and str(m) not in handled]
            if ready:
                try:archive(a.push);handled.update(ready)
                except subprocess.CalledProcessError as exc:print('Publication attempt failed; records retained:',exc,flush=True)
            time.sleep(30)
