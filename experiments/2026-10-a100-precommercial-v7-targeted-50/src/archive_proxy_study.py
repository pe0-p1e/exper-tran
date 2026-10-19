"""Publish only compact measurements; never copy PNGs, credentials or OAuth logs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from proxy_study_protocol import ROOT, EXP, atomic_json


def archive(push=False):
    repo=EXP.parents[1]
    target=EXP/'reports/proxy_study_18h';target.mkdir(parents=True,exist_ok=True)
    for folder,patterns in {'reports':['*.csv','*.md'],
                            'historical_audit':['*.json','*.csv','*.md']}.items():
        for pattern in patterns:
            for p in (ROOT/folder).glob(pattern):
                dest=target/folder/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,dest)
    for name in ['budget.json','method_lock.json','completion.json']:
        p=ROOT/name
        if p.exists():shutil.copyfile(p,target/name)
    repair=ROOT/'audits/reporting_source_repair.json'
    if repair.exists():shutil.copyfile(repair,target/repair.name)
    state=ROOT/'state.json'
    if state.exists():
        d=json.loads(state.read_text());atomic_json(target/'runtime_status.json',
             {k:v for k,v in d.items() if k not in ['active','smokes']})
    events=ROOT/'audits/model_dtype_audit.json'
    if events.exists():
        latest={}
        all_events=json.loads(events.read_text())
        for e in all_events:latest[(e['model_name'],e['role'])]=e
        atomic_json(target/'bf16_audit_latest_per_model_role.json',
                    dict(total_load_events=len(all_events),audits=list(latest.values()),
                         any_failed=any(e['status']!='pass' for e in all_events)))
    # Only completed blinded CLI result records. Never archive settings/auth files.
    for pattern in ['*/*/client_audit.json','*/*/pilot_summary.json','*/*/*/result.json']:
        for p in (ROOT/'commercial_cli').glob(pattern):
            dest=target/'commercial_cli'/p.relative_to(ROOT/'commercial_cli')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for pattern in ['reports/*.csv','reports/*.md','status.json','preflight.json','locks/*.json']:
        for p in (ROOT/'commercial_cli_campaign').glob(pattern):
            dest=target/'commercial_cli_campaign'/p.relative_to(ROOT/'commercial_cli_campaign')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    if push:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
        if branch!='experiment/a100-precommercial-v7-proxy-study-18h':
            raise RuntimeError('Refuse to publish from the wrong branch')
        subprocess.run(['git','add',str(target.relative_to(repo))],cwd=repo,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet','--',str(target.relative_to(repo))],cwd=repo).returncode
        if changed:
            subprocess.run(['git','commit','-m','Archive controlled proxy study measurements and independent audit'],cwd=repo,check=True)
        subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git',
                        'HEAD:refs/heads/experiment/a100-precommercial-v7-proxy-study-18h'],cwd=repo,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--push',action='store_true');a=p.parse_args()
    if a.watch:
        while True:
            state=ROOT/'state.json'
            if state.exists() and json.loads(state.read_text()).get('window_finished'):
                break
            time.sleep(30)
    archive(push=a.push)
