"""Publish compact PP4 diagnostic records; never upload images, weights or credentials."""
import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pp4_protocol import ROOT,PREVIOUS,EXP,atomic_json


def publish():
    repo=EXP.parents[1];dest=EXP/'reports/pull_push_diagnostics';dest.mkdir(parents=True,exist_ok=True)
    patterns=['protocol_lock.json','progress.json','completion.json','commercial_approval.json',
              'resource_status.json','configs/*.json','reuse/*.json','validation/*/*.json','reports/*.csv','reports/*.md']
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            target=dest/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    audit=ROOT/'audits/model_dtype_audit.json'
    if audit.exists():
        events=json.loads(audit.read_text());latest={}
        for e in events:latest[(e['model_name'],e['role'])]=e
        atomic_json(dest/'bf16_audit.json',dict(load_events=len(events),any_failed=any(e['status']!='pass' for e in events),audits=list(latest.values())))
    with (PREVIOUS/'publication.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',str(dest.relative_to(repo))],cwd=repo).returncode:
            subprocess.run(['git','commit','-m','Publish bounded four-layer Pull Push diagnostics'],cwd=repo,check=True)
        subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:experiment/a100-precommercial-v7-proxy-study-18h'],cwd=repo,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        while not (ROOT/'completion.json').exists():time.sleep(30)
    publish()
