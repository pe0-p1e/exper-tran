"""One serial request lane per account route; independent providers run concurrently."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from commercial_matrix_pack import MATRIX,STAGES,freeze
from run_commercial_matrix import entries,reuse,verified
from commercial_cli_adapters import preflight,InterfaceBlocker
from proxy_study_protocol import OUT,EXP,atomic_json
import run_commercial_cli as adapter


def helpers(dest):
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(b'run_codex_cli_pilot.py' in a for a in cmd) and str(dest).encode() in cmd:found.append(int(p.parent.name))
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',choices=['codex','gemini_agy','claude'],required=True);args=p.parse_args()
    cfg=json.loads((EXP/'commercial_model_matrix.json').read_text());models=[s for s in entries(cfg) if s['provider']==args.provider]
    folder=MATRIX/'lanes';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(args.provider+'.lock')).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(provider=args.provider,pid=os.getpid(),models={},started=time.time(),completed=False)
        def save():atomic_json(folder/(args.provider+'.json'),state)
        save();eligible=set()
        provider_blocker=None
        for stage in ['dev10','test150']:
            base=OUT/('commercial_pilot10' if stage=='dev10' else 'commercial_test150')
            secret=json.loads((base/'private/manifest.json').read_text());order=json.loads((base/'request_order.json').read_text())
            hashes={r['request_id']:{k:r[k] for k in ['image_sha256','prompt_sha256']} for r in secret['requests']}
            adapter.ROOT=MATRIX/'stages'/stage;adapter.PACK=base
            for spec in models:
                key=spec['id']+':'+stage
                if stage=='test150' and spec['id'] not in eligible:
                    state['models'][key]=dict(status='BLOCKED',reason='Ten-image interface incomplete; no ASR selection');save();continue
                try:
                    preflight(spec['provider'],spec);freeze(MATRIX/'model_locks'/(spec['id']+'.json'),spec)
                    state['models'][key]=dict(status='RUNNING',started=time.time());save()
                    dest=MATRIX/'stages'/stage/'commercial_cli'/spec['provider']/spec['model']
                    # Old commercial coordinators may have been replaced while a
                    # standalone Codex batch continues. Adopt its files, not a second request.
                    while helpers(dest):
                        state['models'][key]['adopting_helper_pids']=helpers(dest);save();time.sleep(3)
                    reuse(dest,spec,order,hashes)
                    if spec['provider']=='codex':
                        subprocess.run([sys.executable,str(EXP/'src/run_codex_cli_pilot.py'),'--model',spec['model'],
                            '--images',str(STAGES[stage]),'--pack-dir',str(base),'--result-dir',str(dest)],check=True)
                    else:
                        for i,rid in enumerate(order):
                            adapter.invoke(spec['provider'],spec,rid,hashes)
                            state['models'][key]['completed_requests']=i+1;save()
                            print(args.provider,stage,spec['model'],i+1,'/',len(order),flush=True)
                    sessions=[]
                    for rid in order:
                        d=verified(dest/rid/'result.json',{**hashes[rid],'client':spec['client_version'],'model':spec['model']});sessions.append(d['session_id'])
                    if len(set(sessions))!=len(order):raise InterfaceBlocker('Repeated target context')
                    state['models'][key]=dict(status='COMPLETE',finished=time.time(),completed_requests=len(order));save()
                    if stage=='dev10':eligible.add(spec['id'])
                except (InterfaceBlocker,subprocess.SubprocessError,RuntimeError) as exc:
                    state['models'][key]=dict(status='BLOCKED',reason=str(exc),finished=time.time());save()
                    print(args.provider,stage,spec['model'],'BLOCKED',str(exc),flush=True)
                    if 'catalog unavailable' in str(exc).lower():
                        # An account-level Antigravity denial affects every model
                        # on this route. Record the exact client response once and
                        # do not issue more account requests or retry model calls.
                        try:
                            check=subprocess.run(['agy','models'],capture_output=True,text=True,timeout=30)
                            provider_blocker=(check.stdout+'\n'+check.stderr).strip()[-4000:]
                        except Exception as audit_exc:
                            provider_blocker=f'{type(audit_exc).__name__}: {audit_exc}'
                        for remaining in models[models.index(spec)+1:]:
                            state['models'][remaining['id']+':'+stage]=dict(
                                status='BLOCKED',reason='Provider account eligibility blocked; no model request issued',
                                provider_error=provider_blocker,finished=time.time())
                        save()
                        break
            if provider_blocker:
                for remaining_stage in ['dev10','test150']:
                    if remaining_stage==stage:continue
                    for blocked_spec in models:
                        state['models'][blocked_spec['id']+':'+remaining_stage]=dict(
                            status='BLOCKED',reason='Provider account eligibility blocked; no model request issued',
                            provider_error=provider_blocker,finished=time.time())
                break
            if stage=='dev10':state['ten_finished']=time.time();state['full_eligible']=sorted(eligible);save()
        state['completed']=True;state['finished']=time.time();save()


if __name__=='__main__':main()
