"""One resumable three-target CLI pilot. Missing access is not a zero ASR."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from proxy_study_protocol import OUT, ROOT, EXP, atomic_json, sha, fingerprint
from commercial_cli_adapters import (InterfaceBlocker, preflight, build, json_events,
                                      parse_label, parse_gemini, parse_claude, antigravity_media_audit)

CONTROL=ROOT/'commercial_cli_campaign'
PACK=OUT/'commercial_pilot3'


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise InterfaceBlocker('Frozen provider identity changed: '+str(path))
    else:
        atomic_json(path,value)


def verify_pack():
    import numpy as np
    from PIL import Image
    order=json.loads((PACK/'request_order.json').read_text())
    secret=json.loads((PACK/'private/manifest.json').read_text())
    if len(secret['cases'])!=3 or len(order)!=6 or len(set(order))!=6:
        raise InterfaceBlocker('Expected frozen three-image/six-request pilot')
    ids={rid for c in secret['cases'] for rid in c['requests'].values()}
    if ids!=set(order):raise InterfaceBlocker('Blinded order/manifest mismatch')
    hashes={}
    for case in secret['cases']:
        state=json.loads((OUT/case['source_state']).read_text())
        if state['target_access'] or state['steps']!=50 or state['seed']!=42:
            raise InterfaceBlocker('Attack provenance differs from frozen pilot')
        index=state['image_ids'].index(case['image_id'])
        pair=[]
        for condition,key in [('clean','clean_paths'),('adversarial','adv_paths')]:
            rid=case['requests'][condition];src=PACK/'blinded'/rid
            image=src/'image.png';prompt=src/'prompt.txt';original=OUT/state[key][index]
            if sha(image)!=sha(original) or state['png_hashes'][state[key][index]]!=sha(image):
                raise InterfaceBlocker('Pilot PNG no longer matches frozen attack')
            pixels=np.asarray(Image.open(image).convert('RGB'))
            if pixels.shape!=(224,224,3):raise InterfaceBlocker('Incorrect pilot canvas')
            pair.append(pixels.astype('int16'))
            hashes[rid]=dict(image_sha256=sha(image),prompt_sha256=sha(prompt))
        if np.abs(pair[0]-pair[1]).max()>16:raise InterfaceBlocker('Pilot perturbation exceeds16/255')
    if len({x['prompt_sha256'] for x in hashes.values()})!=1:
        raise InterfaceBlocker('Different classification prompts')
    return order,secret,hashes


def validate_record(record,expected):
    if record.get('identity')!=expected:
        raise InterfaceBlocker('Cached model/client/image/prompt differs')
    if record.get('parsed_label')!=parse_label(record['raw_output']):
        raise InterfaceBlocker('Cached parser result differs')


def adopt_codex(spec,order,hashes):
    dest=ROOT/'commercial_cli/codex'/spec['model']
    sessions=[]
    if not all((dest/rid/'result.json').exists() for rid in order):
        subprocess.run([sys.executable,str(EXP/'src/run_codex_cli_pilot.py'),
                        '--model',spec['model']],check=True)
    for rid in order:
        folder=dest/rid;record=json.loads((folder/'result.json').read_text())
        validate_record(record,{**hashes[rid],'client':spec['client_version'],'model':spec['model']})
        events=json_events((folder/'stdout.jsonl').read_text())
        if not any(e.get('type')=='turn.completed' for e in events):
            raise InterfaceBlocker('Missing completed Codex trace')
        threads=[e['thread_id'] for e in events if e.get('type')=='thread.started']
        if len(threads)!=1:raise InterfaceBlocker('Missing fresh Codex session identity')
        sessions.extend(threads)
        items=[e['item'] for e in events if e.get('type')=='item.completed']
        if any(x.get('type') not in ['agent_message','reasoning','error'] for x in items):
            raise InterfaceBlocker('Unexpected Codex tool use in cached trace')
    if len(sessions)!=len(set(sessions)):raise InterfaceBlocker('Codex context reused across images')
    return dest


def invoke(provider,spec,rid,hashes,retry_failed=False):
    dest=ROOT/'commercial_cli'/provider/spec['model']/rid
    dest.mkdir(parents=True,exist_ok=True)
    identity={**hashes[rid],'client':spec['client_version'],'model':spec['model']}
    result=dest/'result.json'
    if result.exists():
        validate_record(json.loads(result.read_text()),identity);return
    attempts=list(dest.glob('attempt_*'))
    if attempts and not retry_failed:
        raise InterfaceBlocker('Prior failed interface attempt preserved; explicit retry required')
    attempt=dest/f'attempt_{len(attempts)+1:03d}';attempt.mkdir()
    atomic_json(attempt/'request_identity.json',dict(identity=identity,adapter_sha256=sha(EXP/'src/commercial_cli_adapters.py'),started=time.time()))
    with tempfile.TemporaryDirectory(prefix='imagecheck-',dir='/tmp') as folder:
        tmp=Path(folder);image=tmp/'image.png';shutil.copyfile(PACK/'blinded'/rid/'image.png',image)
        prompt=(PACK/'blinded'/rid/'prompt.txt').read_text()
        cmd,env=build(provider,spec,tmp,image,attempt)
        if provider in ('claude','gemini_agy'):
            cmd.append('Read image.png using view_file, then answer this question:\n'+prompt)
            stdin=None
        else:stdin=prompt
        start=time.time()
        try:
            run=subprocess.run(cmd,input=stdin,capture_output=True,text=True,cwd=tmp,env=env,timeout=210)
            (attempt/'stdout.jsonl').write_text(run.stdout);(attempt/'stderr.txt').write_text(run.stderr)
            if run.returncode:
                raise InterfaceBlocker('Client returned nonzero; inspect local trace')
            events=json_events(run.stdout)
            if provider=='gemini':
                p=attempt/'image_transport.jsonl'
                transport=json_events(p.read_text()) if p.exists() else []
                answer,session,evidence=parse_gemini(events,spec['model'],transport)
            else:
                media=antigravity_media_audit(events,sha(image))
                answer,session,evidence=parse_claude(events,spec['model'],image,media)
            label=parse_label(answer)
            atomic_json(result,dict(identity=identity,request_id=rid,raw_output=answer,parsed_label=label,
                        parser_status='ok' if label is not None else 'invalid',session_id=session,
                        execution_settings=dict(provider=provider,effort=spec.get('effort','provider_default' if provider=='claude' else 'low') if provider in ('claude','gemini_agy') else 'provider_default'),
                        started=start,finished=time.time(),evidence=evidence,attempt=attempt.name,
                        CLI_context_differs_from_direct_API=True,
                        serving_snapshot='not exposed; explicit model and observed client route only'))
        except (InterfaceBlocker,subprocess.TimeoutExpired) as exc:
            atomic_json(attempt/'interface_error.json',dict(kind=type(exc).__name__,reason=str(exc),
                        elapsed_seconds=time.time()-start,counts_as_target_prediction=False))
            raise InterfaceBlocker(str(exc)) from exc


def aggregate(config,secret,status):
    table=[];image_rows=[]
    for provider,spec in config['providers'].items():
        modelspec=spec.get('model');valid=0;hits=0;correct=0;completed=0
        dest=ROOT/'commercial_cli'/provider/(modelspec or '_unconfigured')
        sessions=[]
        for case in secret['cases']:
            pair={}
            for condition,rid in case['requests'].items():
                p=dest/rid/'result.json'
                if p.exists():
                    pair[condition]=json.loads(p.read_text());completed+=1
                    if pair[condition].get('session_id'):sessions.append(pair[condition]['session_id'])
            if len(pair)==2:
                clean=pair['clean']['parsed_label'];adv=pair['adversarial']['parsed_label'];valid+=1
                hit=adv==case['target_label'];cc=clean==case['source_label'];hits+=hit;correct+=cc
                image_rows.append(dict(provider=provider,model=modelspec,transition=case['transition'],
                                  image_id=case['image_id'],clean_label=clean,adv_label=adv,
                                  clean_correct=cc,target_hit=hit))
        if len(sessions)!=len(set(sessions)):
            raise InterfaceBlocker('Conversation was reused across images')
        complete=valid==3 and completed==6
        table.append(dict(provider=provider,model=modelspec or 'UNCONFIGURED',
                          status='COMPLETE' if complete else status.get(provider,{}).get('status','BLOCKED'),
                          completed_requests=completed,planned_requests=6,paired_images=valid,
                          hits=hits if complete else None,denominator=3,ASR=hits/3 if complete else None,
                          clean_correct=correct if complete else None,
                          reason=status.get(provider,{}).get('reason',''),
                          legacy_conditional_TASR='N/A'))
    reports=CONTROL/'reports';reports.mkdir(parents=True,exist_ok=True)
    for name,rows in [('targets.csv',table),('images.csv',image_rows)]:
        if not rows:continue
        with (reports/name).open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Three-image commercial CLI pilot','',
           'Connectivity/image-input validation only. Three difficult-direction images do not estimate broad commercial ASR.',
           'Serving snapshots and closed-weight precision are not observable. CLI context differs from direct API/browser chat.',
           'No target feedback changes attacks. Invalid replies are failed predictions; access/transport errors are missing observations.',
           '', '| Target route | Model | Status | Requests | Hits/3 |', '|---|---|---|---|---|']
    for r in table:lines.append(f"| {r['provider']} | {r['model']} | {r['status']} | {r['completed_requests']}/6 | {r['hits'] if r['hits'] is not None else 'N/A'}/3 |")
    (reports/'pilot_report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(CONTROL/'status.json',dict(updated=time.time(),providers=status,summary=table,all_complete=all(r['status']=='COMPLETE' for r in table)))
    return table


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=EXP/'commercial_cli_targets.json')
    p.add_argument('--preflight',action='store_true');p.add_argument('--providers',nargs='+',choices=['codex','gemini','claude'],default=['codex','gemini','claude'])
    p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (CONTROL/'campaign.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads(a.config.read_text())
        if config['paid_api_fallback'] or config['max_images']!=3 or config['requests_per_target']!=6:
            raise InterfaceBlocker('Only the approved three-image CLI pilot is implemented here')
        order,secret,hashes=verify_pack();status={}
        for provider in a.providers:
            spec=config['providers'][provider]
            try:
                ready=preflight(provider,spec);status[provider]=ready
                print('[CLI preflight]',provider,'READY',spec['model'],flush=True)
                if a.preflight:continue
                freeze(CONTROL/'locks'/(provider+'.json'),dict(provider=provider,spec=spec,images=hashes,
                       request_order=order,protocol=config['protocol'],selection='GT-only fixed pilot',target_feedback_for_tuning=False))
                if provider=='codex':adopt_codex(spec,order,hashes)
                else:
                    failures=[]
                    for rid in order:
                        try:
                            invoke(provider,spec,rid,hashes,retry_failed=a.retry_failed)
                            print('[CLI observation]',provider,rid,'complete',flush=True)
                        except InterfaceBlocker as exc:
                            failures.append(dict(request=rid,error=str(exc)))
                            print('[CLI local failure]',provider,rid,str(exc),flush=True)
                    if failures:status[provider]=dict(status='PARTIAL',reason='Interface errors',failures=failures)
            except (InterfaceBlocker,subprocess.SubprocessError) as exc:
                status[provider]=dict(status='BLOCKED',reason=str(exc))
                print('[CLI blocked]',provider,str(exc),flush=True)
        table=aggregate(config,secret,status)
        atomic_json(CONTROL/'preflight.json',dict(at=time.time(),providers=status,config_sha256=sha(a.config),
                    image_hashes=hashes,live_inference_attempted=not a.preflight))
        print('Report:',CONTROL/'reports/pilot_report.md',flush=True)
        return 0 if all(r['status']=='COMPLETE' for r in table) else 2


if __name__=='__main__':
    raise SystemExit(main())
