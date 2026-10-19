"""Resumable fixed-method comparisons across explicit commercial model versions."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from proxy_study_protocol import EXP, ROOT, OUT, sha, atomic_json
from commercial_cli_adapters import preflight, InterfaceBlocker, parse_label, json_events, parse_claude
from commercial_matrix_pack import MATRIX, STAGES, prepare_pack, freeze
import run_commercial_cli as adapter


def entries(config):
    if config.get('paid_api_fallback') is not False or config.get('target_feedback_for_tuning') is not False:
        raise ValueError('Paid fallback and target-feedback method selection are disabled')
    if config.get('codex_reasoning_effort')!='low':raise ValueError('GPT effort must stay low in the matched matrix')
    result=[x for x in config['models'] if x['enabled']]
    ids=[x['id'] for x in result]
    if len(ids)!=len(set(ids)) or any(not x.replace('_','').isalnum() for x in ids):raise ValueError('Invalid/duplicate matrix IDs')
    keys=[(x['provider'],x['model']) for x in result if x['model']]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate provider/model rows must reuse one condition')
    if any(x['provider'] not in ['codex','gemini','gemini_agy','claude'] for x in result):raise ValueError('Unknown provider')
    return result


def verified(path,expected):
    d=json.loads(path.read_text());adapter.validate_record(d,expected)
    session=d.get('session_id')
    if expected['client'].startswith('codex-cli'):
        audit=json.loads((path.parent.parent/'client_audit.json').read_text())
        if audit.get('reasoning_effort')!='low' or audit.get('requested_model')!=expected['model']:
            raise InterfaceBlocker('Codex client model/effort audit differs')
        events=json_events((path.parent/'stdout.jsonl').read_text())
        threads=[e['thread_id'] for e in events if e.get('type')=='thread.started']
        items=[e['item'] for e in events if e.get('type')=='item.completed']
        if len(threads)!=1 or not any(e.get('type')=='turn.completed' for e in events):raise InterfaceBlocker('Invalid Codex terminal/session trace')
        if any(e.get('type') not in ['agent_message','reasoning','error'] for e in items):raise InterfaceBlocker('Unexpected Codex model tool')
        messages=[e.get('text','').strip() for e in items if e.get('type')=='agent_message']
        if not messages or messages[-1]!=d['raw_output'].strip():raise InterfaceBlocker('Answer/trace differs')
        session=threads[0]
    elif expected['client']=='1.4.2':
        media=d.get('evidence',{}).get('native_media',{})
        if media.get('status')!='verified' or media.get('image_sha256')!=expected['image_sha256']:
            raise InterfaceBlocker('Missing native Antigravity PNG hash evidence')
        events=json_events((path.parent/d['attempt']/'stdout.jsonl').read_text())
        initial=next((e['init'] for e in events if e.get('event')=='init'),{})
        answer,session,evidence=parse_claude(events,expected['model'],Path(initial.get('cwd',''))/'image.png',media)
        if answer!=d['raw_output']:raise InterfaceBlocker('Native answer/trace differs')
    if not session:raise InterfaceBlocker('Missing isolated target session')
    return dict(d,session_id=session)


def reuse(dest,spec,order,hashes):
    sources=[ROOT/'commercial_cli'/spec['provider']/spec['model'],
             ROOT/'commercial_ten_image/codex'/spec['model'],
             MATRIX/'interface_validation/commercial_cli'/spec['provider']/spec['model']]
    sources += [MATRIX/'stages'/s/'commercial_cli'/spec['provider']/spec['model'] for s in STAGES]
    # Identical image/prompt/model/client only; never reuse across model versions.
    for rid in order:
        target=dest/rid/'result.json'
        expected={**hashes[rid],'model':spec['model'],'client':spec['client_version']}
        if target.exists():verified(target,expected);continue
        for source in sources:
            record=source/rid/'result.json'
            if not record.exists():continue
            if json.loads(record.read_text()).get('identity')!=expected:continue
            verified(record,expected)
            if target.parent.exists():raise InterfaceBlocker('Partial destination requires review before cache adoption')
            shutil.copytree(record.parent,target.parent)
            atomic_json(target.parent/'matrix_reuse.json',dict(source=str(record),source_sha256=sha(record),new_inference=False))
            break


def aggregate(stage,secret,models,status,hashes):
    stage_root=MATRIX/'stages'/stage;rows=[];images=[];directions=[]
    n=STAGES[stage]
    for spec in models:
        model=spec.get('model');folder=stage_root/'commercial_cli'/spec['provider']/(model or '_unconfigured')
        pairrows=[];sessions=[];requests=0
        for c in secret['cases']:
            pair={}
            for condition,rid in c['requests'].items():
                p=folder/rid/'result.json'
                if p.exists():
                    d=verified(p,{**hashes[rid],'model':model,'client':spec['client_version']})
                    pair[condition]=d;requests+=1;sessions.append(d['session_id'])
            done=len(pair)==2
            if done:
                row=dict(matrix_id=spec['id'],provider=spec['provider'],model=model,transition=c['transition'],image_id=c['image_id'],
                        source_label=c['source_label'],target_label=c['target_label'],clean_raw=pair['clean']['raw_output'],
                        adversarial_raw=pair['adversarial']['raw_output'],clean_label=pair['clean']['parsed_label'],
                        adv_label=pair['adversarial']['parsed_label'],clean_correct=pair['clean']['parsed_label']==c['source_label'],
                        target_hit=pair['adversarial']['parsed_label']==c['target_label'],
                        invalid_replies=sum(d['parsed_label'] is None for d in pair.values()))
                pairrows.append(row);images.append(row)
        if len(sessions)!=len(set(sessions)):raise InterfaceBlocker('Repeated session across images')
        complete=len(pairrows)==n
        hits=sum(x['target_hit'] for x in pairrows);cc=[x for x in pairrows if x['clean_correct']]
        rows.append(dict(matrix_id=spec['id'],provider=spec['provider'],model=model or 'UNCONFIGURED',
                    status='COMPLETE' if complete else status.get(spec['id'],{}).get('status','QUEUED'),
                    completed_requests=requests,planned_requests=2*n,hits=hits if complete else None,
                    denominator=n,ASR=hits/n if complete else None,observed_hits=hits,observed_pairs=len(pairrows),clean_correct=len(cc) if complete else None,
                    clean_correct_conditional_hits=sum(x['target_hit'] for x in cc) if complete else None,
                    clean_correct_conditional_denominator=len(cc) if complete else None,
                    clean_correct_conditional_ASR=sum(x['target_hit'] for x in cc)/len(cc) if complete and cc else None,
                    invalid_replies=sum(x['invalid_replies'] for x in pairrows),
                    reason=status.get(spec['id'],{}).get('reason',''),legacy_conditional_TASR='N/A'))
        for tid in sorted({c['transition'] for c in secret['cases']}):
            planned=sum(c['transition']==tid for c in secret['cases']);part=[x for x in pairrows if x['transition']==tid]
            directions.append(dict(matrix_id=spec['id'],transition=tid,hits=sum(x['target_hit'] for x in part) if len(part)==planned else None,
                                   denominator=planned,completed_pairs=len(part)))
    report=stage_root/'reports';report.mkdir(parents=True,exist_ok=True)
    for filename,table in [('models.csv',rows),('images.csv',images),('directions.csv',directions)]:
        path=report/filename
        if table:
            with path.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        elif path.exists():path.unlink()
    atomic_json(report/'summary.json',dict(stage=stage,updated=time.time(),models=rows,
                all_complete=all(r['status']=='COMPLETE' for r in rows),final_target_feedback_for_tuning=False))
    text=['# Frozen-method commercial version comparison','',f'Stage: {stage}; same fixed {n} images per model; clean/adversarial in fresh separate sessions.',
          'Invalid predictions remain misses. Missing access/transport observations are N/A. All parameters/method/images are frozen before requests.',
          'Model IDs are explicit requested client IDs; backend serving snapshots and closed weights are not exposed.',
          'CLI routes and prompts are logged; different clients are not claimed to have identical system context.', '',
          '| Model | Route | Status | Requests | Target hits / N | ASR |','|---|---|---|---|---|---|']
    for r in rows:text.append(f"| {r['model']} | {r['provider']} | {r['status']} | {r['completed_requests']}/{r['planned_requests']} | {r['hits'] if r['hits'] is not None else 'N/A'}/{n} | {r['ASR'] if r['ASR'] is not None else 'N/A'} |")
    (report/'report.md').write_text('\n'.join(text)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=EXP/'commercial_model_matrix.json')
    p.add_argument('--stage',choices=STAGES,default='dev10');p.add_argument('--preflight',action='store_true')
    p.add_argument('--all-stages',action='store_true');args=p.parse_args()
    MATRIX.mkdir(parents=True,exist_ok=True)
    with (MATRIX/'campaign.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads(args.config.read_text());models=entries(config)
        audit=ROOT/'historical_audit/independent_audit.json'
        if not audit.exists() or json.loads(audit.read_text()).get('status')!='PASS_WITH_CAVEATS':
            raise RuntimeError('Existing cohort/PNG/BF16 source audit must pass before commercial work')
        stages=['smoke3','dev10','test150'] if args.all_stages else [args.stage]
        if not args.preflight:
            # Missing model IDs may be resolved after authentication; an executed
            # model row is separately immutable. Never freeze unresolved null IDs.
            freeze(MATRIX/'model_matrix_lock.json',dict(config={k:v for k,v in config.items() if k!='models'},adapter_sha256=sha(EXP/'src/commercial_cli_adapters.py'),
                  codex_adapter_sha256=sha(EXP/'src/run_codex_cli_pilot.py'),
                  shared_invocation_sha256=sha(EXP/'src/run_commercial_cli.py'),
                  matrix_runner_sha256=sha(Path(__file__)),pack_builder_sha256=sha(EXP/'src/commercial_matrix_pack.py'),
                  no_automatic_retry=True,model_choice_by_access_not_ASR=True))
        all_done=True;global_status={}
        for stage in stages:
            base,order,secret,declared=prepare_pack(stage)
            hashes={rid:{k:declared[rid][k] for k in ['image_sha256','prompt_sha256']} for rid in order}
            stage_root=MATRIX/'stages'/stage;status={}
            adapter.ROOT=stage_root;adapter.PACK=base
            if not args.preflight:freeze(stage_root/'cohort_lock.json',dict(manifest_sha256=sha(base/'private/manifest.json'),request_order=order))
            for spec in models:
                key=spec['id'];previous=global_status.get(key)
                try:
                    # Access/interface blockers in an earlier stage must not fan out.
                    if previous and previous['status']=='BLOCKED':raise InterfaceBlocker(previous['reason'])
                    preflight(spec['provider'],spec)
                    status[key]=dict(status='READY')
                    print(f"[{stage}] {key}: READY {spec['model']}",flush=True)
                    if args.preflight:continue
                    freeze(MATRIX/'model_locks'/(key+'.json'),spec)
                    dest=stage_root/'commercial_cli'/spec['provider']/spec['model']
                    reuse(dest,spec,order,hashes)
                    if spec['provider']=='codex':
                        subprocess.run([sys.executable,str(EXP/'src/run_codex_cli_pilot.py'),'--model',spec['model'],
                           '--images',str(STAGES[stage]),'--pack-dir',str(base),'--result-dir',str(dest)],check=True)
                    else:
                        for i,rid in enumerate(order):
                            adapter.invoke(spec['provider'],spec,rid,hashes,retry_failed=False)
                            print(f'[{stage}] {key}: {i+1}/{len(order)} requests',flush=True)
                    status[key]=dict(status='COMPLETE')
                except (InterfaceBlocker,subprocess.SubprocessError,RuntimeError) as exc:
                    status[key]=dict(status='BLOCKED',reason=str(exc))
                    print(f'[{stage}] {key}: BLOCKED {exc}',flush=True)
                if not args.preflight:aggregate(stage,secret,models,status,hashes)
            global_status.update(status)
            rows=aggregate(stage,secret,models,status,hashes)
            atomic_json(stage_root/'preflight.json',dict(config_sha256=sha(args.config),providers=status,
                        live_inference_attempted=not args.preflight,at=time.time(),requested_images=STAGES[stage]))
            if not all(r['status']=='COMPLETE' for r in rows):all_done=False
            print('Report:',stage_root/'reports/report.md',flush=True)
        return 0 if all_done else 2


if __name__=='__main__':raise SystemExit(main())
