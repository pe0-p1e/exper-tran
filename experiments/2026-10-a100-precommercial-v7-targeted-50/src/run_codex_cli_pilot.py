"""Six blinded, fresh-context image calls through the user's existing Codex login."""
import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from proxy_study_protocol import OUT, ROOT, sha, atomic_json


def checked_order(base, images):
    if images not in (3,10):
        raise ValueError('Only approved three- or ten-image cohorts are supported')
    secret=json.loads((base/'private/manifest.json').read_text())
    order=json.loads((base/'request_order.json').read_text())
    ids=[rid for c in secret['cases'] for rid in c['requests'].values()]
    if (len(secret['cases'])!=images or len(order)!=2*images or len(set(order))!=2*images
            or len(ids)!=2*images or set(ids)!=set(order)
            or any(set(c['requests'])!={'clean','adversarial'} for c in secret['cases'])
            or len({c['image_id'] for c in secret['cases']})!=images):
        raise ValueError('Frozen image count and request manifest differ')
    declared={x['request_id']:x for x in secret['requests']}
    for rid in order:
        p=base/'blinded'/rid
        if sha(p/'image.png')!=declared[rid]['image_sha256'] or sha(p/'prompt.txt')!=declared[rid]['prompt_sha256']:
            raise ValueError('Frozen request changed')
    return order


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--limit',type=int)
    p.add_argument('--images',type=int,choices=[3,10],default=3)
    p.add_argument('--pack-dir',type=Path,default=OUT/'commercial_pilot3')
    p.add_argument('--result-dir',type=Path)
    a=p.parse_args()
    limit=a.limit if a.limit is not None else 2*a.images
    if not 1<=limit<=2*a.images:
        raise ValueError('Request limit exceeds the approved cohort')
    audit=ROOT/'historical_audit/independent_audit.json'
    if not audit.exists() or json.loads(audit.read_text())['status']!='PASS_WITH_CAVEATS':
        raise RuntimeError('Complete the historical audit before commercial validation')
    auth=subprocess.run(['codex','login','status'],capture_output=True,text=True)
    if 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:
        raise RuntimeError('Require the configured ChatGPT CLI subscription; no paid API fallback')
    cache=json.loads((Path.home()/'.codex/models_cache.json').read_text())
    model=next((m for m in cache['models'] if m['slug']==a.model),None)
    if model is None or 'image' not in model.get('input_modalities',[]):
        raise RuntimeError('Requested model lacks a verified image-capable local catalog entry')
    base=a.pack_dir; dest=a.result_dir or ROOT/'commercial_cli/codex'/a.model
    version=subprocess.check_output(['codex','--version'],text=True).strip()
    request_ids=checked_order(base,a.images)[:limit]
    identity=dict(requested_model=a.model,client=version,route='Codex CLI / existing ChatGPT login',
                  image_input='direct --image attachment',reasoning_effort='low',
                  fresh_process=True,ephemeral=True,model_catalog_entry=model,
                  hidden_target_weights='unavailable; no BF16 claim about closed weights',
                  actual_server_snapshot='not disclosed by this CLI; explicit requested model only')
    dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/'client_audit.json',identity)
    for rid in request_ids:
        source=base/'blinded'/rid; image=source/'image.png'; prompt=source/'prompt.txt'
        record=dest/rid/'result.json'
        expected=dict(image_sha256=sha(image),prompt_sha256=sha(prompt),client=version,model=a.model)
        if record.exists():
            old=json.loads(record.read_text())
            if old['identity']!=expected:
                raise RuntimeError('Pilot identity changed; use another study directory')
            print('[resume pilot]',rid,flush=True);continue
        record.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='imagecheck-',dir='/tmp') as tmp:
            # No repository, manifests, ground truth, attack state or previous responses in CWD.
            opaque=Path(tmp)/'image.png';shutil.copyfile(image,opaque)
            result=record.parent/'answer.txt'
            cmd=['codex','exec','--ignore-user-config','--ignore-rules','--skip-git-repo-check',
                 '--ephemeral','--sandbox','read-only','--json','--cd',tmp,'--model',a.model,
                 '-c','project_doc_max_bytes=0','-c','model_reasoning_effort="low"',
                 '-c','web_search="disabled"','--output-last-message',str(result)]
            for feature in ['apps','shell_tool','unified_exec','multi_agent','browser_use',
                            'computer_use','image_generation','skill_search','view_image','hooks']:
                cmd+=['--disable',feature]
            cmd+=['--enable','skip_host_skill_discovery','--image',str(opaque),'-']
            began=time.time()
            try:
                run=subprocess.run(cmd,input=prompt.read_text(),capture_output=True,text=True,timeout=180)
            except subprocess.TimeoutExpired as exc:
                atomic_json(record.parent/'transport_error.json',dict(kind='timeout',seconds=time.time()-began))
                raise RuntimeError('CLI request timed out; do not count as a completed observation') from exc
            (record.parent/'stdout.jsonl').write_text(run.stdout)
            (record.parent/'stderr.txt').write_text(run.stderr)
            if run.returncode:
                atomic_json(record.parent/'transport_error.json',dict(kind='nonzero_exit',exit_code=run.returncode,seconds=time.time()-began))
                raise RuntimeError('Codex request failed; inspect local raw traces')
            events=[json.loads(line) for line in run.stdout.splitlines() if line.strip().startswith('{')]
            items=[e.get('item',{}) for e in events if e.get('type')=='item.completed']
            # CLI feature warnings are emitted as error items, not model tool calls.
            warnings=[e for e in items if e.get('type')=='error']
            tools=[e for e in items if e.get('type') not in ['agent_message','reasoning','error']]
            if tools:
                raise RuntimeError('Unexpected tool use invalidates the blinded pilot')
            threads=[e['thread_id'] for e in events if e.get('type')=='thread.started']
            if len(threads)!=1 or not any(e.get('type')=='turn.completed' for e in events):
                raise RuntimeError('Missing fresh session or completed response; preserve trace before retry')
            answer=result.read_text().strip() if result.exists() else ''
            messages=[e.get('text','').strip() for e in items if e.get('type')=='agent_message']
            if not messages or messages[-1]!=answer:
                raise RuntimeError('Final answer differs from structured trace')
            line=next((s.strip() for s in answer.splitlines() if s.strip()),'')
            label=int(line)+1 if line in tuple('0123456789') else None
            atomic_json(record,dict(identity=expected,request_id=rid,started=began,finished=time.time(),
                        raw_output=answer,parsed_label=label,parser_status='ok' if label is not None else 'invalid',
                        session_id=threads[0],
                        tool_calls=0,client_warnings=warnings,CLI_context_differs_from_direct_API=True,
                        model_identity_evidence='explicit --model accepted; image modality in local catalog; server snapshot undisclosed'))
            print('[CLI pilot]',rid,'completed; parser', 'ok' if label is not None else 'invalid',flush=True)
    # Join labels only after all blinded model requests have completed.
    secret=json.loads((base/'private/manifest.json').read_text());measurements=[]
    for case in secret['cases']:
        rr={k:dest/v/'result.json' for k,v in case['requests'].items()}
        if not all(q.exists() for q in rr.values()):
            continue
        clean=json.loads(rr['clean'].read_text());adv=json.loads(rr['adversarial'].read_text())
        measurements.append(dict(transition=case['transition'],image_id=case['image_id'],
                clean_label=clean['parsed_label'],adv_label=adv['parsed_label'],
                clean_correct=clean['parsed_label']==case['source_label'],
                target_hit=adv['parsed_label']==case['target_label']))
    atomic_json(dest/'pilot_summary.json',dict(images=measurements,hits=sum(x['target_hit'] for x in measurements),
                denominator=a.images,completed_pairs=len(measurements),
                ASR=sum(x['target_hit'] for x in measurements)/a.images if len(measurements)==a.images else None,
                study=f'{a.images}-image frozen-method CLI pilot; exploratory small sample',
                target_feedback_for_tuning=False,model=a.model,actual_server_snapshot='not disclosed'))


if __name__=='__main__':
    main()
