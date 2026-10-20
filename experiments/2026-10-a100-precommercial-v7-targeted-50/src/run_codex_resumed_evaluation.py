"""Explicit quota-reset resumption: one account, one request, stop on any failure."""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from commercial_matrix_pack import MATRIX, prepare_pack
from proxy_study_protocol import EXP, atomic_json, sha
from run_commercial_matrix import entries, verified, aggregate


def archive_quota_only(folder):
    """Preserve a prior rejected request only when no model answer was observed."""
    trace=folder/'stdout.jsonl'
    if not trace.exists():
        if (folder/'transport_error.json').exists():
            raise RuntimeError('Ambiguous prior transport failure requires review: '+str(folder))
        return
    events=[json.loads(s) for s in trace.read_text().splitlines() if s.strip()]
    if any(e.get('type')=='turn.completed' or
           (e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message') for e in events):
        raise RuntimeError('An observed answer cannot be retried: '+str(folder))
    if not any('hit your usage limit' in json.dumps(e).lower() for e in events):
        raise RuntimeError('Prior failure was not an explicit quota rejection: '+str(folder))
    archive=folder.parent/'failed_transport_attempts'/(folder.name+'_'+str(time.time_ns()))
    archive.parent.mkdir(exist_ok=True);shutil.move(str(folder),archive)
    atomic_json(archive/'resumption_audit.json',dict(reason='User requested retry after quota recovery',
        no_answer_observed=True,original_trace_sha256=sha(archive/'stdout.jsonl'),at=time.time()))


def main():
    if any(os.environ.get(k) for k in ['OPENAI_API_KEY','CODEX_API_KEY']):
        raise RuntimeError('Unexpected API-key environment; subscription route only')
    with (MATRIX/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        models=entries(json.loads((EXP/'commercial_model_matrix.json').read_text()))
        rows=list(csv.DictReader((MATRIX/'stages/dev10/reports/models.csv').open()))
        eligible={r['matrix_id'] for r in rows if r['status']=='COMPLETE' and r['completed_requests']=='20'}
        chosen=[m for m in models if m['provider']=='codex' and m['id'] in eligible]
        base,order,secret,declared=prepare_pack('test150')
        hashes={r:{k:declared[r][k] for k in ['image_sha256','prompt_sha256']} for r in order}
        state=dict(pid=os.getpid(),started=time.time(),status='RUNNING',models=[m['model'] for m in chosen],
                   concurrent_requests=1,min_interval_seconds=10,stop_on_any_failure=True,
                   model_switch_after_limit=False,automatic_retry=False,completed={})
        status={m['id']:dict(status='QUEUED') for m in models}
        def save():atomic_json(MATRIX/'resumed_gpt_state.json',state)
        save()
        for spec in chosen:
            dest=MATRIX/'stages/test150/commercial_cli/codex'/spec['model'];dest.mkdir(parents=True,exist_ok=True)
            status[spec['id']]=dict(status='RUNNING')
            for index,rid in enumerate(order):
                record=dest/rid/'result.json';expected={**hashes[rid],'client':spec['client_version'],'model':spec['model']}
                if record.exists():verified(record,expected);continue
                archive_quota_only(record.parent)
                # Reuse the single successful reset check, without repeating its inference.
                probe=MATRIX/'access_checks/20261020T0212_gpt61'
                if spec['model']=='gpt-6.1-sol' and (probe/rid/'result.json').exists():
                    verified(probe/rid/'result.json',expected)
                    shutil.copytree(probe/rid,record.parent)
                    shutil.copyfile(probe/'client_audit.json',dest/'client_audit.json')
                    atomic_json(record.parent/'matrix_reuse.json',dict(source=str(probe/rid/'result.json'),new_inference=False))
                else:
                    began=time.monotonic()
                    log=dest/'resume_calls.log'
                    with log.open('a') as stream:
                        run=subprocess.run([sys.executable,str(EXP/'src/run_codex_cli_pilot.py'),
                            '--model',spec['model'],'--images','150','--limit',str(index+1),
                            '--pack-dir',str(base),'--result-dir',str(dest)],stdout=stream,stderr=subprocess.STDOUT)
                    if run.returncode:
                        state.update(status='BLOCKED',model=spec['model'],request_id=rid,
                            reason='Request failed; entire account queue stopped; inspect retained trace',finished=time.time())
                        status[spec['id']]=dict(status='BLOCKED',reason=state['reason'])
                        save();aggregate('test150',secret,models,status,hashes)
                        return
                    verified(record,expected)
                    time.sleep(max(0,10-(time.monotonic()-began)))
                state['completed'][spec['model']]=index+1;save()
                print(spec['model'],index+1,'/',len(order),flush=True)
                if (index+1)%10==0:aggregate('test150',secret,models,status,hashes)
            status[spec['id']]=dict(status='COMPLETE')
            aggregate('test150',secret,models,status,hashes)
        state.update(status='COMPLETE',finished=time.time());save()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(MATRIX/'resumed_gpt_unhandled_error.json',dict(status='BLOCKED',
            at=time.time(),reason=f'{type(exc).__name__}: {exc}',automatic_retry=False))
        raise
