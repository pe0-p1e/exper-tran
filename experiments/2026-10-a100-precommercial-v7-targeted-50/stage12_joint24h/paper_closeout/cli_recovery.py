"""Bounded, auditable transport-only recovery; never retry a scored response."""
import json, re, shutil, subprocess, time
from pathlib import Path
import common as C

MAX_TIMEOUT_RETRIES = 3
REQUEST_TIMEOUT = 600

def recover_timeouts(root, model, order):
    """Archive only timed-out unscored attempts. Unknown/complete traces fail closed."""
    recovered=0
    for rid in order:
        folder=root/'responses'/model/rid
        if (folder/'result.json').exists():
            continue
        error=C.read(folder/'transport_error.json')
        if not error:
            continue
        if error.get('kind')!='timeout':
            raise RuntimeError('Non-timeout failure requires review: '+rid)
        events=[]
        for name in ['stdout.jsonl','timeout_stdout.jsonl']:
            if (folder/name).exists():
                for line in (folder/name).read_text(errors='replace').splitlines():
                    if line.strip().startswith('{'):
                        try: events.append(json.loads(line))
                        except json.JSONDecodeError: raise RuntimeError('Malformed interrupted trace: '+rid)
        if any(e.get('type')=='turn.completed' or e.get('item',{}).get('type')=='agent_message' for e in events) or (folder/'answer.txt').exists():
            raise RuntimeError('Potential returned answer requires manual reconciliation: '+rid)
        text=' '.join((folder/n).read_text(errors='replace') for n in ['stderr.txt','timeout_stderr.txt'] if (folder/n).exists()).lower()
        if any(s in text for s in ['429','rate limit','quota','authentication','unauthorized']):
            raise RuntimeError('Quota/auth failure; no automatic retry: '+rid)
        attempts=folder/'failed_attempts';n=len(list(attempts.glob('attempt_*'))) if attempts.exists() else 0
        if n>=MAX_TIMEOUT_RETRIES:
            raise RuntimeError('Timeout retry limit reached: '+rid)
        dest=attempts/f'attempt_{n+1:02d}';dest.mkdir(parents=True)
        names=[p for p in folder.iterdir() if p.is_file() and p.name!='retry_history.json']
        evidence={p.name:C.sha(p) for p in names}
        for p in names: shutil.move(str(p),str(dest/p.name))
        C.atomic(dest/'audit.json',dict(request_id=rid,model=model,retry=n+1,at=time.time(),files=evidence,
            previous_scored_response=False,server_completion_unknown=True,reason='transport timeout; no returned observable answer',
            science_unchanged=True,timeout_seconds=REQUEST_TIMEOUT))
        C.atomic(folder/'retry_history.json',dict(transport_retries=n+1,all_failed_attempts_retained=True,one_scored_result_only=True))
        recovered+=1
    return recovered

def install_timeout_capture():
    """Adapt timeout/capture only; leave the immutable legacy isolation/parser intact."""
    original=subprocess.run
    def run(cmd,*args,**kwargs):
        is_request=isinstance(cmd,list) and len(cmd)>1 and cmd[:2]==['codex','exec']
        if not is_request: return original(cmd,*args,**kwargs)
        kwargs['timeout']=REQUEST_TIMEOUT
        try: return original(cmd,*args,**kwargs)
        except subprocess.TimeoutExpired as exc:
            folder=Path(cmd[cmd.index('--output-last-message')+1]).parent
            for name,value in [('timeout_stdout.jsonl',exc.stdout),('timeout_stderr.txt',exc.stderr)]:
                if value is not None:
                    (folder/name).write_text(value.decode(errors='replace') if isinstance(value,bytes) else value)
            raise
    subprocess.run=run


class QuotaAuthBlock(RuntimeError):
    pass

def fatal_transport(root, model, order):
    for rid in order:
        folder=root/'responses'/model/rid
        if (folder/'result.json').exists() or not (folder/'transport_error.json').exists(): continue
        text=' '.join((folder/n).read_text(errors='replace') for n in ['stderr.txt','timeout_stderr.txt','stdout.jsonl','timeout_stdout.jsonl'] if (folder/n).exists()).lower()
        if re.search(r'\b429\b|rate.limit|quota|authentication|unauthorized|not logged in',text):
            return dict(request_id=rid,reason='Quota/auth/rate-limit transport evidence; stop whole commercial route')
    return None

def recover_blocked(root, model, order, log):
    fatal=fatal_transport(root,model,order)
    if fatal:raise QuotaAuthBlock(fatal['reason'])
    n=recover_timeouts(root,model,order)
    if n:return dict(retried_timeout_attempts=n)
    # Another method may already have repaired the SAME shared clean request.
    # Reconcile only the exact last timeout request named in the retained log.
    text=Path(log).read_text(errors='replace')
    tail=text.rsplit('Traceback (most recent call last):',1)[-1]
    if 'RuntimeError: CLI request timed out; do not count as a completed observation' not in tail:
        raise RuntimeError('Unknown/non-timeout block requires manual review')
    ids=re.findall(r'/responses/'+re.escape(model)+r'/(r_[0-9a-f]+)/answer\.txt',text)
    if ids and ids[-1] in order:
        rid=ids[-1];result=C.read(root/'responses'/model/rid/'result.json')
        if result and result.get('request_id')==rid and result.get('identity',{}).get('model')==model:
            return dict(shared_timeout_already_resolved=rid,no_duplicate_request=True)
        folder=root/'responses'/model/rid;history=C.read(folder/'retry_history.json',{})
        attempts=sorted((folder/'failed_attempts').glob('attempt_*'))
        active=[folder/n for n in ['transport_error.json','stdout.jsonl','timeout_stdout.jsonl','answer.txt']]
        if attempts and len(attempts)==history.get('transport_retries') and len(attempts)<=MAX_TIMEOUT_RETRIES and not any(p.exists() for p in active):
            audit=C.read(attempts[-1]/'audit.json',{})
            if audit.get('request_id')==rid and audit.get('previous_scored_response') is False:
                return dict(archived_retry_ready=rid,retry_count_unchanged=True)
    raise RuntimeError('No reviewed timeout or resolved shared request is available')
