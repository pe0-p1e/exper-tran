"""CPU-only staged GPT evaluations; reuses tested isolated CLI, never paid APIs."""
import argparse,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as C
from paper_closeout.plan import OUT,MODELS,commercial_specs

def verify_models():
    catalog=C.read(Path.home()/'.codex/models_cache.json')['models'];out=[]
    for name in MODELS:
        if 'astra' in name.lower():raise RuntimeError('Astra is forbidden')
        m=next((v for v in catalog if v['slug']==name),None)
        if not m or 'image' not in m.get('input_modalities',[]) or not any(v['effort']=='low' for v in m.get('supported_reasoning_levels',[])):raise RuntimeError('Model/image/low not verified: '+name)
        out.append(dict(model=name,image=True,effort='low'))
    auth=subprocess.run(['codex','login','status'],capture_output=True,text=True)
    if 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:raise RuntimeError('Subscription auth unavailable; no new login or API fallback')
    C.atomic(OUT/'commercial_preflight.json',dict(models=out,client=subprocess.check_output(['codex','--version'],text=True).strip(),route='existing ChatGPT subscription CLI'))

def pack(study,method,c,model):
    if not C.attack_done(c):return None
    root=OUT/study;dest=root/'packs'/model/method
    prompt=next((C.ROOT/'stage2/packs/uniform/blinded').glob('*/prompt.txt')).read_text()
    cases=C.cohort(c['split'])['images'];cases=sorted(cases,key=lambda r:(C.digest(r['image_id']),r['transition']))
    # Interleave directions in every prefix so progress is not class-order biased.
    groups={t:[x for x in cases if x['transition']==t] for t in C.TIDS}
    cases=[groups[t][i] for i in range(len(cases)//5) for t in C.TIDS]
    reqs=[];secret=[];order=[];cache_roots=[C.ROOT/'stage2/responses',C.ROOT/'fda_selected_commercial/stage2/responses']
    version=subprocess.check_output(['codex','--version'],text=True).strip()
    for r in cases:
        state=C.read(C.image_receipt(c,r['image_id']));ids={}
        for kind,key in [('clean','clean'),('adversarial','adv')]:
            src=Path(state[key]);rid='r_'+C.digest(['V8',kind,C.sha(src),C.digest(prompt),model,'low'])[:24];ids[kind]=rid
            image=dest/'blinded'/rid/'image.png';image.parent.mkdir(parents=True,exist_ok=True)
            if not image.exists():shutil.copyfile(src,image)
            (image.parent/'prompt.txt').write_text(prompt)
            expected=dict(image_sha256=C.sha(src),prompt_sha256=C.sha(image.parent/'prompt.txt'),client=version,model=model)
            reqs.append(dict(request_id=rid,image_sha256=expected['image_sha256'],prompt_sha256=expected['prompt_sha256']));order.append(rid)
            result=root/'responses'/model/rid/'result.json'
            if not result.exists():
                for old in cache_roots:
                    cached=C.read(old/rid/'result.json')
                    if cached and cached.get('identity')==expected and cached.get('request_id')==rid:
                        result.parent.mkdir(parents=True,exist_ok=True)
                        for n in ['result.json','stdout.jsonl','stderr.txt','answer.txt']:
                            if (old/rid/n).exists():shutil.copyfile(old/rid/n,result.parent/n)
                        break
        secret.append(dict(image_id=r['image_id'],transition=r['transition'],source_label=r['source'],target_label=r['target'],requests=ids))
    C.freeze(dest/'private/manifest.json',dict(cases=secret,requests=reqs));C.freeze(dest/'request_order.json',order)
    C.freeze(dest/'method_lock.json',dict(condition=c,model=model,effort='low',N=len(cases),selection_feedback=study=='commercial50'))
    return dest

def summarize(study):
    root=OUT/study;rows=[]
    for p in sorted((root/'packs').glob('*/*')):
        d=C.read(p/'private/manifest.json')
        if not d:continue
        model=p.parent.name;items=[]
        for r in d['cases']:
            paths={k:root/'responses'/model/rid/'result.json' for k,rid in r['requests'].items()}
            if not all(q.exists() for q in paths.values()):continue
            clean,adv=C.read(paths['clean']),C.read(paths['adversarial'])
            items.append(dict(image_id=r['image_id'],transition=r['transition'],target_hit=adv['parsed_label']==r['target_label'],
              clean_correct=clean['parsed_label']==r['source_label'],clean_label=clean['parsed_label'],adv_label=adv['parsed_label'],parser_status=adv['parser_status'],
              requests=r['requests'],seconds=adv['finished']-adv['started']))
        N=len(d['cases']);done=len(items)
        rows.append(dict(method=p.name,model=model,N=N,completed_pairs=done,hits=sum(x['target_hit'] for x in items),
          ASR=sum(x['target_hit'] for x in items)/N if done==N else None,
          directions=[dict(transition=t,hits=sum(x['target_hit'] for x in items if x['transition']==t),N=N//5,observed=sum(x['transition']==t for x in items)) for t in C.TIDS],images=items))
    expected={'commercial50':7,'commercial150':9,'ratio30':5}[study]
    result=dict(complete=len(rows)==expected and all(r['completed_pairs']==r['N'] for r in rows),methods=rows,
      expected_conditions=expected,effort='low',Astra=False,selection_feedback=study=='commercial50')
    C.atomic(root/'summary.json',result);return result

def run_pack(path,model,limit):
    """One legacy child; only cohort-size adapter changes, isolation/parser unchanged."""
    C.check_lock()
    import run_codex_cli_pilot as old
    count=len(C.read(path/'private/manifest.json')['cases']);original_add=old.argparse.ArgumentParser.add_argument
    def add(parser,*args,**kwargs):
        if '--images' in args:kwargs['choices']=[3,10,30,50,150]
        return original_add(parser,*args,**kwargs)
    old.argparse.ArgumentParser.add_argument=add
    original_check=old.checked_order
    def checked(base,N):
        if N not in [30,50]:return original_check(base,N)
        d=C.read(base/'private/manifest.json');order=C.read(base/'request_order.json');declared={r['request_id']:r for r in d['requests']}
        if len(d['cases'])!=N or len({r['image_id'] for r in d['cases']})!=N or len(order)!=2*N or len(set(order))!=2*N:raise RuntimeError('Changed packet count')
        if set(order)!={rid for r in d['cases'] for rid in r['requests'].values()}:raise RuntimeError('Packet request mismatch')
        for rid in order:
            for file,key in [('image.png','image_sha256'),('prompt.txt','prompt_sha256')]:
                if C.sha(base/'blinded'/rid/file)!=declared[rid][key]:raise RuntimeError('Packet checksum mismatch')
        return order
    old.checked_order=checked
    study=path.parents[2];sys.argv=[sys.argv[0],'--images',str(count),'--model',model,'--pack-dir',str(path),
      '--result-dir',str(study/'responses'/model),'--limit',str(limit)]
    from paper_closeout.cli_recovery import install_timeout_capture
    install_timeout_capture()
    old.main()

def driver(deadline):
    import signal
    stop=False
    def handler(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    verify_models();(OUT/'commercial_driver.lock').parent.mkdir(parents=True,exist_ok=True)
    with (OUT/'commercial_driver.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not C.deadline_reached(deadline,3600) and not stop:
            C.check_lock();pending=[]
            from paper_closeout.plan import prepare_ratios
            for c in prepare_ratios()['qwen']:
                p=pack('ratio30','pull_'+str(c['pull']),c,'gpt-6.1-sol')
                if p:pending.append(('ratio30',p,'gpt-6.1-sol'))
            conditions=C.read(OUT/'followup_conditions.json',{})
            if len(conditions)==3:
                specs=commercial_specs(conditions)
                for method,values in specs.items():
                    if 'qwen' not in values:continue
                    p=pack('commercial50',method,values['qwen'],'gpt-6.1-sol')
                    if p:pending.append(('commercial50',p,'gpt-6.1-sol'))
            lock=C.read(OUT/'final_method_lock.json')
            if lock:
                for model in MODELS:
                    for method,v in lock['families']['qwen'].items():
                        p=pack('commercial150',method,v['condition'],model)
                        if p:pending.append(('commercial150',p,model))
            for study,path,model in pending:
                if stop or C.deadline_reached(deadline,3600):break
                N=len(C.read(path/'private/manifest.json')['cases']);root=OUT/study;order=C.read(path/'request_order.json')
                if all((root/'responses'/model/rid/'result.json').exists() for rid in order):continue
                # Fair bounded chunks; no unverified account/model fallback on failure.
                done=sum((root/'responses'/model/rid/'result.json').exists() for rid in order)
                limit=min(2*N,(done//20+1)*20)
                key=C.digest([study,model,path.name])[:18];blocked=C.read(root/'blocked'/key)
                if blocked:
                    # Only reviewed timeout-only interruptions may be retried. Successful
                    # results are immutable and skipped by the existing request runner.
                    from paper_closeout.cli_recovery import recover_timeouts
                    try:
                        recovered=recover_timeouts(root,model,order)
                        if not recovered:raise RuntimeError('No timeout-only unscored attempt available for recovery')
                    except RuntimeError as exc:
                        C.atomic(root/'blocked'/key,{**blocked,'recovery_error':str(exc)});continue
                    stamp=str(time.time_ns());archive=root/'recovery_history'/stamp
                    C.atomic(archive,dict(study=study,model=model,method=path.name,previous=blocked))
                    (root/'blocked'/key).unlink()
                log=root/'logs'/(model+'_'+path.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as f:
                    child=subprocess.Popen([sys.executable,__file__,'--run-pack',str(path),'--model',model,'--limit',str(limit)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                C.atomic(OUT/'commercial_active.json',dict(identity=C.await_identity(child.pid,__file__),study=study,method=path.name,model=model))
                while child.poll() is None:
                    if stop or C.deadline_reached(deadline,3600):child.terminate();child.wait(timeout=30);break
                    C.atomic(OUT/'commercial_heartbeat.json',dict(at=time.time(),model=model,method=path.name,limit=limit));time.sleep(3)
                if child.returncode:
                    C.atomic(root/'blocked'/key,dict(log=str(log),reason='Inspect retained incomplete trace; no duplicate scored request',exit=child.returncode))
                    # A transport/quota/auth failure must not cause model/account rotation.
                    text=log.read_text(errors='replace').lower()
                    if any(x in text for x in ['429','rate limit','quota','logged in','authentication']):
                        C.atomic(OUT/'commercial_driver_result.json',dict(status='BLOCKED_QUOTA_OR_AUTH',log=str(log)));return 2
                summarize(study)
            a=summarize('commercial50');b=summarize('commercial150');ratios=summarize('ratio30')
            if a['complete'] and b['complete'] and ratios['complete']:
                C.atomic(OUT/'commercial_driver_result.json',dict(status='COMPLETE'));return 0
            C.atomic(OUT/'commercial_status.json',dict(at=time.time(),stage50_complete=a['complete'],stage150_complete=b['complete']))
            time.sleep(10)
    C.atomic(OUT/'commercial_driver_result.json',dict(status='PARTIAL_DEADLINE_OR_STOP'));return 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float);p.add_argument('--run-pack',type=Path);p.add_argument('--model');p.add_argument('--limit',type=int);a=p.parse_args()
    if a.run_pack:run_pack(a.run_pack,a.model,a.limit)
    else:sys.exit(driver(a.deadline))
