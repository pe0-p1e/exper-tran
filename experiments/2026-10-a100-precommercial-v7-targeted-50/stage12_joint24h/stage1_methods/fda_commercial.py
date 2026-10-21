"""Paired selected/FDA/Uniform50 through the existing isolated subscription CLI."""
import argparse,os,shutil,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C
from stage1_methods import fda_confirmation as F
OUT=C.ROOT/'fda_selected_commercial'

def prepare():
    old=C.read(OUT/'method_lock.json')
    if old:return old
    fda=F.prepare();selected=C.read(C.ROOT/'selected_followup/method_lock.json');fam='qwen'
    methods=dict(uniform=selected['families'][fam]['uniform'],selected=selected['families'][fam]['selected'],FDA=fda['families'][fam]['main50'])
    lock=dict(methods=methods,cohort_sha256=C.sha(C.ROOT/'manifests/confirm.json'),
      FDA_lock_sha256=C.sha(F.OUT/'method_lock.json'),selected_lock_sha256=C.sha(C.ROOT/'selected_followup/method_lock.json'),
      model='gpt-6.1-sol',reasoning_effort='low',route='existing subscription Codex CLI',
      family_rule='predeclared qwen-style legal proxy configurations, not chosen from GPT outcomes',
      no_target_retuning=True,baseline_already_observed=True,N=50,
      disclosure='Follow-up paired confirmation on existing50 images; Uniform GPT already observed, not untouched commercial testing')
    C.freeze(OUT/'method_lock.json',lock);return lock

def cache_matches(result,request):
    i=result.get('identity',{})
    return i.get('model')=='gpt-6.1-sol' and all(i.get(k)==request[k] for k in ['image_sha256','prompt_sha256']) and 'parsed_label' in result and result.get('request_id')==request['request_id']

def prepare_packs():
    lock=prepare()
    if lock['cohort_sha256']!=C.sha(C.ROOT/'manifests/confirm.json'):raise RuntimeError('Commercial cohort changed')
    if not all(C.attack_done(c) for c in lock['methods'].values()):raise RuntimeError('Wait for all three frozen50-image attack methods')
    from stage1_methods.early_confirm import ordered_cases
    cases=ordered_cases(C.cohort('confirm')['images']);source=next((C.ROOT/'stage2/packs/uniform/blinded').glob('*/prompt.txt'));prompt=source.read_text();packs=[]
    oldroot=C.ROOT/'stage2/responses';reused=[]
    provenance=[C.read(p,{}) for p in oldroot.glob('*.json')]
    reuse_allowed=any(d.get('requested_model')=='gpt-6.1-sol' and d.get('reasoning_effort')=='low' for d in provenance)
    for method,c in lock['methods'].items():
        pack=OUT/'packs'/method;private=[];requests=[];order=[]
        for case in cases:
            receipt=C.read(C.image_receipt(c,case['image_id']));ids={}
            for kind,key in [('clean','clean'),('adversarial','adv')]:
                image=Path(receipt[key]);rid='r_'+C.digest(['V8',kind,C.sha(image),C.digest(prompt),'gpt-6.1-sol','low'])[:24];ids[kind]=rid
                dest=pack/'blinded'/rid;dest.mkdir(parents=True,exist_ok=True)
                if not (dest/'image.png').exists():shutil.copyfile(image,dest/'image.png')
                (dest/'prompt.txt').write_text(prompt)
                req=dict(request_id=rid,image_sha256=C.sha(image),prompt_sha256=C.sha(dest/'prompt.txt'));requests.append(req);order.append(rid)
                cached=C.read(oldroot/rid/'result.json')
                if cached and reuse_allowed and cache_matches(cached,req):
                    target=OUT/'stage2/responses'/rid;target.mkdir(parents=True,exist_ok=True)
                    for name in ['result.json','stdout.jsonl','stderr.txt','answer.txt']:
                        src=oldroot/rid/name
                        if src.exists() and not (target/name).exists():shutil.copyfile(src,target/name)
                    reused.append(dict(request_id=rid,source_result_sha256=C.sha(oldroot/rid/'result.json')))
            private.append(dict(image_id=case['image_id'],transition=case['transition'],source_label=case['source'],target_label=case['target'],requests=ids))
        C.freeze(pack/'private/manifest.json',dict(cases=private,requests=requests));C.freeze(pack/'request_order.json',order);packs.append(pack)
    unique={rid for p in packs for rid in C.read(p/'request_order.json')}
    C.freeze(OUT/'pack_lock.json',dict(method_lock_sha256=C.sha(OUT/'method_lock.json'),packs={p.name:C.sha(p/'private/manifest.json') for p in packs},
      unique_planned_requests=len(unique),reused=sorted({r['request_id'] for r in reused}),target_feedback_for_selection=False))
    return packs

def summarize():
    methods=[]
    for method in prepare()['methods']:
        manifest=C.read(OUT/'packs'/method/'private/manifest.json',{});pairs=[]
        for case in manifest.get('cases',[]):
            clean=C.read(OUT/'stage2/responses'/case['requests']['clean']/'result.json');adv=C.read(OUT/'stage2/responses'/case['requests']['adversarial']/'result.json')
            if clean is None or adv is None:continue
            pairs.append(dict(image_id=case['image_id'],transition=case['transition'],target_hit=adv['parsed_label']==case['target_label'],clean_correct=clean['parsed_label']==case['source_label'],parser_status=adv['parser_status']))
        methods.append(dict(method=method,N=50,completed_pairs=len(pairs),hits=sum(x['target_hit'] for x in pairs),ASR=sum(x['target_hit'] for x in pairs)/50 if len(pairs)==50 else None,
          directions=[dict(transition=t,hits=sum(x['target_hit'] for x in pairs if x['transition']==t),N=10,completed=sum(x['transition']==t for x in pairs)) for t in C.TIDS],images=pairs))
    result=dict(complete=all(m['completed_pairs']==50 for m in methods),methods=methods,model='gpt-6.1-sol',effort='low',
      no_retuning=True,requests_reused=C.read(OUT/'pack_lock.json',{}).get('reused',[]))
    C.atomic(OUT/'summary.json',result);return result

def run_pack(pack):
    C.check_lock()
    import stage2_cli as existing
    # Child-local adapter only: preserve original Stage2 locks, responses and summaries.
    existing.ROOT=OUT
    existing.run_legacy(Path(pack),100)

def driver(deadline):
    C.check_lock();lock=prepare();stop=False
    def handler(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    while not all(C.attack_done(c) for c in lock['methods'].values()):
        C.atomic(OUT/'status.json',dict(phase='WAITING_FROZEN_THREE_METHODS_50',at=time.time()))
        if stop or time.time()>=deadline-3600:return 2
        time.sleep(15)
    packs=prepare_packs();C.atomic(OUT/'status.json',dict(phase='EVALUATING',at=time.time()))
    for pack in packs:
        if stop or time.time()>=deadline-3600:break
        log=OUT/'logs'/(pack.name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run-pack',str(pack)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        C.atomic(OUT/'active_request_worker.json',dict(identity=C.await_identity(p.pid,str(Path(__file__).resolve())),method=pack.name))
        while p.poll() is None:
            if stop or time.time()>=deadline-3600:p.terminate();p.wait(timeout=30);break
            time.sleep(3)
        summarize()
        if p.returncode:
            C.atomic(OUT/'driver_result.json',dict(status='BLOCKED_OR_PARTIAL',method=pack.name,log=str(log),at=time.time()));return 2
    result=summarize();C.atomic(OUT/'driver_result.json',dict(status='COMPLETE' if result['complete'] else 'PARTIAL',at=time.time()))
    return 0 if result['complete'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-pack');p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.run_pack:run_pack(a.run_pack)
    else:sys.exit(driver(a.deadline))
