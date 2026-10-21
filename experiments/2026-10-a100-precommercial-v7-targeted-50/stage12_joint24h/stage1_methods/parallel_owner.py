"""Adapters for ONE GPU owner and a separate CPU-only commercial-call child."""
import os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C
from stage1_methods import planner as P
HERE=Path(__file__).resolve().parent

def verify_extension():
    lock=C.read(P.OUT/'activation.json')
    if lock:
        for name,h in lock.get('extension_sources',{}).items():
            if C.sha(HERE/name)!=h:raise RuntimeError('Activated extension source changed: '+name)

def research_pending(phase):
    if phase not in ['INTEGRATED_EXPLORE30','INTEGRATED_CONFIRM50','STAGE1_COMPLETE'] and not (C.ROOT/'method_lock.json').exists():
        return [],False
    if C.read(P.OUT/'blocked.json'):return [],False
    try:
        verify_extension()
        import combo_selection as J
        from stage1_methods.fixed_four import plan as fixed_four_plan
        four_conditions,four_ready=fixed_four_plan()
        from stage1_methods.balanced_layers import plan as three_proxy_plan
        from stage1_methods.selected_confirmation import plan as selected_plan,report as selected_report
        from stage1_methods.fda_confirmation import plan as fda_confirmation_plan
        g=J.source_geometry()
        balanced_conditions,balanced_ready=three_proxy_plan(g)
        status,conditions=P.plan_next(g)
        selected_conditions,selected_ready=selected_plan()
        fda_conditions,fda_ready=fda_confirmation_plan()
        from stage1_methods.fda_matched import plan as matched_plan
        matched_conditions,matched_ready=matched_plan()
        selected_report()
        P.report()
        return list({c['id']:c for c in selected_conditions+fda_conditions+matched_conditions+balanced_conditions+four_conditions+conditions}.values()),status=='READY' and four_ready and balanced_ready and selected_ready and fda_ready and matched_ready
    except Exception as e:
        # Supplemental research failure cannot stop frozen primary confirmation/CLI.
        import traceback
        C.atomic(P.OUT/'blocked.json',dict(status='BLOCKED_REQUIRES_INTERVENTION',error=str(e),trace=traceback.format_exc(),primary_unaffected=True))
        return [],False

def stage2_progress(deadline):
    """No GPU admission here. Invoke the existing tested CLI path exactly once."""
    verify_extension()
    path=C.ROOT/'stage2/parallel_driver.json';old=C.read(path)
    if old:
        if C.same_process(old['identity']):return 'RUNNING'
        done=C.read(C.ROOT/'stage2/parallel_driver_result.json')
        if done:return done['status']
        # Ordinary controller restart adopts child; unexpected driver death needs inspection,
        # never blindly retry commercial calls with ambiguous scored requests.
        C.atomic(C.ROOT/'stage2/blocker.json',dict(status='BLOCKED_DRIVER_EXIT',driver=old))
        return 'BLOCKED'
    if not (C.ROOT/'method_lock.json').exists():raise RuntimeError('Commercial confirmation requires frozen method')
    log=C.ROOT/'logs/stage2_parallel_driver.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(HERE/'stage2_driver.py'),'--deadline',str(deadline)],
            stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},start_new_session=True)
    C.atomic(path,dict(identity=C.await_identity(proc.pid,str(HERE/'stage2_driver.py')),started=time.time(),CPU_only=True))
    return 'RUNNING'

def stop_stage2():
    old=C.read(C.ROOT/'stage2/parallel_driver.json')
    if old and C.same_process(old['identity']):os.kill(old['identity']['pid'],signal.SIGTERM)
    extra=C.read(C.ROOT/'fda_selected_commercial/driver.json')
    if extra and C.same_process(extra['identity']):os.kill(extra['identity']['pid'],signal.SIGTERM)

def fda_cli_progress(deadline):
    verify_extension()
    from stage1_methods.fda_commercial import OUT,prepare
    prepare();old=C.read(OUT/'driver.json')
    if old:
        if C.same_process(old['identity']):return 'RUNNING'
        result=C.read(OUT/'driver_result.json')
        return result['status'] if result else 'BLOCKED_DRIVER_EXIT'
    log=OUT/'driver.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        p=subprocess.Popen([sys.executable,str(HERE/'fda_commercial.py'),'--deadline',str(deadline)],stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''},start_new_session=True)
    C.atomic(OUT/'driver.json',dict(identity=C.await_identity(p.pid,str(HERE/'fda_commercial.py')),CPU_only=True,started=time.time()))
    return 'RUNNING'
