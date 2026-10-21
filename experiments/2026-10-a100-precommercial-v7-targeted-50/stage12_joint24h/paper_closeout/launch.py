"""Single-command activation with tested, graceful single-owner handoff."""
import argparse,datetime,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as C
from paper_closeout.plan import OUT

def amendments():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=C.ROOT/'study_lock_history'/stamp;archive.mkdir(parents=True,exist_ok=True)
    for rel in ['study_lock.json','stage1_methods/source_lock.json','stage1_methods/activation.json','execution_limits.json']:
        p=C.ROOT/rel;(archive/rel.replace('/','__')).write_bytes(p.read_bytes())
    original=C.sha(C.ROOT/'study_lock.json');lock=C.read(C.ROOT/'study_lock.json');lock['sources']=C.source_hashes()
    lock['manifests']['final150.json']=C.sha(C.ROOT/'manifests/final150.json')
    for r in C.cohort('final150')['images']:lock['input_hashes'][r['path']]=r['image_sha256']
    C.atomic(C.ROOT/'study_lock.json',lock)
    here=C.HERE/'stage1_methods';sources={p.name:C.sha(p) for p in here.glob('*.py')}
    p=C.ROOT/'stage1_methods/source_lock.json';s=C.read(p);s.update(sources=sources,base_study=C.sha(C.ROOT/'study_lock.json'));C.atomic(p,s)
    p=C.ROOT/'stage1_methods/activation.json';s=C.read(p);s.update(extension_sources=sources,new_study_sha256=C.sha(C.ROOT/'study_lock.json'));C.atomic(p,s)
    p=C.ROOT/'execution_limits.json';s=C.read(p);s.update(previous=s['max_attack_conditions'],max_attack_conditions=420,reason='User final closeout: selected4/FDA3/4 confirmations, GPT50 selection, final150, fixed Uniform4 five-ratio controls');C.atomic(p,s)
    C.atomic(C.ROOT/'lock_amendments'/(stamp+'_paper_closeout.json'),dict(old_study=original,new_study=C.sha(C.ROOT/'study_lock.json'),archive=str(archive),
      reason='Explicit user final scope; original window/locks preserved, new deadline separately recorded',
      GPT50_selection_feedback=True,final150_no_retuning=True,old_results_untouched=True,workers_max=8))
    C.check_lock()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','status','report']);p.add_argument('--deadline-hours',type=float,default=24);a=p.parse_args()
    if a.mode=='status':print(C.json.dumps(C.read(OUT/'status.json',{'state':'NOT_ACTIVATED'}),indent=2));return
    if a.mode=='report':
        from paper_closeout.analysis import calculate,publish
        calculate();publish();return
    if not 6<a.deadline_hours<=48:raise ValueError('Closeout deadline must be6..48h')
    from paper_closeout.data import prepare
    from paper_closeout.commercial import verify_models
    prepare();verify_models()
    verification=C.read(OUT/'tests.json')
    if not verification or verification.get('status')!='PASS':raise RuntimeError('Run closeout tests first')
    existing=C.read(OUT/'activation.json')
    if existing:
        C.check_lock();print('Already activated; deadline unchanged',C.read(C.ROOT/'closeout_window.json'))
        subprocess.run(['bash',str(C.EXP/'run_stage12_joint24h.sh'),'--resume','--deadline-hours',str(a.deadline_hours)],check=True)
        return
    controller=C.read(C.ROOT/'controller.json',{}).get('identity')
    if controller and C.same_process(controller):os.kill(controller['pid'],signal.SIGSTOP)
    amendments()
    now=time.time();C.freeze(C.ROOT/'closeout_window.json',dict(start=now,deadline=now+a.deadline_hours*3600,hours=a.deadline_hours,
      original_window_sha256=C.sha(C.ROOT/'window.json'),user_authorized_new_scope=True))
    C.freeze(OUT/'activation.json',dict(started=now,deadline=C.effective_deadline(),models=['gpt-6.1-sol','gpt-6-sol','gpt-5.6-sol'],effort='low',
      scope=['selected4_FDA3_FDA4_30_50','matched_GPT50_selection','three_methods_final150','Uniform4_five_PP_ratios_three_large_plus_GPT61','analysis_and_publication'],
      all_historical_tables_rerun=False,full_other_ablations=False,GPT50_is_development=True,final150_holdout=True,old_stage2_untouched=True,
      persistent_output=str(OUT),scientific_sources=C.source_hashes()))
    if controller and C.same_process(controller):
        os.kill(controller['pid'],signal.SIGTERM);os.kill(controller['pid'],signal.SIGCONT)
    # Preserve atomic image checkpoints; never kill foreign or running model jobs.
    until=time.time()+600
    while time.time()<until:
        s=C.read(C.ROOT/'scheduler.json',{});owned=[j for j in s.get('active',[]) if C.same_process(j['process_identity'])]
        sup=C.read(C.ROOT/'supervisor.json',{}).get('identity');live_sup=sup and C.same_process(sup)
        if not owned and not live_sup:break
        time.sleep(3)
    else:raise RuntimeError('Graceful handoff waiting; inspect own leases before resume')
    (C.ROOT/'stop.request').unlink(missing_ok=True)
    subprocess.run(['bash',str(C.EXP/'run_stage12_joint24h.sh'),'--resume','--deadline-hours',str(a.deadline_hours)],check=True)
    print('Activated single owner closeout',OUT,'deadlineUTC',datetime.datetime.fromtimestamp(C.effective_deadline(),datetime.timezone.utc).isoformat())

if __name__=='__main__':main()
