"""Isolated BF16 PP4 worker; immutable original runner and shared cache locks."""
import argparse
import fcntl
import json
import sys
from pathlib import Path
import proxy_study_worker as shared
from pp4_protocol import ROOT,OUT,EXP,VALIDATORS,plan,condition,validation_path,development_ids,sha,atomic_json


def install_audit():
    import vision_features,strict_model_audit
    def audit(model,**kwargs):
        path=ROOT/'audits/writer.lock';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            kwargs['output_dir']=ROOT
            return shared.original_audit(model,**kwargs)
    vision_features.audit_loaded_model=audit;strict_model_audit.audit_loaded_model=audit


def load_condition(cid):
    p=plan();candidates=[r['condition'] for r in p['ladder']+p['weights']]+[p['smoke']]
    return next(c for c in candidates if c['id']==cid)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['attack','validate'])
    parser.add_argument('--condition',required=True);parser.add_argument('--model');args=parser.parse_args()
    lock=json.loads((ROOT/'protocol_lock.json').read_text())
    for name,h in lock['source_hashes'].items():
        if sha(EXP/'src'/name)!=h:raise RuntimeError('Locked scientific source changed: '+name)
    install_audit();c=load_condition(args.condition)
    if args.mode=='attack':
        import run_v7_attack as runner
        from pp4_objective import objective
        if c['options']['loss']=='pull_push':runner.objective=objective
        runner.METHODS['PP4']=dict(c['options']);runner.prepare_data=shared.verify_existing_data
        if c['split']=='dev':
            original=runner.records
            def selected(path):
                rows=original(path)
                return rows[2:4] if Path(path).parent==OUT/'manifests/dev' else rows
            runner.records=selected
        sys.argv=[sys.argv[0],'--models',*c['models'],'--method','PP4','--fusion','equal',
                  '--split',c['split'],'--steps',str(c['steps']),'--seed','42','--batch-size','2',
                  '--restarts','1','--run-label',c['id'],'--config-json',str(ROOT/'configs'/(c['id']+'.json'))]
        if c['di']:sys.argv+=['--input-diversity']
        if c['steps']!=50:sys.argv+=['--transitions','T01']
        if c['split']=='dev':sys.argv+=['--limit-per-direction','2']
        runner.main()
    elif args.mode=='validate':
        if args.model not in VALIDATORS or args.model in c['models']:raise RuntimeError('Validator must be held out')
        import evaluate_surrogate_outputs as runner
        original=runner.records
        def selected(path):
            rows=original(path)
            return rows[2:4] if Path(path).parent==OUT/'manifests/dev' else rows
        runner.records=selected
        sys.argv=[sys.argv[0],'--root',str(OUT/'attacks/dev'/c['name']),'--models',args.model,'--limit-per-direction','2']
        runner.main()
        from proxy_study_protocol import fingerprint
        src=OUT/'focused_optimization/native_validation'/fingerprint(development_ids())[:12]/c['name']/(args.model.replace('/','_')+'.json')
        d=json.loads(src.read_text());d['source']=str(src);d['source_sha256']=sha(src)
        atomic_json(validation_path(c,args.model),d)


if __name__=='__main__':main()
