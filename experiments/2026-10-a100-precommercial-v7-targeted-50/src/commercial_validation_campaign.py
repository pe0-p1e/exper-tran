"""Ten-image report, then150-image evaluation gated only on interface completeness."""
import json
import subprocess
import sys
import time
from proxy_study_protocol import EXP,atomic_json
from commercial_matrix_pack import MATRIX


def main():
    runner=EXP/'src/run_commercial_matrix.py';config=EXP/'commercial_model_matrix.json'
    smoke=subprocess.run([sys.executable,str(runner),'--stage','dev10','--config',str(config)])
    report=MATRIX/'stages/dev10/reports/summary.json'
    if not report.exists():raise RuntimeError('Ten-image report missing; no large run is permitted')
    result=json.loads(report.read_text());rows=result['models']
    eligible={r['matrix_id'] for r in rows if r['status']=='COMPLETE' and r['completed_requests']==20 and r['observed_pairs']==10}
    # Neither hits nor ASR is used in this gate. Keep every valid negative result.
    cfg=json.loads(config.read_text())
    for row in cfg['models']:row['enabled']=row['id'] in eligible
    fullconfig=MATRIX/'full_interface_verified_models.json';atomic_json(fullconfig,cfg)
    atomic_json(MATRIX/'ten_to_full_gate.json',dict(at=time.time(),ten_image_returncode=smoke.returncode,
          ten_image_report=str(report),eligible_models=sorted(eligible),criterion='20 verified requests and10 complete pairs; NO ASR threshold',
          blocked=[r for r in rows if r['matrix_id'] not in eligible],target_feedback_for_method_selection=False))
    print('[commercial ten-image report]',MATRIX/'stages/dev10/reports/report.md',flush=True)
    for r in rows:print('[commercial10]',r['model'],r['status'],r['hits'],'/',r['denominator'],flush=True)
    if not eligible:
        print('No image-capable complete interface; large evaluation blocked.',flush=True);return 2
    print('[commercial full] starting frozen150-image evaluation for',len(eligible),'interface-verified models',flush=True)
    full=subprocess.run([sys.executable,str(runner),'--stage','test150','--config',str(fullconfig)])
    atomic_json(MATRIX/'campaign_completion.json',dict(at=time.time(),ten_returncode=smoke.returncode,
          full_returncode=full.returncode,requested_models=len(rows),interface_verified_models=len(eligible),
          all_requested_completed=(smoke.returncode==0 and full.returncode==0),remaining_access_or_interface_failures_retained=True))
    return 0 if smoke.returncode==0 and full.returncode==0 else 2


if __name__=='__main__':raise SystemExit(main())
