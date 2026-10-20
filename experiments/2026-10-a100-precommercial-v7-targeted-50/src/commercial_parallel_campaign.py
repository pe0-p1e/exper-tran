"""Three provider lanes, one report owner; same frozen protocol and model requests."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from commercial_matrix_pack import MATRIX,prepare_pack,freeze
from run_commercial_matrix import entries,aggregate
from proxy_study_protocol import EXP,ROOT,atomic_json,sha


def main():
    MATRIX.mkdir(parents=True,exist_ok=True)
    with (MATRIX/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((EXP/'commercial_model_matrix.json').read_text());models=entries(config)
        packs={}
        for stage in ['dev10','test150']:
            base,order,secret,declared=prepare_pack(stage);packs[stage]=(secret,{r:{k:declared[r][k] for k in ['image_sha256','prompt_sha256']} for r in order})
            freeze(MATRIX/'stages'/stage/'cohort_lock.json',dict(manifest_sha256=sha(base/'private/manifest.json'),request_order=order))
        atomic_json(MATRIX/'parallel_orchestration_addendum.json',dict(at=time.time(),pid=os.getpid(),
            reason='User prioritizes Gemini/Claude; replace serial commercial coordinators with independent account lanes',
            changed_model_image_prompt_effort=False,model_source_lock_preserved=True,
            max_simultaneous_provider_lanes=3,gpu_controller_unchanged=True,
            sources={n:sha(EXP/'src'/n) for n in ['commercial_provider_lane.py','commercial_parallel_campaign.py']}))
        jobs=[]
        for provider in ['codex','gemini_agy','claude']:
            stream=(MATRIX/(provider+'.log')).open('a')
            proc=subprocess.Popen([sys.executable,str(EXP/'src/commercial_provider_lane.py'),'--provider',provider],
                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((provider,proc,stream))
        atomic_json(ROOT/'unified_scheduler/commercial_handoff.json',dict(pid=os.getpid(),providers={p:q.pid for p,q,s in jobs},
                local_gpu_workers_added=0,reason='Three independent commercial provider lanes; existing GPU owner retained'))
        while True:
            lane_states={}
            for provider,proc,stream in jobs:
                p=MATRIX/'lanes'/(provider+'.json')
                if p.exists():lane_states[provider]=json.loads(p.read_text())
            tables={}
            for stage,(secret,hashes) in packs.items():
                statuses={}
                for spec in models:
                    statuses[spec['id']]=lane_states.get(spec['provider'],{}).get('models',{}).get(spec['id']+':'+stage,dict(status='QUEUED'))
                tables[stage]=aggregate(stage,secret,models,statuses,hashes)
            if any(d.get('ten_finished') for d in lane_states.values()):
                eligible=[r['matrix_id'] for r in tables['dev10'] if r['status']=='COMPLETE']
                atomic_json(MATRIX/'ten_to_full_gate.json',dict(at=time.time(),eligible_models=eligible,
                    criterion='20 image-verified requests and10 complete pairs; no ASR threshold; per-provider stage progression',
                    provider_ten_finished=[p for p,d in lane_states.items() if d.get('ten_finished')]))
            if len(lane_states)==3 and all(d.get('ten_finished') for d in lane_states.values()):
                marker=MATRIX/'ten_image_stage_finished.json'
                if not marker.exists():atomic_json(marker,dict(at=time.time(),models=tables['dev10']))
            if all(proc.poll() is not None for p,proc,s in jobs):
                for p,q,s in jobs:s.close()
                atomic_json(MATRIX/'campaign_completion.json',dict(at=time.time(),providers={p:q.returncode for p,q,s in jobs},
                    all_requested_completed=all(r['status']=='COMPLETE' for r in tables['test150']),models=tables['test150']))
                break
            time.sleep(12)


if __name__=='__main__':main()
