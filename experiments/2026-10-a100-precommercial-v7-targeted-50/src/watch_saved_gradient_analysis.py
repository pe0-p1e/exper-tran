"""Refresh saved gradient diagnostics on CPU; never load a model or admit GPU work."""
import fcntl
import os
import subprocess
import sys
import time
from proxy_study_protocol import ROOT, OUT, EXP, atomic_json


def main():
    folder=ROOT/'mechanism_gradients';folder.mkdir(exist_ok=True)
    with (folder/'refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            files=list((OUT/'attacks/test').glob('*/T*/batch_*/state.json'))
            signature=tuple(sorted((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files))
            if signature!=previous:
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2',
                     'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
                began=time.time()
                run=subprocess.run([sys.executable,str(EXP/'src/analyze_saved_gradients.py')],env=env)
                atomic_json(folder/'refresh_status.json',dict(at=time.time(),runtime=time.time()-began,
                    returncode=run.returncode,model_inference_calls=0,gpu_jobs_added=0,
                    observed_state_files=len(files)))
                if run.returncode==0:previous=signature
            if (ROOT/'unified_scheduler/completion.json').exists():break
            time.sleep(60)


if __name__=='__main__':main()
