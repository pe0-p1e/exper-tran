#!/usr/bin/env python3
"""Resumable orchestration of the frozen 13 x 5 recovery matrix."""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]
OUT=REPO/"outputs/a100_precommercial_v6_multi_repr_recovery"
PY=REPO/".venv-a100/bin/python"

def load_matrix():
    m=yaml.safe_load((ROOT/"experiment_matrix.yaml").read_text())["multiple_proxy"]
    seen=set(); rows=[]
    for block in ("block_1","block_2"):
        for row in m[block]:
            for proxies in row["ensembles"]:
                key=(row["target"],tuple(proxies))
                if key in seen: continue
                seen.add(key)
                eid=f"E{len(rows)+1:02d}_"+"_".join(proxies)
                rows.append({"ensemble_id":eid,"target":row["target"],"proxies":proxies,"block_source":block})
    if len(rows)!=13: raise RuntimeError(f"expected 13 unique ensembles, got {len(rows)}")
    return rows

def transitions():
    raw=yaml.safe_load((ROOT/"config/runner_template.yaml").read_text())
    return [str(t["id"]) for t in raw["transitions"] if int(t["source"])<int(t["target"])]

def run(args,log_path):
    env=os.environ.copy(); env["PYTHONPATH"]=str(REPO/"src")+os.pathsep+str(REPO/"experiments/2026-09-a100-final-v5/src")+os.pathsep+str(ROOT/"src")
    with log_path.open("a",encoding="utf-8") as f:
        f.write("\n$ "+" ".join(map(str,args))+"\n"); f.flush()
        proc=subprocess.Popen(list(map(str,args)),cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end="",flush=True); f.write(line); f.flush()
        return proc.wait()

def attack(batch_size):
    rows=load_matrix(); tids=transitions(); fails=[]; started=time.time()
    root=OUT/"multiple_proxy/attacks"; logs=OUT/"logs/attacks"; logs.mkdir(parents=True,exist_ok=True)
    for ix,e in enumerate(rows,1):
        # Keep the campaign driver's state lookup identical to the runner's
        # filesystem slug; model names such as "Gemma 4 E2B-it" contain spaces.
        slug=re.sub(r"[^A-Za-z0-9_.-]+", "_", e["ensemble_id"])
        print(f"[Multiple Proxy] ensemble {ix}/{len(rows)}: {e['ensemble_id']} -> {e['target']}",flush=True)
        log=logs/f"{slug}.log"
        code=run([PY,ROOT/"src/run_recovery_ensemble.py","--ensemble-id",e["ensemble_id"],"--target",e["target"],"--proxies",*e["proxies"],"--transitions",*tids,"--batch-size",batch_size,"--output-dir",OUT],log)
        # A single transition failure must not prevent any other transition from
        # being attempted. Resume valid cells; retry only missing/failed cells.
        for tid in tids:
            state=root/slug/f"{tid}.json"
            valid=False
            if state.is_file():
                try:
                    s=json.loads(state.read_text()); valid=s.get("status")=="attack_complete" and len(s.get("image_ids",[]))==30 and all((OUT/p).is_file() for p in s.get("adv_paths",[]))
                except Exception: pass
            if valid: continue
            print(f"[Multiple Proxy] retry cell {tid} for {e['ensemble_id']} (prior status={code})",flush=True)
            rc=run([PY,ROOT/"src/run_recovery_ensemble.py","--ensemble-id",e["ensemble_id"],"--target",e["target"],"--proxies",*e["proxies"],"--transitions",tid,"--batch-size",batch_size,"--output-dir",OUT],log)
            if rc and batch_size>1:
                print(f"[Multiple Proxy] retrying {e['ensemble_id']}/{tid} with microbatch=1; precision remains BF16",flush=True)
                rc=run([PY,ROOT/"src/run_recovery_ensemble.py","--ensemble-id",e["ensemble_id"],"--target",e["target"],"--proxies",*e["proxies"],"--transitions",tid,"--batch-size",1,"--output-dir",OUT],log)
            if rc:
                fails.append({"ensemble_id":e["ensemble_id"],"target":e["target"],"transition_id":tid,"phase":"attack","exit_code":rc})
    (OUT/"audits").mkdir(parents=True,exist_ok=True)
    (OUT/"audits/attack_failures.json").write_text(json.dumps(fails,indent=2)+"\n")
    print(f"[Multiple Proxy] attempted {len(rows)*len(tids)} cells; failures={len(fails)}; elapsed={time.time()-started:.1f}s",flush=True)

def evaluate():
    targets=sorted({e["target"] for e in load_matrix()}); logs=OUT/"logs/evaluation"; logs.mkdir(parents=True,exist_ok=True); fails=[]
    for i,target in enumerate(targets,1):
        print(f"[Target evaluation] {i}/{len(targets)} {target}",flush=True)
        rc=run([PY,ROOT/"src/evaluate_recovery_target.py","--target",target,"--output-dir",OUT,"--batch-size",1],logs/(target.replace("/","_")+".log"))
        if rc: fails.append({"target":target,"phase":"evaluation","exit_code":rc})
    (OUT/"audits/evaluation_failures.json").write_text(json.dumps(fails,indent=2)+"\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["attacks","evaluate"]); p.add_argument("--batch-size",type=int,default=2); a=p.parse_args()
    if a.phase=="attacks": attack(a.batch_size)
    else: evaluate()
if __name__=="__main__": main()
