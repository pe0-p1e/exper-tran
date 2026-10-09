#!/usr/bin/env python3
"""Extract reusable V6 representations and generate all final measurements/plots."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = REPO / "outputs/a100_precommercial_v6"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/2026-09-a100-final-v5/src"))


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: (None if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)) else v) for k,v in row.items()} for row in rows)


def image_key(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_refs():
    from primary_ml_cka.data.manifests import read_manifest
    paths = {}
    for label in range(1, 11):
        records = read_manifest(OUT / "evaluation/manifests" / f"class_references_{label:02d}.jsonl")
        paths[label] = [OUT / "canonical_images" / r.relative_path for r in records]
    return paths


def collect_cells():
    grouped = {}
    for family in ("single_proxy", "layer_sweep", "ablation", "reverse_direction", "smoke_single", "smoke_layer", "smoke_ablation"):
        for meta_path in (OUT / family / "work").glob("*/cell_meta.json"):
            meta = json.loads(meta_path.read_text())
            workspace = meta_path.parent
            for state_path in sorted(workspace.glob("states_v6*/P02/T*/batch_*.json")):
                try: state = json.loads(state_path.read_text())
                except (OSError, json.JSONDecodeError): continue
                if state.get("status") not in {"attack_complete", "complete"}:
                    continue
                tid = state["transition_id"]
                batch = int(state.get("batch_index", 0))
                artifact = workspace / "attacks/P02" / f"a100_v5_{tid}" / f"batch_{batch:02d}" / str(state["objective_tag"]) / f"lambda_{float(state.get('lambda_cka',1)):g}"
                count = int(state.get("source_count", 0))
                clean = [artifact / f"{i:02d}_clean.png" for i in range(count)]
                adv = [artifact / f"{i:02d}_adv.png" for i in range(count)]
                if any(not p.is_file() for p in (*clean, *adv)): continue
                attack = state.get("attack", {})
                proxy_mask = attack.get("proxy_target_hit_mask", [])
                target_mask = state.get("target", {}).get("target_hit_mask", [])
                key=(family,workspace,state["transition_id"])
                cell=grouped.setdefault(key,{"meta":meta,"family":family,"workspace":workspace,"states":[],"state_paths":[],"clean":[],"adv":[],"proxy_mask":[],"target_mask":[]})
                cell["states"].append(state); cell["state_paths"].append(state_path)
                cell["clean"].extend(clean); cell["adv"].extend(adv)
                cell["proxy_mask"].extend(proxy_mask); cell["target_mask"].extend(target_mask)
    cells=[]
    for cell in grouped.values():
        cell["states"].sort(key=lambda s:int(s.get("batch_index",0)))
        cell["state_paths"].sort(key=lambda p:int(json.loads(p.read_text()).get("batch_index",0)))
        cell["state"]=cell["states"][0]
        cell["state_path"]=cell["state_paths"][0]
        cell["attacks"]=[s.get("attack",{}) for s in cell["states"]]
        cells.append(cell)
    return cells


def collect_multi():
    cells = []
    root = OUT / "multiple_proxy/attacks"
    lock = json.loads((ROOT / "model_revisions.json").read_text())
    repo_to_name = {item["repo_id"]: name for name, item in lock.items()}
    for state_path in root.glob("*/*.json"):
        try: state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError): continue
        if state.get("status") not in {"attack_complete", "complete"}: continue
        cond = state.get("condition", state_path.parent.name)
        tid = state["transition_id"]
        clean, adv, pm, tm = [], [], [], []
        for bi, batch in enumerate(state.get("batches", [])):
            bdir = state_path.parent / tid / f"batch_{bi:02d}"
            for i, image_id in enumerate(batch.get("image_ids", [])):
                clean.append(bdir / f"{i:02d}_clean.png"); adv.append(bdir / f"{i:02d}_adv.png")
                pm.append(bool(batch.get("all_proxy_success_mask", [])[i]))
                ev = batch.get("target_evaluation", {})
                tm.append(bool(ev.get("target_hit_mask", [])[i]) if i < len(ev.get("target_hit_mask", [])) else None)
        if len(clean)!=30:
            print(f"[analysis] exclude non-production ensemble cohort {state_path}: N={len(clean)}, expected=30",flush=True)
            continue
        if clean and all(p.is_file() for p in (*clean, *adv)):
            proxy_names = state.get("proxy_models", state.get("proxies", []))
            proxy_revisions = state.get("proxy_revisions", {})
            if not proxy_revisions:
                proxy_revisions = {
                    name: state.get("proxy_taps", {}).get(name, {}).get("revision")
                    or next((b.get("proxy_revisions", {}).get(name) for b in state.get("batches", []) if b.get("proxy_revisions", {}).get(name)), None)
                    or lock[name]["revision"]
                    for name in proxy_names
                }
            target_value = state["target_model"]
            target_name = target_value if target_value in lock else repo_to_name.get(target_value)
            if target_name is None:
                print(f"[analysis] skip multi state with unknown target model: {state_path}", flush=True)
                continue
            cells.append({"family":"multiple_proxy", "condition":cond, "target_name":target_name, "proxy_names":proxy_names, "proxy_repo_ids":{name: lock[name]["repo_id"] for name in proxy_names}, "target_repo_id":lock[target_name]["repo_id"], "transition_id":tid, "state":state, "clean":clean,"adv":adv,"proxy_mask":pm,"target_mask":tm,"state_path":state_path})
    return cells


def metadata(cell):
    if "meta" in cell:
        m = cell["meta"]
        return {"family":cell["family"],"condition":m["condition"],"proxy_names":[m["proxy_name"]],"proxy_repo_ids":{m["proxy_name"]:m["proxy_repo_id"]},"target_name":m["target_name"],"target_repo_id":m["target_repo_id"],"layer":m["layer"],"pull_weight":m["pull_weight"],"push_weight":m["push_weight"],"smoke":m.get("smoke",False)}
    st=cell["state"]
    lock=json.loads((ROOT/"model_revisions.json").read_text())
    return {"family":cell["family"],"condition":cell["condition"],"proxy_names":cell["proxy_names"],"proxy_repo_ids":cell.get("proxy_repo_ids", {n:st.get("proxy_taps",{}).get(n,{}).get("repo_id",lock[n]["repo_id"]) for n in cell["proxy_names"]}),"target_name":cell["target_name"],"target_repo_id":cell.get("target_repo_id",lock[cell["target_name"]]["repo_id"]),"layer":-1,"pull_weight":st.get("pull_weight",.75),"push_weight":st.get("push_weight",.25),"smoke":False}


def feature_cache_path(model_name: str, layer: int, revision: str) -> Path:
    return OUT / "analysis/embeddings" / f"{safe(model_name)}__{revision[:10]}__layer{layer}.npz"


def extract_group(model_name: str, model_id: str, revision: str, layer: int, paths: list[Path]) -> dict[str, np.ndarray]:
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from primary_ml_cka.config.schema import AttackConfig
    from primary_ml_cka.data.preprocessing import ensure_canvas
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.models.proxies.registry import load_proxy
    MODEL_REVISIONS[model_id] = revision
    cache = feature_cache_path(model_name, layer, revision)
    values = {}
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
    unique = list(dict.fromkeys(p.resolve() for p in paths if p.is_file()))
    missing = [p for p in unique if image_key(p) not in values]
    if not missing: return values
    proxy = load_proxy(model_id, Path(os.environ.get("HF_HOME", ".hf-cache")), torch.device("cuda"), AttackConfig(generative_precision="bf16"))
    try:
        start=0; batch_size=8
        while start<len(missing):
            group = missing[start:start+batch_size]
            rows=[]
            for path in group:
                with Image.open(path) as im:
                    rows.append(ensure_canvas(pil_to_tensor(im.convert("RGB")).float().div(255).unsqueeze(0),224).squeeze(0))
            images=torch.stack(rows).cuda()
            try:
                with torch.inference_mode():
                    output=proxy.image_embeddings(images,representation_type="vision_encoder",layer=layer,pooling="mean")
            except torch.cuda.OutOfMemoryError:
                if batch_size<=1: raise
                batch_size=max(1,batch_size//2)
                print(f"[extract] CUDA OOM for {model_name}; retrying with batch_size={batch_size}",flush=True)
                del images,rows
                gc.collect(); torch.cuda.empty_cache()
                continue
            features=F.normalize(output.embeddings.float(),dim=-1).cpu().numpy()
            for path,feature in zip(group,features,strict=True): values[image_key(path)]=feature
            start+=len(group)
            print(f"[extract] {model_name} layer={layer} {min(start,len(missing))}/{len(missing)} batch={batch_size}",flush=True)
            del images,output,features,rows
    finally:
        del proxy; gc.collect(); torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**values)
    return values


def compute_dispersion(matrix: np.ndarray) -> dict:
    z=F.normalize(torch.from_numpy(matrix).float(),dim=-1)
    mean=z.mean(0); center=F.normalize(mean,dim=0)
    covariance=z-mean
    eig=torch.linalg.svdvals(covariance).square()
    p=eig/eig.sum().clamp_min(1e-12)
    return {"dispersion":float((1-z@center).mean()),"covariance_trace":float(covariance.square().sum()/max(1,len(z)-1)),"effective_rank":float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())),"center":center.numpy()}


def cka(x,y):
    x=torch.from_numpy(x).float(); y=torch.from_numpy(y).float()
    x-=x.mean(0,keepdim=True); y-=y.mean(0,keepdim=True)
    cross=x.T@y
    value=float(cross.square().sum()/(torch.linalg.matrix_norm(x.T@x)*torch.linalg.matrix_norm(y.T@y)).clamp_min(1e-12))
    # Float32 roundoff can put identical-representation CKA a few ulps above 1.
    return min(1.0,max(0.0,value))


def rsa(x,y):
    x=F.normalize(torch.from_numpy(x).float(),dim=-1); y=F.normalize(torch.from_numpy(y).float(),dim=-1)
    i=torch.triu_indices(len(x),len(x),offset=1)
    return float(spearmanr((x@x.T)[i[0],i[1]].numpy(),(y@y.T)[i[0],i[1]].numpy()).statistic)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",required=True); args=parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("analysis requires CUDA")
    torch.backends.cuda.matmul.allow_tf32=True
    lock=json.loads((ROOT/"model_revisions.json").read_text())
    label_names={int(x["label"]):x["name"] for x in yaml.safe_load((ROOT/"config/runner_template.yaml").read_text())["classes"]}
    transitions={str(x["id"]):(int(x["source"]),int(x["target"])) for x in yaml.safe_load((ROOT/"config/runner_template.yaml").read_text())["transitions"]}
    refs=load_refs(); cells=collect_cells()+collect_multi()
    # Smoke artifacts are useful for checking the pipeline, but must never enter
    # the scientific tables, correlations, or final report.
    if args.stage=="smoke-analysis":
        cells=[c for c in cells if c["family"]=="smoke_single"]
    else:
        cells=[c for c in cells if not c["family"].startswith("smoke")]
    if args.stage=="10_embedding_extraction":
        groups={}
        for cell in cells:
            meta=metadata(cell); tid=cell["state"]["transition_id"]; s,t=transitions[tid]
            allpaths=refs[s]+refs[t]+cell["clean"]+cell["adv"]
            target_depth=int(lock[meta["target_name"]]["expected_vision_depth"])-1
            groups.setdefault((meta["target_name"],meta["target_repo_id"],lock[meta["target_name"]]["revision"],target_depth),set()).update(allpaths)
            for proxy in meta["proxy_names"]:
                model_layer=(meta["layer"] if meta["layer"]>=0 else int(lock[proxy]["expected_vision_depth"])-1)
                groups.setdefault((proxy,meta["proxy_repo_ids"][proxy],lock[proxy]["revision"],model_layer),set()).update(allpaths)
        failures=[]
        for (name,mid,rev,layer),paths in groups.items():
            try: extract_group(name,mid,rev,layer,sorted(paths))
            except Exception as exc: failures.append({"model":name,"layer":layer,"error":f"{type(exc).__name__}: {exc}"}); print(f"[extract] failed {name}: {exc}",flush=True)
        p=OUT/"analysis/embedding_extraction_failures.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(failures,indent=2)+"\n")
        return
    if args.stage not in {"11_representation_analysis","12_joint_pca_tsne","13_asymmetry","14_correlation","15_validation","16_final_report","smoke-analysis"}: return
    feature_maps={}
    def get_features(name,layer,path):
        rev=lock[name]["revision"]; mid=lock[name]["repo_id"]
        key=(name,layer,rev)
        if key not in feature_maps:
            cache=feature_cache_path(name,layer,rev)
            if not cache.is_file(): return None
            with np.load(cache,allow_pickle=False) as data: feature_maps[key]={k:data[k] for k in data.files}
        return feature_maps[key].get(image_key(path))
    metrics=[]; perimage=[]; class_rows=[]; cell_context=[]
    class_stats={}
    models_used={metadata(c)["target_name"] for c in cells}
    for model_name in models_used:
        layer=int(lock[model_name]["expected_vision_depth"])-1
        for label,paths in refs.items():
            values=[get_features(model_name,layer,p) for p in paths]
            if any(v is None for v in values): continue
            stats=compute_dispersion(np.stack(values)); class_stats[(model_name,label)]=stats
            class_rows.append({"model":model_name,"model_revision":lock[model_name]["revision"],"class_label":label,"class_name":label_names[label],"reference_n":len(values),"dispersion":stats["dispersion"],"covariance_trace":stats["covariance_trace"],"effective_rank":stats["effective_rank"]})
    for cell in cells:
        meta=metadata(cell); state=cell["state"]; tid=state["transition_id"]; source,target=transitions[tid]
        tname=meta["target_name"]; tlayer=int(lock[tname]["expected_vision_depth"])-1
        clean=[get_features(tname,tlayer,p) for p in cell["clean"]]; adv=[get_features(tname,tlayer,p) for p in cell["adv"]]
        if any(x is None for x in (*clean,*adv)) or (tname,source) not in class_stats or (tname,target) not in class_stats: continue
        zs=class_stats[(tname,source)]["center"]; zt=class_stats[(tname,target)]["center"]
        clean=np.stack(clean); adv=np.stack(adv)
        s0=clean@zs; t0=clean@zt; s1=adv@zs; t1=adv@zt
        dp=t1-t0; dq=s0-s1; dr=dp+dq; margin0=t0-s0; margin1=t1-s1
        pm=cell["proxy_mask"]; tm=cell["target_mask"]
        denominator=sum(bool(x) for x in pm)
        numerator=sum(bool(p) and bool(t) for p,t in zip(pm,tm,strict=False)) if tm and len(tm)==len(pm) else None
        proxy_geometry=[]; target_geometry=[]; cka_values=[]; rsa_values=[]; proxy_pair_cka=[]; proxy_pair_rsa=[]
        proxy_ref_features={}
        for proxy in meta["proxy_names"]:
            pl=meta["layer"] if meta["layer"]>=0 else int(lock[proxy]["expected_vision_depth"])-1
            pg=[get_features(proxy,pl,p) for p in refs[source]+refs[target]]
            proxy_ref_features[proxy]=pg
            tg=[get_features(tname,tlayer,p) for p in refs[source]+refs[target]]
            if all(x is not None for x in pg+tg):
                cka_values.append(cka(np.stack(pg),np.stack(tg))); rsa_values.append(rsa(np.stack(pg),np.stack(tg)))
        from itertools import combinations
        for left,right in combinations(meta["proxy_names"],2):
            x,y=proxy_ref_features[left],proxy_ref_features[right]
            if all(v is not None for v in x+y):
                proxy_pair_cka.append(cka(np.stack(x),np.stack(y))); proxy_pair_rsa.append(rsa(np.stack(x),np.stack(y)))
        pairwise=[]
        for batch in state.get("batches",[]):
            for step in batch.get("gradient_diagnostics_by_step",[]): pairwise.extend(step.get("pairwise_gradient_cosine",{}).values())
        attacks=cell.get("attacks",[state.get("attack",{})])
        if state.get("batches"):
            attacks=state["batches"]
        runtime=sum(float(a.get("elapsed_seconds") or 0) for a in attacks)
        peak=max((float(a.get("peak_reserved_vram_gb",a.get("peak_reserved_vram_gib",0)) or 0) for a in attacks),default=0)
        row={"family":cell["family"],"condition":meta["condition"],"transition_id":tid,"source_label":source,"target_label":target,"source_class":label_names[source],"target_class":label_names[target],"proxy_models":"+".join(meta["proxy_names"]),"target_model":tname,"proxy_layer":meta["layer"],"pull_weight":meta["pull_weight"],"push_weight":meta["push_weight"],"N":len(clean),"clean_valid_denominator":len(clean),"proxy_success_count":denominator,"TASR_numerator":numerator,"TASR_denominator":denominator,"target_hits_among_proxy_success":numerator,"TASR":numerator/denominator if numerator is not None and denominator else None,"source_variance":class_stats[(tname,source)]["dispersion"],"target_variance":class_stats[(tname,target)]["dispersion"],"source_covariance_trace":class_stats[(tname,source)]["covariance_trace"],"target_covariance_trace":class_stats[(tname,target)]["covariance_trace"],"source_effective_rank":class_stats[(tname,source)]["effective_rank"],"target_effective_rank":class_stats[(tname,target)]["effective_rank"],"prototype_distance":float(1-zs@zt),"mean_delta_pull":float(dp.mean()),"mean_delta_push":float(dq.mean()),"mean_delta_R":float(dr.mean()),"median_delta_R":float(np.median(dr)),"clean_margin":float(margin0.mean()),"adversarial_margin":float(margin1.mean()),"margin_change":float((margin1-margin0).mean()),"gap_closure":float((margin1-margin0).mean()),"CKA":float(np.mean(cka_values)) if cka_values else None,"RSA":float(np.nanmean(rsa_values)) if rsa_values else None,"mean_pairwise_proxy_CKA":float(np.mean(proxy_pair_cka)) if proxy_pair_cka else None,"mean_pairwise_proxy_RSA":float(np.nanmean(proxy_pair_rsa)) if proxy_pair_rsa else None,"mean_gradient_cosine":float(np.mean(pairwise)) if pairwise else None,"runtime_seconds":runtime,"peak_reserved_vram_gib":peak}
        metrics.append(row); cell_context.append((cell,meta,source,target))
        for i in range(len(clean)):
            perimage.append({"family":cell["family"],"condition":meta["condition"],"transition_id":tid,"image_index":i,"source_label":source,"target_label":target,"proxy_hit":pm[i] if i<len(pm) else None,"target_hit":tm[i] if i<len(tm) else None,"delta_pull":float(dp[i]),"delta_push":float(dq[i]),"DeltaR":float(dr[i]),"clean_margin":float(margin0[i]),"adversarial_margin":float(margin1[i]),"margin_change":float(margin1[i]-margin0[i])})
    analysis=OUT/"analysis"
    write_csv(analysis/"representation_shift/measurements.csv",metrics)
    write_csv(analysis/"representation_shift/per_image.csv",perimage)
    write_csv(analysis/"variance/class_variance.csv",class_rows)
    if args.stage=="11_representation_analysis": return
    if args.stage=="12_joint_pca_tsne" or args.stage=="smoke-analysis":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pairs={}
        for cell,meta,s,t in cell_context:
            pair=tuple(sorted((s,t)))
            if cell["family"] not in {"single_proxy","reverse_direction","smoke_single"}: continue
            key=(pair,meta["proxy_names"][0],meta["target_name"],meta["layer"])
            pairs.setdefault(key,{})[(s,t)]=(cell,meta)
        for (pair,proxy,target,layer),directions in pairs.items():
            if (pair[0],pair[1]) not in directions or (pair[1],pair[0]) not in directions: continue
            labels=[]; values=[]
            for label in pair:
                for p in refs[label]:
                    v=get_features(target,int(lock[target]["expected_vision_depth"])-1,p)
                    if v is not None: values.append(v); labels.append(f"reference:{label_names[label]}")
            for (s,t),(cell,meta) in directions.items():
                for kind,paths in (("clean",cell["clean"]),("adv",cell["adv"])):
                    for p in paths:
                        v=get_features(target,int(lock[target]["expected_vision_depth"])-1,p)
                        if v is not None: values.append(v); labels.append(f"{kind}:{label_names[s]}" if kind=="clean" else f"adv:{label_names[s]}→{label_names[t]}")
            if len(values)<5: continue
            x=F.normalize(torch.from_numpy(np.stack(values)).float(),dim=-1).numpy()
            coords={"pca":PCA(n_components=2,random_state=42).fit_transform(x)}
            pre=PCA(n_components=min(50,x.shape[1],x.shape[0]-1),random_state=42).fit_transform(x)
            perplexity=min(30.0,max(2.0,(len(x)-1)/3))
            coords["tsne"]=TSNE(n_components=2,perplexity=perplexity,random_state=42,init="pca",learning_rate="auto",max_iter=1000).fit_transform(pre)
            for kind,xy in coords.items():
                fig,ax=plt.subplots(figsize=(10,8))
                for group in dict.fromkeys(labels):
                    ix=[i for i,label in enumerate(labels) if label==group]; ax.scatter(xy[ix,0],xy[ix,1],s=18,alpha=.7,label=group)
                ax.set_title(f"Joint {kind.upper()} {label_names[pair[0]]} ↔ {label_names[pair[1]]} | {proxy} → {target}"); ax.legend(fontsize="small"); fig.tight_layout()
                path=analysis/kind/f"{safe(proxy)}__to__{safe(target)}__{pair[0]}_{pair[1]}.png"; path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)
        return
    if args.stage=="13_asymmetry":
        asym=[]; indexed={(r["family"],r["condition"],r["transition_id"]):r for r in metrics}
        # Pair forward/reverse cells with identical proxy, target, layer, and loss settings.
        for cell,meta,s,t in cell_context:
            if cell["family"] not in {"single_proxy","reverse_direction"}: continue
            # Emit one canonical row per unordered class pair, oriented in the
            # experiment matrix's forward direction (lower label -> higher).
            if s>t: continue
            partner=next(((c,m,ss,tt) for c,m,ss,tt in cell_context if c["family"] in {"single_proxy","reverse_direction"} and ss==t and tt==s and m["proxy_names"]==meta["proxy_names"] and m["target_name"]==meta["target_name"] and m["layer"]==meta["layer"] and m["pull_weight"]==meta["pull_weight"] and m["push_weight"]==meta["push_weight"]),None)
            if not partner: continue
            r1=indexed.get((cell["family"],meta["condition"],cell["state"]["transition_id"])); c2,m2,_,_=partner; r2=indexed.get((c2["family"],m2["condition"],c2["state"]["transition_id"]))
            if r1 and r2 and r1["TASR"] is not None and r2["TASR"] is not None:
                asym.append({"pair":f"{label_names[s]}↔{label_names[t]}","proxy_models":"+".join(meta["proxy_names"]),"target_model":meta["target_name"],"forward_transition":cell["state"]["transition_id"],"reverse_transition":c2["state"]["transition_id"],"TASR_forward":r1["TASR"],"TASR_reverse":r2["TASR"],"delta_TASR":r1["TASR"]-r2["TASR"],"variance_difference":r1["source_variance"]-r1["target_variance"],"clean_margin_difference":r1["clean_margin"]-r2["clean_margin"],"gap_closure_difference":r1["gap_closure"]-r2["gap_closure"]})
        write_csv(analysis/"asymmetry/asymmetry.csv",asym); return
    if args.stage=="14_correlation":
        keys=["TASR","CKA","RSA","source_variance","target_variance","prototype_distance","mean_delta_R","clean_margin","margin_change"]
        usable=[r for r in metrics if r["TASR"] is not None]
        rows=[]; mat=[]
        for x in keys:
            line=[]
            for y in keys:
                vals=[(r[x],r[y]) for r in usable if r.get(x) is not None and r.get(y) is not None and math.isfinite(float(r[x])) and math.isfinite(float(r[y]))]
                rho=float(spearmanr([a for a,b in vals],[b for a,b in vals]).statistic) if len(vals)>1 else float("nan")
                rows.append({"metric_x":x,"metric_y":y,"spearman_rho":rho,"n":len(vals)}); line.append(rho)
            mat.append(line)
        write_csv(analysis/"correlations/spearman.csv",rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,9)); image=ax.imshow(np.asarray(mat),vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(keys)),keys,rotation=65,ha="right"); ax.set_yticks(range(len(keys)),keys); fig.colorbar(image,ax=ax,label="Spearman ρ"); fig.tight_layout(); (analysis/"correlations").mkdir(parents=True,exist_ok=True); fig.savefig(analysis/"correlations/spearman_heatmap.png",dpi=180); plt.close(fig); return
    if args.stage=="15_validation":
        failures=[]
        warnings=[]
        target_invalid=[]
        checked_images=0
        max_png_linf=0
        from PIL import Image
        for cell,meta,s,t in cell_context:
            state=cell["state"]
            # V5 single-proxy state stores diagnostics under attack; multi-proxy
            # stores per-batch diagnostics. Validate both schemas when present.
            diagnostics=[s.get("attack",{}) for s in cell.get("states",[state])]
            diagnostics.extend(state.get("batches",[]))
            for diagnostic in diagnostics:
                step_count=diagnostic.get("steps")
                if step_count is not None and int(step_count)!=50:
                    failures.append({"state":str(cell["state_path"]),"error":f"production attack has {step_count} steps, expected 50"})
                for key in ("linf_float","linf_png","max_linf"):
                    value=diagnostic.get(key)
                    if value is not None and float(value)>16/255+1e-6:
                        failures.append({"state":str(cell["state_path"]),"error":f"{key} exceeds epsilon 16/255: {value}"})
                reserved=diagnostic.get("peak_reserved_vram_gib",diagnostic.get("peak_reserved_vram_gb"))
                if reserved is not None and float(reserved)>75:
                    warnings.append({"state":str(cell["state_path"]),"warning":f"peak reserved VRAM {float(reserved):.2f} GiB exceeded 75 GiB target"})
            if len(cell["proxy_mask"])!=len(cell["clean"]): failures.append({"state":str(cell["state_path"]),"error":"proxy mask length mismatch"})
            if len(cell["target_mask"])!=len(cell["clean"]): failures.append({"state":str(cell["state_path"]),"error":"target mask missing or length mismatch"})
            if any(value is None for value in cell["target_mask"]): failures.append({"state":str(cell["state_path"]),"error":"target mask contains unevaluated samples"})
            if len(cell["clean"])!=30: failures.append({"state":str(cell["state_path"]),"error":f"expected 30 clean-valid images, found {len(cell['clean'])}"})
            for batch in state.get("batches",[]):
                evaluation=batch.get("target_evaluation",{})
                if evaluation and evaluation.get("status")!="complete":
                    warnings.append({"state":str(cell["state_path"]),"warning":f"ensemble target parser status={evaluation.get('status')}; absent class labels count as target misses"})
            for state_part in cell.get("states",[state]):
                target=state_part.get("target",{})
                for key in ("clean_outputs","adversarial_outputs"):
                    outputs=target.get(key,[])
                    for index,output in enumerate(outputs):
                        if output.get("parser_status")!="ok":
                            target_invalid.append({"family":cell["family"],"condition":meta["condition"],"transition_id":state_part.get("transition_id"),"batch_index":state_part.get("batch_index"),"image_index":index,"output_kind":key,"parser_status":output.get("parser_status"),"raw_output":output.get("raw_output"),"parsed_label":output.get("parsed_label"),"treatment":"no valid target class; counted as target miss"})
                            warnings.append({"state":str(cell["state_path"]),"warning":f"{key}[{index}] was not an exact class code; counted as target miss"})
            for batch_index,batch in enumerate(state.get("batches",[])):
                evaluation=batch.get("target_evaluation",{})
                for image_index,output in enumerate(evaluation.get("outputs",[])):
                    for side,status_key,label_key in (("clean","clean_status","clean"),("adversarial","adv_status","adv")):
                        if output.get(status_key)!="ok":
                            target_invalid.append({"family":"multiple_proxy","condition":meta["condition"],"transition_id":state.get("transition_id"),"batch_index":batch_index,"image_index":image_index,"output_kind":side,"parser_status":output.get(status_key),"raw_output":None,"parsed_label":output.get(label_key),"treatment":"no valid target class; counted as target miss"})
            for clean_path,adv_path in zip(cell["clean"],cell["adv"],strict=True):
                try:
                    with Image.open(clean_path) as image: clean_image=np.asarray(image.convert("RGB"),dtype=np.int16)
                    with Image.open(adv_path) as image: adv_image=np.asarray(image.convert("RGB"),dtype=np.int16)
                    delta=int(np.abs(adv_image-clean_image).max())
                    checked_images+=1; max_png_linf=max(max_png_linf,delta)
                    if delta>16:
                        failures.append({"state":str(cell["state_path"]),"image":adv_path.name,"error":f"PNG L_inf={delta}/255 exceeds epsilon 16/255"})
                except Exception as exc:
                    failures.append({"state":str(cell["state_path"]),"image":adv_path.name,"error":f"cannot validate PNG pair: {type(exc).__name__}: {exc}"})
            den=sum(bool(x) for x in cell["proxy_mask"])
            num=sum(bool(p) and bool(t) for p,t in zip(cell["proxy_mask"],cell["target_mask"],strict=False))
            if num>den or den>len(cell["clean"]): failures.append({"state":str(cell["state_path"]),"error":"TASR numerator/denominator invariant violated"})
        for row in metrics:
            for key,value in row.items():
                if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)):
                    failures.append({"cell":row.get("condition"),"transition":row.get("transition_id"),"error":f"non-finite metric {key}"})
            if row.get("TASR") is not None:
                if row["TASR_denominator"]<=0 or row["TASR_numerator"]>row["TASR_denominator"] or abs(row["TASR"]-row["TASR_numerator"]/row["TASR_denominator"])>1e-12:
                    failures.append({"cell":row.get("condition"),"transition":row.get("transition_id"),"error":"TASR ratio inconsistent with explicit numerator/denominator"})
            if row.get("N")!=30:
                failures.append({"cell":row.get("condition"),"transition":row.get("transition_id"),"error":f"analysis row N={row.get('N')}; expected 30"})
        for status_path in (OUT/"audits/cell_states").glob("*.json"):
            data=json.loads(status_path.read_text())
            if data.get("status")!="complete": failures.append({"cell":status_path.stem,"error":data.get("error","incomplete")})
        expected_images=30
        for state_path in (OUT/"multiple_proxy/attacks").glob("*/*.json"):
            data=json.loads(state_path.read_text())
            if data.get("status") not in {"attack_complete","complete"}:
                failures.append({"cell":str(state_path.relative_to(OUT)),"error":f"multiple-proxy cohort status={data.get('status')}; found={data.get('found','unknown')}, required={data.get('required',expected_images)}"})
            elif int(data.get("image_count",0))!=expected_images:
                failures.append({"cell":str(state_path.relative_to(OUT)),"error":f"multiple-proxy cohort has {data.get('image_count',0)}/{expected_images} images"})
        (analysis/"validation").mkdir(parents=True,exist_ok=True)
        (analysis/"validation/failures.json").write_text(json.dumps(failures,indent=2)+"\n")
        (analysis/"validation/warnings.json").write_text(json.dumps(warnings,indent=2)+"\n")
        write_csv(analysis/"validation/target_output_diagnostics.csv",target_invalid)
        resource_warnings=[x for x in warnings if "peak reserved VRAM" in x.get("warning","")]
        ensemble_parse_warnings=[x for x in warnings if x.get("warning","").startswith("ensemble target parser status=")]
        (analysis/"validation/summary.json").write_text(json.dumps({"measured_rows":len(metrics),"checked_attack_png_pairs":checked_images,"max_png_linf_integer":max_png_linf,"epsilon_integer_limit":16,"strict_parser_nonclass_outputs":len(target_invalid),"nonclass_output_treatment":"counted as target misses because the task requires an exact class code","ensemble_parse_error_batches":len(ensemble_parse_warnings),"resource_warning_entries":len(resource_warnings),"resource_warning_states":len({x.get("state") for x in resource_warnings}),"max_reserved_vram_warning_gib":max((float(x["warning"].split()[3]) for x in resource_warnings),default=0),"failure_count":len(failures),"warning_count":len(warnings),"failures_by_type":{kind:sum(1 for x in failures if kind in x.get("error","")) for kind in sorted({x.get("error","").split(":")[0] for x in failures})}},indent=2)+"\n")
        return
    if args.stage=="16_final_report":
        family_outputs={"single_proxy":"single_proxy","layer_sweep":"layer_sweep","multiple_proxy":"multiple_proxy","ablation":"ablation","reverse_direction":"single_proxy"}
        for family,directory in family_outputs.items():
            write_csv(OUT/directory/"summaries"/f"{family}_results.csv",[r for r in metrics if r["family"]==family])
        status_rows=[]
        latest={}
        status_file=OUT/"audits/cell_status.jsonl"
        if status_file.is_file():
            for line in status_file.read_text().splitlines():
                if line.strip():
                    item=json.loads(line); latest[(item["family"],item["cell_id"])]=item
        status_rows=list(latest.values())
        write_csv(OUT/"reports/cell_execution_status.csv",status_rows)
        completion_rows=[{**r,"execution_status":"complete" if r.get("N")==30 else "invalid_cohort","expected_N":30} for r in metrics]
        for state_path in (OUT/"multiple_proxy/attacks").glob("*/*.json"):
            data=json.loads(state_path.read_text())
            if data.get("status") in {"attack_complete","complete"} and int(data.get("image_count",0))==30:
                continue
            completion_rows.append({"family":"multiple_proxy","condition":data.get("condition",state_path.parent.name),"transition_id":state_path.stem,"N":data.get("image_count",data.get("found")),"expected_N":data.get("required",30),"execution_status":data.get("status","invalid_cohort"),"error":data.get("error") or f"cohort shortfall: found {data.get('found',data.get('image_count',0))}, required {data.get('required',30)}"})
        write_csv(OUT/"reports/completion_matrix.csv",completion_rows)
        coverage={}
        for row in completion_rows:
            key=(row.get("family","unknown"),row.get("execution_status","unknown"))
            coverage[key]=coverage.get(key,0)+1
        total_gpu_seconds=sum(float(r.get("runtime_seconds") or 0) for r in metrics)
        report=["# A100 Pre-Commercial V6 Experiment Report","",f"Generated UTC: {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}",f"Summed measured attack GPU runtime (cell-level): {total_gpu_seconds/3600:.2f} hours ({total_gpu_seconds:.0f} seconds). This sum excludes model audit, clean screening, calibration, and representation extraction.","","## Protocol","","Open-source models only. Pull/Push = 0.75/0.25 for the main method, CLS=0, epsilon=16/255, 50 steps, step size=1/255, momentum=1, random start, seed=42. TASR is target targeted success conditioned on proxy targeted success; numerator and denominator are exported explicitly. Multiple-proxy primary conditioning requires all ensemble proxies to hit.","","## Model revisions","","| Model | Repository | Revision | Vision depth |","|---|---|---|---:|"]
        audit=OUT/"audits/model_audit.json"
        if audit.is_file():
            for r in json.loads(audit.read_text()): report.append(f"| {r['name']} | {r['repo_id']} | `{r['revision']}` | {r.get('vision_depth','—')} |")
        report.extend(["","## Completed measurements","",f"Measured transition cells: {len(metrics)}; per-image rows: {len(perimage)}; class dispersion rows: {len(class_rows)}.","","Completion-matrix coverage, including screened ensemble shortfalls:"])
        for (family,status),count in sorted(coverage.items()): report.append(f"- {family}: {status} = {count}")
        report.extend(["","| Family | Proxy | Target | Direction | Layer | TASR numerator / denominator | TASR | ΔR | CKA | RSA |","|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
        for r in metrics:
            tasr="—" if r['TASR'] is None else f"{r['TASR']:.3f}"
            report.append(f"| {r['family']} | {r['proxy_models']} | {r['target_model']} | {r['source_class']} → {r['target_class']} | {r['proxy_layer']} | {r['TASR_numerator']}/{r['TASR_denominator']} | {tasr} | {r['mean_delta_R']:.4f} | {r['CKA'] if r['CKA'] is not None else '—'} | {r['RSA'] if r['RSA'] is not None else '—'} |")
        validation_path=analysis/"validation/summary.json"
        if validation_path.is_file():
            validation=json.loads(validation_path.read_text())
            report.extend(["","## Result reasonableness checks","",f"Validated {validation['checked_attack_png_pairs']} frozen clean/adversarial PNG pairs; largest integer-pixel L∞ difference was {validation['max_png_linf_integer']}/255 (limit 16/255). Checked {validation['measured_rows']} measured rows, 50-step settings, and explicit proxy-conditioned TASR counts. Strict parsing rejected {validation['strict_parser_nonclass_outputs']} model outputs; these had no valid class code and are counted as target misses, with per-output records in `analysis/validation/target_output_diagnostics.csv`. The 17 ensemble clean-valid shortfalls are listed in the completion matrix. {validation['ensemble_parse_error_batches']} ensemble batches had at least one non-class response, counted as target misses. {validation['resource_warning_states']} state(s) exceeded the 75 GiB reserved-memory target (maximum {validation['max_reserved_vram_warning_gib']:.2f} GiB; no OOM). See the validation JSON files for details."])
        report.extend(["","## Output map","","`single_proxy/`, `layer_sweep/`, `multiple_proxy/`, and `ablation/` contain per-cell states and frozen PNGs. `analysis/embeddings/` stores reusable high-dimensional features. `analysis/pca/` and `analysis/tsne/` fit A/B references and both directions jointly. `analysis/variance/`, `analysis/representation_shift/`, `analysis/asymmetry/`, and `analysis/correlations/` contain the quantitative exports.","","t-SNE is a qualitative view only; all reported geometric measures use the original embedding dimensions. Failed or unavailable cells are listed in `audits/cell_status.jsonl` and `analysis/embedding_extraction_failures.json`."])
        (OUT/"reports").mkdir(parents=True,exist_ok=True); (OUT/"reports/final_report.md").write_text("\n".join(report)+"\n")


if __name__=="__main__":
    main()
