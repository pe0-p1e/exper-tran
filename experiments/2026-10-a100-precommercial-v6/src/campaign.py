#!/usr/bin/env python3
"""Resumable, cell-isolated orchestration for the V6 open-source campaign."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = REPO / "outputs/a100_precommercial_v6"
SRC = ROOT / "src"
PY = sys.executable


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def registry():
    return json.loads((ROOT / "model_revisions.json").read_text())


def matrix():
    return yaml.safe_load((ROOT / "experiment_matrix.yaml").read_text())


def run(args: list[str], *, name: str, check: bool = True) -> int:
    print(f"[campaign] {name}: {' '.join(args)}", flush=True)
    result = subprocess.run(args, cwd=REPO, env=os.environ.copy())
    if result.returncode and check:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result.returncode


def record_cell(family: str, cell_id: str, status: str, error: str = "") -> None:
    path = OUT / "audits/cell_status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"family": family, "cell_id": cell_id, "status": status, "error": error, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def transition_sets():
    raw = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    fwd = [str(item["id"]) for item in raw["transitions"] if int(item["source"]) < int(item["target"]) and (int(item["source"]) % 2 == 1)]
    rev = [str(item["id"]) for item in raw["transitions"] if int(item["source"]) > int(item["target"]) and (int(item["source"]) % 2 == 0)]
    return fwd, rev


def cohort_slug(proxies: tuple[str, ...], target: str) -> str:
    return safe("+".join(proxies) + "__to__" + target)


def prepare_single(*, family: str, proxies: tuple[str, ...], target: str, layer: int, pull: float, push: float, transitions: list[str], steps: int = 50, count: int = 30, smoke: bool = False, batch: int | None = None) -> tuple[Path, Path, str]:
    tag = safe(f"{family}__{'_'.join(proxies)}__to__{target}__l{layer}__p{pull:.2f}-{push:.2f}" + ("__smoke" if smoke else ""))
    cond = cohort_slug(proxies, target)
    workspace = OUT / family / "work" / tag
    config = OUT / "configs" / f"{tag}.yaml"
    args = [PY, str(SRC / "prepare_cell_workspace.py"), "--condition", cond, "--transitions", *transitions, "--workspace", str(workspace)]
    run(args, name=f"workspace {tag}")
    calibration = json.loads((OUT / "audits/performance_benchmark.json").read_text()) if (OUT / "audits/performance_benchmark.json").is_file() else {}
    selected_batch = batch or calibration.get("models", {}).get(proxies[0], {}).get("selected_batch_size") or 8
    config_args = [PY, str(SRC / "make_runner_config.py"), "--output", str(config), "--layer", str(layer), "--batch-size", str(selected_batch), "--pull-weight", str(pull), "--push-weight", str(push), "--steps", str(steps), "--attack-count", str(count), "--objective-tag", tag, "--state-namespace", f"states_v6_{tag}"]
    run(config_args, name=f"config {tag}")
    return workspace, config, tag


def run_single(*, family: str, proxy: str, target: str, layer: int, pull: float, push: float, transitions: list[str], steps: int = 50, count: int = 30, smoke: bool = False) -> None:
    workspace, config, tag = prepare_single(family=family, proxies=(proxy,), target=target, layer=layer, pull=pull, push=push, transitions=transitions, steps=steps, count=count, smoke=smoke)
    args = [PY, str(SRC / "run_v5_cell.py"), "--config", str(config), "--output-dir", str(workspace), "--proxy-repo", registry()[proxy]["repo_id"], "--target-repo", registry()[target]["repo_id"], "--transitions", *transitions]
    if smoke:
        args.append("--smoke")
    run(args, name=tag)
    (workspace / "cell_meta.json").write_text(json.dumps({"family": family, "condition": tag, "proxy_name": proxy, "target_name": target, "proxy_repo_id": registry()[proxy]["repo_id"], "target_repo_id": registry()[target]["repo_id"], "proxy_revision": registry()[proxy]["revision"], "target_revision": registry()[target]["revision"], "layer": layer, "pull_weight": pull, "push_weight": push, "transitions": transitions, "attack_count": count, "steps": steps, "smoke": smoke}, indent=2) + "\n")


def run_cell(family: str, cell_id: str, callback) -> None:
    status_file = OUT / "audits/cell_states" / f"{safe(family)}__{safe(cell_id)}.json"
    # Earlier failed multi-proxy attempts incorrectly wrote complete statuses
    # after the runner swallowed an exception. Always retry this family; its
    # batch artifacts themselves remain resumable.
    retry_family = family in {"multiple_proxy", "smoke"}
    if not retry_family and status_file.is_file() and json.loads(status_file.read_text()).get("status") == "complete":
        print(f"[{family}] resume complete {cell_id}", flush=True)
        return
    started = time.perf_counter()
    try:
        callback()
        payload = {"status": "complete", "elapsed_seconds": time.perf_counter() - started}
        record_cell(family, cell_id, "complete")
    except Exception as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started}
        record_cell(family, cell_id, "error", payload["error"])
        print(f"[{family}] CELL FAILED {cell_id}: {payload['error']}", flush=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(payload, indent=2) + "\n")


def one_proxy_rows():
    m = matrix()
    rows = []
    for relation in ("cross_family", "intra_family"):
        for proxy, target, layer in m["single_proxy"][relation]:
            rows.append((proxy, target, int(layer)))
    return rows


def ensemble_rows():
    m = matrix()
    rows = []
    for block_key in ("block_1", "block_2"):
        for row in m["multiple_proxy"][block_key]:
            for ensemble in row["ensembles"]:
                rows.append((tuple(ensemble), row["target"], block_key))
    return list(dict.fromkeys(rows))


def stage(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = matrix()
    fwd, rev = transition_sets()
    if name == "00_preflight":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if "A100" not in torch.cuda.get_device_name(0):
            print(f"[preflight] warning: GPU is {torch.cuda.get_device_name(0)}", flush=True)
        root = Path(os.environ.get("IMAGENET_ROOT", REPO / "data/imagenet_diverse10_minimal"))
        for cls in yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())["classes"]:
            images = list((root / "train" / cls["wnid"]).glob("*"))
            if len(images) < 112:
                raise RuntimeError(f"{cls['name']} requires >=112 source images, found {len(images)}")
        os.environ["HF_HOME"] = str(Path(os.environ.get("HF_HOME", REPO / ".hf-cache")).resolve())
        run([PY, str(SRC / "download_models.py")], name="download-pinned-models")
    elif name == "01_model_audit":
        run([PY, str(SRC / "audit_models.py")], name="audit-all-pinned-models", check=False)
    elif name == "02_dataset_prepare":
        run([PY, str(REPO / "experiments/2026-09-a100-final-v5/src/prepare_data.py"), "--config", str(ROOT / "config/runner_template.yaml"), "--output-dir", str(OUT)], name="prepare-dataset")
        from primary_ml_cka.data.manifests import read_manifest
        for label in range(1, 11):
            refs = read_manifest(OUT / "evaluation/manifests" / f"class_references_{label:02d}.jsonl")
            if len(refs) != 48: raise RuntimeError(f"class {label}: expected 48 refs, found {len(refs)}")
        print("[dataset] verified 48 references/class, 64 candidates/direction, frozen deterministic manifests", flush=True)
    elif name == "03_clean_screen":
        run([PY, str(SRC / "clean_screen.py"), "--output-dir", str(OUT)], name="clean-screen-all-models", check=False)
    elif name == "04_batch_calibration":
        run([PY, str(SRC / "calibrate_batches.py")], name="calibrate-all-proxies", check=False)
    elif name == "05_single_proxy":
        rows = one_proxy_rows()
        total = len(rows) * len(fwd)
        for index, (proxy, target, layer) in enumerate(rows, 1):
            cid = f"{proxy}__to__{target}__forward"
            run_cell("single_proxy", cid, lambda p=proxy,t=target,l=layer: run_single(family="single_proxy", proxy=p, target=t, layer=l, pull=.75, push=.25, transitions=fwd))
            print(f"[Single Proxy] {index}/{len(rows)} conditions ({min(total,index*len(fwd))}/{total} transitions scheduled)", flush=True)
    elif name == "06_embedding_layers":
        m = matrix()
        lock = registry()
        from math import ceil
        for proxy, target in m["embedding_layer"]["pairs"]:
            depth = int(lock[proxy]["expected_vision_depth"])
            for percent in m["embedding_layer"]["normalized_depths"]:
                layer = min(depth - 1, max(0, ceil(depth * float(percent)) - 1))
                cid = f"{proxy}__to__{target}__{percent:.2f}"
                run_cell("layer_sweep", cid, lambda p=proxy,t=target,l=layer: run_single(family="layer_sweep", proxy=p, target=t, layer=l, pull=.75, push=.25, transitions=fwd))
    elif name == "07_multiple_proxy":
        for proxies, target, block in ensemble_rows():
            cond = cohort_slug(proxies, target)
            cid = f"{block}__{'_'.join(proxies)}__to__{target}"
            args = [PY, str(SRC / "run_multi_proxy.py"), "--proxies", *proxies, "--target", target, "--condition", cond, "--transitions", *fwd, "--output-dir", str(OUT)]
            def execute_multi(a=args, c=cond, t=target, cell=cid):
                run(a, name=cell)
                run([PY, str(SRC / "evaluate_multi_proxy.py"), "--condition", c, "--target", t, "--transitions", *fwd], name=f"evaluate-{cell}")
            run_cell("multiple_proxy", cid, execute_multi)
    elif name == "08_ablation":
        for proxy, target in m["ablation"]["model_pairs"]:
            layer = int(registry()[proxy]["expected_vision_depth"]) - 1
            for pull, push in m["ablation"]["pull_push_weights"]:
                cid = f"{proxy}__to__{target}__pull{pull:.2f}_push{push:.2f}"
                run_cell("ablation", cid, lambda p=proxy,t=target,l=layer,w=pull,v=push: run_single(family="ablation", proxy=p, target=t, layer=l, pull=w, push=v, transitions=fwd))
    elif name == "09_reverse_direction":
        for proxy, target, layer in one_proxy_rows():
            cid = f"{proxy}__to__{target}__reverse"
            run_cell("reverse_direction", cid, lambda p=proxy,t=target,l=layer: run_single(family="reverse_direction", proxy=p, target=t, layer=l, pull=.75, push=.25, transitions=rev))
    elif name in {"10_embedding_extraction", "11_representation_analysis", "12_joint_pca_tsne", "13_asymmetry", "14_correlation", "15_validation", "16_final_report"}:
        run([PY, str(SRC / "analyze_campaign.py"), "--stage", name], name=name)
    elif name == "smoke":
        def smoke_one(family, proxy, target, layer, pull=.75, push=.25):
            tag = f"smoke_{family}_{proxy}_{target}_{layer}_{pull}_{push}"
            def execute():
                run_single(family=family, proxy=proxy, target=target, layer=layer, pull=pull, push=push, transitions=[fwd[0]], steps=2, count=2, smoke=True)
            run_cell("smoke", tag, execute)
        first_proxy, first_target, first_layer = one_proxy_rows()[0]
        smoke_one("smoke_single", first_proxy, first_target, first_layer)
        run_cell("smoke", f"reverse_{first_proxy}_{first_target}", lambda: run_single(family="smoke_single", proxy=first_proxy, target=first_target, layer=first_layer, pull=.75, push=.25, transitions=[rev[0]], steps=2, count=2, smoke=True))
        smoke_one("smoke_layer", first_proxy, first_target, 0)
        # Invoke the same stable condition once more and verify completed state reuse.
        ws, cfg, _ = prepare_single(family="smoke_single", proxies=(first_proxy,), target=first_target, layer=first_layer, pull=.75, push=.25, transitions=[fwd[0]], steps=2, count=2, smoke=True)
        run([PY, str(SRC / "run_v5_cell.py"), "--config", str(cfg), "--output-dir", str(ws), "--proxy-repo", registry()[first_proxy]["repo_id"], "--target-repo", registry()[first_target]["repo_id"], "--transitions", fwd[0], "--smoke"], name="smoke-resume-check")
        a_proxy, a_target = m["ablation"]["model_pairs"][0]
        smoke_one("smoke_ablation", a_proxy, a_target, int(registry()[a_proxy]["expected_vision_depth"])-1, 1.0, 0.0)
        proxies, target, _ = next(row for row in ensemble_rows() if len(row[0]) == 1)
        def multi_smoke():
            args = [PY, str(SRC / "run_multi_proxy.py"), "--proxies", *proxies, "--target", target, "--condition", cohort_slug(proxies,target), "--transitions", fwd[0], "--output-dir", str(OUT), "--smoke"]
            run(args, name="smoke-multi-proxy")
            run([PY, str(SRC / "evaluate_multi_proxy.py"), "--condition", cohort_slug(proxies,target), "--target", target, "--transitions", fwd[0]], name="smoke-multi-target-evaluation")
        run_cell("smoke", "multi_proxy", multi_smoke)
        run([PY, str(SRC / "analyze_campaign.py"), "--stage", "10_embedding_extraction"], name="smoke-feature-extraction")
        run([PY, str(SRC / "analyze_campaign.py"), "--stage", "smoke-analysis"] , name="smoke-analysis")
        smoke_states = [json.loads(p.read_text()) for p in (OUT / "audits/cell_states").glob("smoke__*.json")]
        failed = [p for p in smoke_states if p.get("status") != "complete"]
        if failed:
            raise RuntimeError(f"smoke checks failed: {failed}")
    else:
        raise ValueError(f"unknown campaign stage {name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: campaign.py STAGE")
    stage(sys.argv[1])
