#!/usr/bin/env python3
"""Calibrate per-proxy attack batch sizes with a real one-step image gradient."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main() -> None:
    import yaml
    import sys
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments/2026-09-a100-final-v5/src"))
    from common import transitions
    from primary_ml_cka.config.schema import AttackConfig
    from primary_ml_cka.data.manifests import read_manifest
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.experiment.attack_generation import _cuda_images
    from primary_ml_cka.models.proxies.registry import load_proxy

    output = REPO / "outputs/a100_precommercial_v6"
    matrix = yaml.safe_load((ROOT / "experiment_matrix.yaml").read_text())
    lock = json.loads((ROOT / "model_revisions.json").read_text())
    by_name = {name: item["repo_id"] for name, item in lock.items()}
    all_proxies = set()
    for section in (matrix["single_proxy"],):
        for rows in section.values():
            all_proxies.update(row[0] for row in rows)
    all_proxies.update(proxy for pair in matrix["embedding_layer"]["pairs"] for proxy in [pair[0]])
    for block in (matrix["multiple_proxy"]["block_1"], matrix["multiple_proxy"]["block_2"]):
        all_proxies.update(proxy for row in block for ensemble in row["ensembles"] for proxy in ensemble)
    all_proxies.update(pair[0] for pair in matrix["ablation"]["model_pairs"])
    first = transitions(yaml.safe_load((ROOT / "config/runner_template.yaml").read_text()))[0]
    records = read_manifest(output / "evaluation/manifests/transitions" / first.transition_id / "candidates.jsonl")
    canonical = output / "canonical_images"
    classes = tuple(read_manifest(output / "evaluation/manifests" / f"class_references_{label:02d}.jsonl") for label in range(1, 11))
    hf_home = Path(os.environ.get("HF_HOME", ".hf-cache"))
    candidates = matrix["hardware"]["batch_candidates"]
    report = {"hardware": torch.cuda.get_device_name(0), "torch": torch.__version__, "max_peak_reserved_gib": matrix["hardware"]["max_peak_reserved_gib"], "models": {}}
    for name in sorted(all_proxies):
        model_id = by_name[name]
        MODEL_REVISIONS[model_id] = lock[name]["revision"]
        item = {"repo_id": model_id, "revision": lock[name]["revision"], "trials": [], "selected_batch_size": None}
        proxy = None
        try:
            proxy = load_proxy(model_id, hf_home, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
            model_config = getattr(getattr(proxy, "model", None), "config", None)
            vision = getattr(model_config, "vision_config", None)
            depth = next((getattr(c, f, None) for c in (vision, model_config) if c is not None for f in ("depth", "num_hidden_layers", "num_layers") if isinstance(getattr(c, f, None), int)), None)
            layer = int(depth) - 1 if depth else -1
            refs = []
            for label in (first.source, first.target):
                ref = read_manifest(output / "evaluation/manifests" / f"class_references_{label:02d}.jsonl")
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(ref), 8):
                        x = _cuda_images(canonical, tuple(ref[start:start+8]), 224)
                        z = proxy.image_embeddings(x, representation_type="vision_encoder", layer=layer, pooling="mean").embeddings.float()
                        chunks.append(z.cpu())
                refs.append(F.normalize(torch.cat(chunks).mean(0), dim=0).cuda())
            trial_images = tuple(records[i % len(records)] for i in range(max(candidates)))
            max_valid = 0
            for batch in candidates:
                torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
                row = {"batch_size": int(batch), "status": "error"}
                try:
                    x = _cuda_images(canonical, trial_images[:batch], 224).requires_grad_(True)
                    started = time.perf_counter()
                    z = proxy.image_embeddings(x, representation_type="vision_encoder", layer=layer, pooling="mean").embeddings.float()
                    z = F.normalize(z, dim=-1)
                    loss = (0.75 * (1 - (z * refs[1]).sum(-1)) + 0.25 * (1 + (z * refs[0]).sum(-1))).mean()
                    grad = torch.autograd.grad(loss, x)[0]
                    if not torch.isfinite(grad).all() or grad.abs().max() == 0:
                        raise RuntimeError("invalid image gradient")
                    torch.cuda.synchronize()
                    row.update(status="ok", seconds=time.perf_counter()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    max_valid = int(batch) if row["peak_reserved_gib"] <= report["max_peak_reserved_gib"] else max_valid
                    del x, z, grad, loss
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        row.update(status="oom", error=str(exc).splitlines()[0]); torch.cuda.empty_cache()
                    else:
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                item["trials"].append(row)
                print(f"[calibration] {name} batch={batch} {row['status']} reserved={row.get('peak_reserved_gib')}", flush=True)
                if row["status"] == "oom":
                    break
            item["vision_depth"] = depth
            item["deepest_block"] = layer
            item["selected_batch_size"] = max_valid or next((r["batch_size"] for r in item["trials"] if r["status"] == "ok"), None)
            item["selection_note"] = "largest measured batch with peak reserved <= limit; if all exceed limit, smallest successful batch"
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if proxy is not None: del proxy
            gc.collect(); torch.cuda.empty_cache()
        report["models"][name] = item
    path = output / "audits/performance_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
