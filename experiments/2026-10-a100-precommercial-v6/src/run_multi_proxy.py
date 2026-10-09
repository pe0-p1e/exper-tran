#!/usr/bin/env python3
"""Resumable multi-proxy PGD with per-model Pull/Push and equal L1-gradient fusion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V5_SRC = REPO / "experiments/2026-09-a100-final-v5/src"


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def vision_depth(proxy) -> int:
    model = getattr(proxy, "model", None)
    config = getattr(model, "config", None)
    vision = getattr(config, "vision_config", None)
    for candidate in (vision, config):
        if candidate is None:
            continue
        for field in ("depth", "num_hidden_layers", "num_layers"):
            value = getattr(candidate, field, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError(f"Cannot resolve vision depth for {getattr(proxy, 'model_id', type(proxy).__name__)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxies", nargs="+", required=True, help="V6 experiment-facing names")
    parser.add_argument("--target", required=True)
    parser.add_argument("--condition", required=True, help="stable cohort slug from clean_screen.py")
    parser.add_argument("--transitions", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/a100_precommercial_v6")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--pull-weight", type=float, default=0.75)
    parser.add_argument("--push-weight", type=float, default=0.25)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if abs(args.pull_weight + args.push_weight - 1.0) > 1e-8:
        raise ValueError("Pull and Push weights must sum to 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-proxy attacks")
    import sys
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(V5_SRC))
    import yaml
    from common import classification_prompt, transitions, transition_dir
    from primary_ml_cka.artifacts.png import load_png_tensor, save_png_tensor
    from primary_ml_cka.config.schema import AttackConfig
    from primary_ml_cka.data.manifests import read_manifest
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.experiment.attack_generation import _cuda_images, _save_png_batch
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.proxies.registry import load_proxy
    from primary_ml_cka.attack.optimization.projection import project_linf
    from primary_ml_cka.attack.optimization.random_start import shared_random_start

    out = args.output_dir.resolve()
    root = Path(os.environ.get("HF_HOME", ".hf-cache"))
    matrix = yaml.safe_load((ROOT / "experiment_matrix.yaml").read_text())
    registry = json.loads((ROOT / "model_revisions.json").read_text())
    repo = {name: item["repo_id"] for name, item in registry.items()}
    for name in (*args.proxies, args.target):
        MODEL_REVISIONS[repo[name]] = registry[name]["revision"]
    raw = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    trans_by_id = {item.transition_id: item for item in transitions(raw)}
    selected_transitions = [trans_by_id[item] for item in args.transitions]
    if args.smoke:
        selected_transitions = selected_transitions[:1]
    prompt = classification_prompt(raw)
    canonical = out / "canonical_images"
    reference_records = tuple(
        read_manifest(out / "evaluation/manifests" / f"class_references_{label:02d}.jsonl")
        for label in range(1, 11)
    )
    manifest_digest = hashlib.sha256(
        json.dumps([[record.image_id for record in group] for group in reference_records], sort_keys=True).encode()
    ).hexdigest()
    proxies = {}
    reference_banks = {}
    tap_records = {}
    load_started = time.perf_counter()
    try:
        for name in args.proxies:
            model_id = repo[name]
            proxy = load_proxy(model_id, root, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
            proxies[name] = proxy
            depth = vision_depth(proxy)
            layer = depth - 1
            tap_records[name] = {"repo_id": model_id, "revision": registry[name]["revision"], "vision_depth": depth, "block": layer}
            cache_key = hashlib.sha256(f"{model_id}|{registry[name]['revision']}|{manifest_digest}|{layer}|vision_encoder|mean".encode()).hexdigest()
            cache_path = out / "analysis/embeddings/proxy_reference_cache" / f"{cache_key}.npz"
            if cache_path.is_file():
                archive = np.load(cache_path)
                reference_banks[name] = tuple(torch.from_numpy(archive[f"class_{i}"]).to("cuda") for i in range(10))
            else:
                banks = []
                for records in reference_records:
                    chunks = []
                    for start in range(0, len(records), 8):
                        images = _cuda_images(canonical, tuple(records[start : start + 8]), 224)
                        with torch.inference_mode():
                            output = proxy.image_embeddings(images, representation_type="vision_encoder", layer=layer, pooling="mean")
                        chunks.append(output.embeddings.detach().float().cpu())
                    banks.append(torch.cat(chunks))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, **{f"class_{i}": tensor.numpy() for i, tensor in enumerate(banks)})
                reference_banks[name] = tuple(tensor.to("cuda") for tensor in banks)
            print(f"[multiple-proxy] loaded {name}, deepest={layer}/{depth-1}, refs={manifest_digest[:10]}", flush=True)

        model_load_seconds = time.perf_counter() - load_started
        all_rows = []
        for t_index, transition in enumerate(selected_transitions, 1):
            cohort = out / "manifests/condition_cohorts" / args.condition / f"{transition.transition_id}.jsonl"
            if not cohort.is_file():
                state_path = out / "multiple_proxy/attacks" / safe(args.condition) / f"{transition.transition_id}.json"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps({"status": "error", "error": "clean cohort manifest missing"}, indent=2) + "\n")
                continue
            records = read_manifest(cohort)
            expected = int(matrix["data"]["attack_images_per_transition"])
            if len(records) != expected:
                state_path = out / "multiple_proxy/attacks" / safe(args.condition) / f"{transition.transition_id}.json"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps({"status": "insufficient_clean_valid", "found": len(records), "required": expected}, indent=2) + "\n")
                continue
            if args.smoke:
                records = records[:2]
            cell_path = out / "multiple_proxy/attacks" / safe(args.condition) / f"{transition.transition_id}.json"
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            if cell_path.is_file():
                old = json.loads(cell_path.read_text())
                if old.get("status") in {"attack_complete", "complete"} and int(old.get("image_count", 0)) == expected and not args.smoke:
                    print(f"[multiple-proxy] resume {t_index}/{len(selected_transitions)} {transition.transition_id}", flush=True)
                    continue
            batch_size = min(args.batch_size, len(records))
            batch_details = []
            for batch_index, start in enumerate(range(0, len(records), batch_size)):
                batch_records = tuple(records[start : start + batch_size])
                step_count = 2 if args.smoke else args.steps
                seed = int(matrix["attack"]["seed"]) + batch_index
                batch_dir = out / "multiple_proxy/attacks" / safe(args.condition) / transition.transition_id / f"batch_{batch_index:02d}"
                state_file = batch_dir / "state.json"
                if state_file.is_file() and not args.smoke:
                    old_batch = json.loads(state_file.read_text())
                    expected_ids = [record.image_id for record in batch_records]
                    if old_batch.get("status") in {"attack_complete", "complete"} and old_batch.get("image_ids") == expected_ids:
                        batch_details.append(old_batch)
                        continue
                batch_dir.mkdir(parents=True, exist_ok=True)
                clean = _cuda_images(canonical, batch_records, 224)
                epsilon = float(Fraction(str(matrix["attack"]["epsilon"])))
                step_size = float(Fraction(str(matrix["attack"]["step_size"])))
                momentum = float(matrix["attack"]["momentum"])
                adv = shared_random_start(clean, epsilon, seed)
                momentum_buffer = torch.zeros_like(adv)
                torch.cuda.reset_peak_memory_stats()
                gradients_by_step = []
                started = time.perf_counter()
                for step in range(step_count):
                    per_proxy_grads = []
                    norms = {}
                    for name in args.proxies:
                        proxy = proxies[name]
                        source_refs = reference_banks[name][transition.source - 1]
                        target_refs = reference_banks[name][transition.target - 1]
                        target_center = functional.normalize(functional.normalize(target_refs.float(), dim=-1).mean(dim=0), dim=0)
                        source_center = functional.normalize(functional.normalize(source_refs.float(), dim=-1).mean(dim=0), dim=0)
                        output = proxy.image_embeddings(adv, representation_type="vision_encoder", layer=tap_records[name]["block"], pooling="mean")
                        z = functional.normalize(output.embeddings.float(), dim=-1)
                        target_similarity = (z * target_center).sum(dim=-1)
                        source_similarity = (z * source_center).sum(dim=-1)
                        loss = (args.pull_weight * (1 - target_similarity) + args.push_weight * (1 + source_similarity)).mean()
                        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
                        flat = grad.flatten(1)
                        norms[name] = float(flat.norm(dim=1).mean().detach())
                        grad = grad / (flat.abs().sum(dim=1).view(-1, 1, 1, 1) + 1e-12)
                        per_proxy_grads.append(grad.detach())
                    pairwise = {}
                    for left in range(len(args.proxies)):
                        for right in range(left + 1, len(args.proxies)):
                            a = functional.normalize(per_proxy_grads[left].flatten(1), dim=1)
                            b = functional.normalize(per_proxy_grads[right].flatten(1), dim=1)
                            pairwise[f"{args.proxies[left]}|{args.proxies[right]}"] = float((a * b).sum(dim=1).mean())
                    fused = torch.stack(per_proxy_grads, dim=0).mean(dim=0)
                    momentum_buffer = momentum * momentum_buffer + fused
                    adv = project_linf(adv - step_size * momentum_buffer.sign(), clean, epsilon).detach().requires_grad_(True)
                    gradients_by_step.append({"step": step + 1, "proxy_gradient_l2": norms, "pairwise_gradient_cosine": pairwise})
                elapsed = time.perf_counter() - started
                linf_float, linf_png, _ = _save_png_batch(batch_dir, clean, adv.detach(), epsilon)
                if linf_png > epsilon + 1 / 255 + 1e-7:
                    raise RuntimeError(f"PNG perturbation violation {linf_png} > {epsilon}")
                proxy_hit_masks = {}
                for name in args.proxies:
                    proxy = proxies[name]
                    if name == "DINOv2-Large":
                        centers = []
                        for bank in reference_banks[name]:
                            centers.append(functional.normalize(functional.normalize(bank.float(), dim=-1).mean(dim=0), dim=0))
                        output = proxy.image_embeddings(adv.detach(), representation_type="vision_encoder", layer=tap_records[name]["block"], pooling="mean")
                        labels = (functional.normalize(output.embeddings.float(), dim=-1) @ torch.stack(centers).T).argmax(dim=1).add(1)
                    else:
                        with torch.inference_mode():
                            scored = proxy.target_loss(adv.detach(), transition.target, prompt, "none")
                        labels = scored.class_logits.argmax(dim=-1).add(1)
                    proxy_hit_masks[name] = [bool(int(label) == transition.target) for label in labels.tolist()]
                batch_state = {
                    "status": "attack_complete",
                    "target_evaluation_status": "pending",
                    "condition": args.condition,
                    "target_model": repo[args.target],
                    "target_revision": registry[args.target]["revision"],
                    "proxies": args.proxies,
                    "proxy_revisions": {name: registry[name]["revision"] for name in args.proxies},
                    "proxy_taps": tap_records,
                    "transition_id": transition.transition_id,
                    "source_label": transition.source,
                    "target_label": transition.target,
                    "image_ids": [record.image_id for record in batch_records],
                    "proxy_hit_masks": proxy_hit_masks,
                    "all_proxy_success_mask": [all(proxy_hit_masks[name][i] for name in args.proxies) for i in range(len(batch_records))],
                    "pull_weight": args.pull_weight,
                    "push_weight": args.push_weight,
                    "steps": step_count,
                    "step_size": step_size,
                    "epsilon": epsilon,
                    "momentum": momentum,
                    "random_start": True,
                    "seed": seed,
                    "linf_float": linf_float,
                    "linf_png": linf_png,
                    "elapsed_seconds": elapsed,
                    "model_load_seconds": model_load_seconds,
                    "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "gradient_diagnostics_by_step": gradients_by_step,
                }
                state_file.write_text(json.dumps(batch_state, indent=2) + "\n")
                batch_details.append(batch_state)
                del clean, adv, momentum_buffer
                torch.cuda.empty_cache()
            cell_state = {
                "status": "attack_complete",
                "condition": args.condition,
                "transition_id": transition.transition_id,
                "proxy_models": args.proxies,
                "target_model": args.target,
                "pull_weight": args.pull_weight,
                "push_weight": args.push_weight,
                "image_count": sum(len(item["image_ids"]) for item in batch_details),
                "proxy_success_numerator": sum(sum(item["all_proxy_success_mask"]) for item in batch_details),
                "proxy_success_denominator": sum(len(item["all_proxy_success_mask"]) for item in batch_details),
                "batches": batch_details,
            }
            cell_path.write_text(json.dumps(cell_state, indent=2) + "\n")
            print(f"[multiple-proxy] {t_index}/{len(selected_transitions)} attack {transition.transition_id} images={cell_state['image_count']} all-proxy-hits={cell_state['proxy_success_numerator']}/{cell_state['proxy_success_denominator']}", flush=True)
    except Exception as exc:
        for transition in selected_transitions:
            cell_path = out / "multiple_proxy/attacks" / safe(args.condition) / f"{transition.transition_id}.json"
            if not cell_path.exists() or json.loads(cell_path.read_text()).get("status") != "attack_complete":
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                cell_path.write_text(json.dumps({"status": "error", "condition": args.condition, "transition_id": transition.transition_id, "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        print(f"[multiple-proxy] ERROR condition={args.condition}: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        del reference_banks
        del proxies
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
