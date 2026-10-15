#!/usr/bin/env python3
"""Run one fixed-cohort V6 equal-gradient ensemble, with strict BF16 weights."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers.utils import logging as transformers_logging

# Transformers 5 emits this deprecated processor-kwargs warning on every
# InternVL image batch; keep recovery logs focused on actionable errors.
transformers_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments/2026-09-a100-final-v5/src")]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-id", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--proxies", nargs="+", required=True)
    p.add_argument("--transitions", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs/a100_precommercial_v6_multi_repr_recovery")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if os.environ.get("PRIMARY_ML_CKA_STRICT_BF16") != "1":
        raise RuntimeError("Set PRIMARY_ML_CKA_STRICT_BF16=1 for this campaign")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Strict BF16 CUDA device required")

    from common import classification_prompt, transitions
    from primary_ml_cka.config.schema import AttackConfig
    from primary_ml_cka.data.manifests import read_manifest
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.experiment.attack_generation import _cuda_images, _save_png_batch
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.proxies.registry import load_proxy
    from primary_ml_cka.attack.optimization.projection import project_linf
    from strict_model_audit import audit_loaded_model

    out = args.output_dir.resolve()
    cache = Path(os.environ["HF_HOME"])
    registry = json.loads((ROOT / "model_revisions.json").read_text())
    names = list(dict.fromkeys(args.proxies))
    for name in [*names, args.target]:
        MODEL_REVISIONS[registry[name]["repo_id"]] = registry[name]["revision"]
    raw = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    prompt = classification_prompt(raw)
    tmap = {t.transition_id: t for t in transitions(raw)}
    eps = float(Fraction("16/255")); alpha = float(Fraction("1/255"))
    step_count = 2 if args.smoke else 50
    reference_records = [tuple(read_manifest(out / "manifests/class_references" / f"class_references_{i:02d}.jsonl")) for i in range(1, 11)]
    refs_digest = hashlib.sha256("".join(r.image_id for bank in reference_records for r in bank).encode()).hexdigest()
    canonical = out / "canonical_images"
    condition = slug(args.ensemble_id)
    model_objs: dict[str, object] = {}
    ref_banks: dict[str, tuple[torch.Tensor, ...]] = {}
    tap = {}
    parameter_gib: dict[str, float] = {}
    resident_names: set[str] = set()

    def load_one(name: str):
        if name in model_objs:
            model = model_objs[name]
            model.model.to("cuda")
            return model
        spec = registry[name]
        mid = spec["repo_id"]
        proxy = load_proxy(mid, cache, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
        # Assert again at the runner boundary before any embedding/gradient use.
        depth = None
        for c in (getattr(getattr(proxy.model, "config", None), "vision_config", None), getattr(proxy.model, "config", None)):
            if c is not None:
                for field in ("depth", "num_hidden_layers", "num_layers"):
                    val = getattr(c, field, None)
                    if isinstance(val, int) and val > 0:
                        depth = val; break
            if depth: break
        if depth is None:
            # vision-only encoders expose config.num_hidden_layers directly
            depth = getattr(getattr(proxy.model, "config", None), "num_hidden_layers", None)
        if not depth:
            raise RuntimeError(f"Cannot determine visual depth for {name}")
        audit_loaded_model(proxy.model, model_name=name, revision=spec["revision"], role="proxy", repo_id=mid, output_dir=out, vision_depth=int(depth))
        parameter_gib[name] = sum(p.numel()*p.element_size() for p in proxy.model.parameters() if p.is_floating_point())/2**30
        layer = int(depth) - 1
        tap[name] = {"repo_id": mid, "revision": spec["revision"], "visual_depth": depth, "block_index": layer, "pooling": "mean"}
        key = hashlib.sha256(f"{mid}|{spec['revision']}|{refs_digest}|{layer}|vision_encoder|mean".encode()).hexdigest()
        cache_file = out / "cache/reference_embeddings" / f"{key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            banks = tuple(torch.from_numpy(data[f"class_{i}"]).float().cuda() for i in range(10))
        else:
            banks = []
            with torch.inference_mode():
                for records in reference_records:
                    emb = []
                    for s in range(0, len(records), 4):
                        images = _cuda_images(canonical, tuple(records[s:s+4]), 224)
                        emb.append(proxy.image_embeddings(images, representation_type="vision_encoder", layer=layer, pooling="mean").embeddings.float().cpu())
                    banks.append(torch.cat(emb))
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, **{f"class_{i}": x.numpy() for i, x in enumerate(banks)})
            banks = tuple(x.cuda() for x in banks)
        ref_banks[name] = banks
        # Keep the model on CPU until the ensemble's measured BF16 parameter
        # footprint is known; choose a safe resident set below.
        proxy.model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
        return proxy

    try:
        for name in names:
            proxy = load_one(name)
            model_objs[name] = proxy
            print(f"[attack:{condition}] audited {name}; deepest={tap[name]['block_index']}/{tap[name]['visual_depth']-1}; parameters={parameter_gib[name]:.2f} GiB", flush=True)
        total_params_gib=sum(parameter_gib.values())
        # Gemma representation attacks call only the visual tower. Keep its
        # BF16 visual module on GPU and its unused language weights on CPU;
        # reload the full model only for optional post-freeze label diagnostics.
        visual_only_names={n for n in names if n.startswith("Gemma 4 ")}
        non_visual_names=[n for n in names if n not in visual_only_names]
        if sum(parameter_gib[n] for n in non_visual_names)<=45.0:
            resident_names=set(non_visual_names)
        else:
            largest=max(non_visual_names,key=lambda n:parameter_gib[n])
            next_largest=max((parameter_gib[n] for n in non_visual_names if n!=largest),default=0.0)
            if parameter_gib[largest]+next_largest<=65.0:
                # Leave measured activation headroom for one resident model;
                # smaller models are transferred sequentially in strict BF16.
                resident_names={largest}
        for name in visual_only_names:
            model_objs[name].model.to("cpu")
            model_objs[name].model.model.vision_tower.to("cuda")
        for name in resident_names:model_objs[name].model.to("cuda")
        offload_mode=("Gemma-vision-only-resident+" if visual_only_names else "")+("all-resident-BF16" if len(resident_names)==len(non_visual_names) else "largest-resident-plus-sequential-BF16-CPU-offload" if resident_names else "sequential-BF16-CPU-offload")
        print(f"[attack:{condition}] resident={sorted(resident_names)} visual-only-resident={sorted(visual_only_names)} total-params={total_params_gib:.2f} GiB mode={offload_mode}",flush=True)
        for tid in args.transitions:
            transition = tmap[tid]
            manifest = out / "manifests/attack_cohorts" / f"{tid}.jsonl"
            records = tuple(read_manifest(manifest))
            if len(records) != 30:
                raise RuntimeError(f"{tid}: expected 30 frozen GT cohort images, found {len(records)}")
            if args.smoke:
                records = records[:2]
            cell = out / "multiple_proxy/attacks" / condition / f"{tid}.json"
            cell.parent.mkdir(parents=True, exist_ok=True)
            expected_ids_for_cell = [r.image_id for r in records]
            if cell.exists():
                prior = json.loads(cell.read_text())
                if prior.get("status") == "attack_complete" and prior.get("image_ids") == expected_ids_for_cell and all((out / x).is_file() for x in prior.get("adv_paths", [])):
                    print(f"[attack:{condition}] resume {tid}", flush=True); continue
            bs = min(args.batch_size, len(records))
            start_generator = torch.Generator(device="cuda").manual_seed(42)
            cohort_noise = torch.empty((len(records), 3, 224, 224), device="cuda").uniform_(-eps, eps, generator=start_generator)
            batch_states = []
            for bi, start in enumerate(range(0, len(records), bs)):
                batch_records = records[start:start+bs]
                expected_ids = [r.image_id for r in batch_records]
                batch_dir = out / "multiple_proxy/attacks" / condition / tid / f"batch_{bi:02d}"
                state_path = batch_dir / "state.json"
                if state_path.is_file():
                    old = json.loads(state_path.read_text())
                    if old.get("status") == "attack_complete" and old.get("image_ids") == expected_ids and all((out / x).is_file() for x in old.get("adv_paths", [])):
                        batch_states.append(old); continue
                batch_dir.mkdir(parents=True, exist_ok=True)
                clean = _cuda_images(canonical, batch_records, 224)
                adv = (clean + cohort_noise[start:start+len(batch_records)]).clamp(0, 1)
                adv = torch.maximum(torch.minimum(adv, clean + eps), clean - eps).detach().requires_grad_(True)
                momentum = torch.zeros_like(adv)
                trace = []
                pair_sums: dict[str, list[float]] = {}
                proxy_deltas: dict[str, list[dict[str, float]]] = {n: [] for n in names}
                torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                for step in range(step_count):
                    grads = []
                    step_losses = {}
                    for name in names:
                        proxy = model_objs[name]
                        if name not in resident_names and name not in visual_only_names:proxy.model.to("cuda")
                        source_refs = ref_banks[name][transition.source - 1]
                        target_refs = ref_banks[name][transition.target - 1]
                        sc = F.normalize(F.normalize(source_refs.float(), dim=-1).mean(0), dim=0)
                        tc = F.normalize(F.normalize(target_refs.float(), dim=-1).mean(0), dim=0)
                        output = proxy.image_embeddings(adv, representation_type="vision_encoder", layer=tap[name]["block_index"], pooling="mean")
                        z = F.normalize(output.embeddings.float(), dim=-1)
                        st = (z * tc).sum(-1); ss = (z * sc).sum(-1)
                        loss = (0.75 * (1.0-st) + 0.25 * (1.0+ss)).mean()
                        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0].detach().float()
                        flat = grad.flatten(1)
                        normalized = grad / (flat.abs().sum(1).clamp_min(1e-12).view(-1,1,1,1))
                        grads.append(normalized)
                        step_losses[name] = float(loss.detach())
                        if step == 0 or step == step_count - 1:
                            with torch.no_grad():
                                clean_out = proxy.image_embeddings(clean, representation_type="vision_encoder", layer=tap[name]["block_index"], pooling="mean").embeddings.float()
                                clean_z = F.normalize(clean_out, dim=-1)
                                proxy_deltas[name].append({"step": step+1, "delta_pull": float(((z*tc).sum(-1)-(clean_z*tc).sum(-1)).mean()), "delta_push": float(((clean_z*sc).sum(-1)-(z*sc).sum(-1)).mean()), "delta_R": float((((z*tc).sum(-1)-(clean_z*tc).sum(-1))+((clean_z*sc).sum(-1)-(z*sc).sum(-1))).mean())})
                        if name not in resident_names and name not in visual_only_names:
                            proxy.model.to("cpu"); gc.collect(); torch.cuda.empty_cache()
                        del output, loss, grad
                    per_image_cos = {}
                    for i in range(len(names)):
                        for j in range(i+1, len(names)):
                            a = F.normalize(grads[i].flatten(1), dim=1); b = F.normalize(grads[j].flatten(1), dim=1)
                            cos = (a*b).sum(1)
                            key = f"{names[i]}|{names[j]}"
                            pair_sums.setdefault(key, []).extend(cos.detach().cpu().tolist())
                            per_image_cos[key] = {"mean": float(cos.mean()), "negative_fraction": float((cos<0).float().mean())}
                    fused = torch.stack(grads).mean(0)
                    fused_norm = fused.flatten(1).norm(dim=1)
                    momentum = momentum * 1.0 + fused
                    adv = project_linf(adv - alpha * momentum.sign(), clean, eps).detach().requires_grad_(True)
                    trace.append({"step": step+1, "proxy_losses": step_losses, "pairwise_gradient_cosine": per_image_cos, "fused_gradient_l2_mean": float(fused_norm.mean())})
                runtime = time.perf_counter()-t0
                linf_float, linf_png, adv_png = _save_png_batch(batch_dir, clean, adv.detach(), eps)
                if linf_png > eps + 1e-7:
                    raise RuntimeError(f"serialized L-inf {linf_png} exceeds epsilon")
                from primary_ml_cka.artifacts.png import load_png_tensor
                clean_png = torch.stack([load_png_tensor(batch_dir / f"{i:02d}_clean.png") for i in range(len(batch_records))]).cuda()
                proxy_hits: dict[str, list[bool]] = {}
                proxy_deltas = {n: [] for n in names}
                for name in names:
                    proxy = model_objs[name]
                    if name not in resident_names:proxy.model.to("cuda")
                    with torch.inference_mode():
                        clean_z = F.normalize(proxy.image_embeddings(clean_png, representation_type="vision_encoder", layer=tap[name]["block_index"], pooling="mean").embeddings.float(), dim=-1)
                        adv_z = F.normalize(proxy.image_embeddings(adv_png, representation_type="vision_encoder", layer=tap[name]["block_index"], pooling="mean").embeddings.float(), dim=-1)
                        source_refs = ref_banks[name][transition.source - 1]
                        target_refs = ref_banks[name][transition.target - 1]
                        sc = F.normalize(F.normalize(source_refs.float(), dim=-1).mean(0), dim=0)
                        tc = F.normalize(F.normalize(target_refs.float(), dim=-1).mean(0), dim=0)
                        clean_target = (clean_z * tc).sum(-1); adv_target = (adv_z * tc).sum(-1)
                        clean_source = (clean_z * sc).sum(-1); adv_source = (adv_z * sc).sum(-1)
                        proxy_deltas[name] = [{"image_id": rec.image_id, "delta_pull": float(adv_target[i]-clean_target[i]), "delta_push": float(clean_source[i]-adv_source[i]), "delta_R": float(adv_target[i]-clean_target[i]+clean_source[i]-adv_source[i]), "clean_margin": float(clean_target[i]-clean_source[i]), "adversarial_margin": float(adv_target[i]-adv_source[i])} for i,rec in enumerate(batch_records)]
                        if name == "DINOv2-Large":
                            centers = torch.stack([F.normalize(F.normalize(bank.float(),dim=-1).mean(0),dim=0) for bank in ref_banks[name]])
                            predicted = (adv_z @ centers.T).argmax(-1).add(1)
                        else:
                            scored = proxy.target_loss(adv_png, transition.target, prompt, "none")
                            predicted = scored.class_logits.argmax(dim=-1).add(1)
                        proxy_hits[name] = [int(v)==transition.target for v in predicted.tolist()]
                    if name not in resident_names:
                        proxy.model.to("cpu")
                        if name in visual_only_names:proxy.model.model.vision_tower.to("cuda")
                        gc.collect(); torch.cuda.empty_cache()
                rel_adv = [str(p.relative_to(out)) for p in [batch_dir / f"{i:02d}_adv.png" for i in range(len(batch_records))]]
                rel_clean = [str((batch_dir / f"{i:02d}_clean.png").relative_to(out)) for i in range(len(batch_records))]
                state = {"status":"attack_complete", "ensemble_id":args.ensemble_id, "target":args.target, "proxies":names, "transition_id":tid, "source_label":transition.source, "target_label":transition.target, "image_ids":expected_ids, "clean_paths":rel_clean, "adv_paths":rel_adv, "proxy_hit_masks":proxy_hits, "all_proxy_success_mask":[all(proxy_hits[n][i] for n in names) for i in range(len(batch_records))], "pull_weight":0.75, "push_weight":0.25, "steps":step_count, "epsilon":16/255, "step_size":1/255, "momentum":1.0, "random_start":True, "seed":42, "gradient_fusion":"mean_of_per_image_l1_normalized_proxy_gradients", "proxy_taps":tap, "proxy_parameter_gib":parameter_gib, "resident_proxies":sorted(resident_names), "visual_only_resident_proxies":sorted(visual_only_names), "execution_device_placement":{n:("BF16 visual tower CUDA; remaining model parameters BF16 CPU" if n in visual_only_names else "BF16 model CUDA during attack" if n in resident_names else "BF16 sequential CPU/GPU offload") for n in names}, "offload_mode":offload_mode, "proxy_representation_deltas":proxy_deltas, "gradient_cosine_summary":{k:{"mean":float(np.mean(v)),"median":float(np.median(v)),"negative_fraction":float(np.mean(np.asarray(v)<0)),"n":len(v)} for k,v in pair_sums.items()}, "gradient_trace":trace, "runtime_seconds":runtime, "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30, "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30, "forward_backward_counts":{"proxy_forward_per_image_step":len(names),"proxy_backward_per_image_step":len(names),"steps":step_count,"images":len(batch_records)}, "offload":"BF16 residency set selected from actual parameter bytes; independent per-proxy gradients and equal normalized fusion retained", "linf_float":linf_float, "linf_png":linf_png, "status_detail":"target not accessed during attack; proxy label diagnostics after PNG freeze"}
                state_path.write_text(json.dumps(state, indent=2)+"\n")
                batch_states.append(state)
                print(f"[attack:{condition}] {tid} batch {bi+1}/{(len(records)+bs-1)//bs}: {len(batch_records)} images; {runtime:.1f}s; peak={state['peak_reserved_gib']:.1f} GiB", flush=True)
            aggregate = {"status":"attack_complete", "ensemble_id":args.ensemble_id, "target":args.target, "proxies":names, "transition_id":tid, "source_label":tmap[tid].source, "target_label":tmap[tid].target, "image_ids":[r.image_id for r in records], "batches":batch_states, "adv_paths":[p for b in batch_states for p in b["adv_paths"]], "cohort_sha256":file_hash(out/"manifests/attack_cohorts"/f"{tid}.jsonl"), "target_evaluation_status":"pending"}
            cell.write_text(json.dumps(aggregate, indent=2)+"\n")
    finally:
        for model in model_objs.values():
            try: model.model.to("cpu")
            except Exception: pass
        model_objs.clear(); gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
