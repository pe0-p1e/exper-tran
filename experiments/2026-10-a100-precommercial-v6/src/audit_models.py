#!/usr/bin/env python3
"""Pre-experiment A100 smoke and mandatory all-parameter BF16 audit."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/2026-09-a100-final-v5/src"))
sys.path.insert(0, str(ROOT / "src"))

from common import classification_prompt, transitions  # noqa: E402
from primary_ml_cka.config.schema import AttackConfig  # noqa: E402
from primary_ml_cka.data.preprocessing import ensure_canvas  # noqa: E402
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS  # noqa: E402
from primary_ml_cka.experiment.attack_generation import _cuda_images  # noqa: E402
from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation  # noqa: E402
from primary_ml_cka.models.backends.transformers_backend import load_processor  # noqa: E402
from primary_ml_cka.models.common.loading import local_snapshot  # noqa: E402
from primary_ml_cka.models.proxies.registry import load_proxy  # noqa: E402
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator  # noqa: E402
from strict_model_audit import audit_loaded_model  # noqa: E402


def resolve_depth(model: torch.nn.Module, family: str) -> int:
    cfg = model.config
    for candidate in (getattr(cfg, "vision_config", None), cfg):
        if candidate is None:
            continue
        for field in ("depth", "num_hidden_layers", "num_layers"):
            value = getattr(candidate, field, None)
            if isinstance(value, int) and value > 0:
                return value
    if family == "dinov2":
        return int(cfg.num_hidden_layers)
    raise RuntimeError(f"Cannot resolve vision depth for {family}/{type(model).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/a100_precommercial_v6_multi_repr_recovery")
    parser.add_argument("--models", nargs="*", help="Optional experiment-facing model subset")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise SystemExit("Strict BF16 model audit requires the available NVIDIA A100")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The A100 reports that BF16 is unsupported")
    lock = json.loads((ROOT / "model_revisions.json").read_text())
    matrix = yaml.safe_load((ROOT / "experiment_matrix.yaml").read_text())
    proxy_names = sorted({name for block in ("block_1", "block_2") for row in matrix["multiple_proxy"][block] for ensemble in row["ensembles"] for name in ensemble})
    target_names = sorted({row["target"] for block in ("block_1", "block_2") for row in matrix["multiple_proxy"][block]})
    names = args.models or sorted(set(proxy_names) | set(target_names))
    unknown = set(names) - set(lock)
    if unknown:
        raise RuntimeError(f"Unknown models in recovery matrix: {sorted(unknown)}")
    raw = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    prompt = classification_prompt(raw)
    fwd = [item for item in transitions(raw) if item.source < item.target]
    source = fwd[0].source
    canonical = out / "canonical_images"
    refs = out / "manifests/attack_cohorts" / f"{fwd[0].transition_id}.jsonl"
    if not refs.is_file():
        raise FileNotFoundError(f"Fixed GT-only smoke cohort missing: {refs}")
    from primary_ml_cka.data.manifests import read_manifest
    sample_records = read_manifest(refs)[:2]
    if len(sample_records) != 2:
        raise RuntimeError("Need two fixed cohort images for smoke test")
    images = _cuda_images(canonical, sample_records, 224)
    hf_home = Path(os.environ["HF_HOME"])
    results = []
    failures = []
    for index, name in enumerate(names, 1):
        spec = lock[name]
        repo_id, revision, family = spec["repo_id"], spec["revision"], spec["family"]
        if MODEL_REVISIONS.get(repo_id) != revision:
            raise RuntimeError(f"Code revision lock differs from model lock for {name}")
        is_proxy = name in proxy_names
        is_target = name in target_names
        row = {"model_name": name, "repo_id": repo_id, "revision": revision, "role": "proxy+target" if is_proxy and is_target else "proxy" if is_proxy else "target", "expected_vision_depth": int(spec["expected_vision_depth"]), "status": "error"}
        model = proxy = processor = generator = None
        try:
            if is_proxy:
                proxy = load_proxy(repo_id, hf_home, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
                model = proxy.model
                depth = resolve_depth(model, family)
                audit_loaded_model(model, model_name=name, repo_id=repo_id, revision=revision, role="proxy", output_dir=out, vision_depth=depth)
                if depth != int(spec["expected_vision_depth"]):
                    raise RuntimeError(f"Vision depth {depth} != pinned expected {spec['expected_vision_depth']}")
                image = images.detach().clone().requires_grad_(True)
                emb = proxy.image_embeddings(image, representation_type="vision_encoder", layer=depth - 1, pooling="mean")
                direction = torch.arange(emb.embeddings.shape[-1], device="cuda", dtype=torch.float32).add(1).view(1, -1)
                gradient = torch.autograd.grad((emb.embeddings.float() * direction).sum(), image)[0]
                if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0:
                    raise RuntimeError("BF16 model failed finite nonzero image-gradient smoke test")
                row.update({"vision_depth": depth, "deepest_block": depth - 1, "gradient_linf": float(gradient.abs().max()), "gradient_smoke": "pass"})
                del gradient, emb, image
            if is_target:
                snapshot = local_snapshot(hf_home, repo_id, revision)
                processor = load_processor(snapshot)
                # Audit happens inside loader and again into the detailed report
                # before the first generate() call.
                model = load_target_for_generation(snapshot, torch.device("cuda"), precision="bf16")
                depth = resolve_depth(model, family)
                audit_loaded_model(model, model_name=name, repo_id=repo_id, revision=revision, role="target_evaluator", output_dir=out, vision_depth=depth)
                if depth != int(spec["expected_vision_depth"]):
                    raise RuntimeError(f"Vision depth {depth} != pinned expected {spec['expected_vision_depth']}")
                # Do not query a target on clean attack-cohort images. Generation
                # smoke is deferred until the adversarial smoke PNGs are frozen.
                row["target_generation_smoke"] = "deferred_until_frozen_smoke_pngs"
            if family in {"clip", "siglip2", "dinov2"} and not is_target:
                row["target_generation_smoke"] = "not_applicable_visual_only_proxy"
            row["status"] = "pass"
            print(f"[dtype-audit] {index}/{len(names)} PASS {name} roles={row['role']}", flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(row)
            print(f"[dtype-audit] {index}/{len(names)} FAIL {name}: {row['error']}", flush=True)
        finally:
            if generator is not None: del generator
            if proxy is not None: del proxy
            if model is not None: del model
            if processor is not None: del processor
            gc.collect()
            torch.cuda.empty_cache()
        results.append(row)
    out_audit = out / "audits/model_audit.json"
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(results, indent=2) + "\n")
    if failures:
        raise SystemExit(f"BF16/model audit failed for {len(failures)} model(s); see {out_audit}")
    if len([r for r in results if r["role"] in {"proxy", "proxy+target"}]) < len(proxy_names):
        raise SystemExit("Not all required proxy models were audited")
    if len([r for r in results if r["role"] in {"target", "proxy+target"}]) < len(target_names):
        raise SystemExit("Not all target models were audited")
    print(f"[dtype-audit] PASS: {len(results)} model names; all floating parameters BF16 and unquantized", flush=True)


if __name__ == "__main__":
    main()
