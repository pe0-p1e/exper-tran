#!/usr/bin/env python3
"""Audit pinned model configs, deepest-layer extraction, gradients, and target generation."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from primary_ml_cka.data.preprocessing import ensure_canvas

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; model audit requires the A100")
    torch.backends.cuda.matmul.allow_tf32 = True
    lock = json.loads((ROOT / "model_revisions.json").read_text())
    names = args.models or list(lock)
    image_paths = sorted((REPO / "data/imagenet_diverse10_minimal/train/n01443537").glob("*"))[:2]
    if len(image_paths) < 2:
        raise FileNotFoundError("Model audit requires two goldfish images")
    image_rows = []
    for image_path in image_paths:
        with Image.open(image_path) as source:
            image_rows.append(ensure_canvas(pil_to_tensor(source.convert("RGB")).float().div(255).unsqueeze(0).cuda(), 224).squeeze(0))
    canvas = torch.stack(image_rows)
    hf_home = Path(os.environ.get("HF_HOME", ".hf-cache"))
    results = []
    for index, name in enumerate(names, 1):
        spec = lock[name]
        record = {"name": name, "repo_id": spec["repo_id"], "revision": spec["revision"], "expected_depth": spec["expected_vision_depth"], "status": "error"}
        proxy = model = processor = generator = None
        try:
            if MODEL_REVISIONS.get(spec["repo_id"]) != spec["revision"]:
                raise RuntimeError("runtime revision pin differs from V6 lock")
            proxy = load_proxy(spec["repo_id"], hf_home, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
            model = getattr(proxy, "model", None)
            cfg = getattr(model, "config", None)
            vision_cfg = getattr(cfg, "vision_config", None)
            depth = next((getattr(c, field, None) for c in (vision_cfg, cfg) if c is not None for field in ("depth", "num_hidden_layers", "num_layers") if isinstance(getattr(c, field, None), int)), None)
            if depth is None and name == "DINOv2-Large":
                depth = int(cfg.num_hidden_layers)
            if int(depth) != int(spec["expected_vision_depth"]):
                raise RuntimeError(f"vision depth mismatch: actual {depth}, expected {spec['expected_vision_depth']}")
            layer = int(depth) - 1
            image = canvas.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                embedding = proxy.image_embeddings(image, representation_type="vision_encoder", layer=layer, pooling="mean")
                direction = torch.arange(embedding.embeddings.shape[-1], device="cuda", dtype=torch.float32).add_(1).view(1, -1)
                objective = (embedding.embeddings.float() * direction).sum()
                gradient = torch.autograd.grad(objective, image)[0]
            if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0:
                raise RuntimeError("vision gradient was non-finite or identically zero")
            record.update({"vision_depth": depth, "deepest_block": layer, "representation_shape": list(embedding.tokens.shape), "representation_dtype": str(embedding.tokens.dtype), "preprocessing": spec["family"], "gradient_linf": float(gradient.abs().max())})
            if spec["family"] not in {"clip", "siglip2", "dinov2"}:
                processor = getattr(proxy, "processor", None)
                if processor is None:
                    snapshot = local_snapshot(hf_home, spec["repo_id"], spec["revision"])
                    processor = load_processor(snapshot)
                if model is None:
                    snapshot = local_snapshot(hf_home, spec["repo_id"], spec["revision"])
                    model = load_target_for_generation(snapshot, torch.device("cuda"), precision="bf16")
                generator = TransformersTargetGenerator(model, processor)
                prompt = "Classify the main object into exactly one category: 0 goldfish; 1 monarch butterfly; 2 pineapple; 3 acoustic guitar; 4 laptop; 5 espresso; 6 volcano; 7 rocking chair; 8 soccer ball; 9 school bus. Return only one integer from 0 to 9."
                generated = [generator.generate_label(path, prompt) for path in image_paths]
                record["target_inference"] = {"status": [item.parser_status for item in generated], "parsed_labels": [item.parsed_label for item in generated]}
                if any(item.parser_status != "ok" for item in generated):
                    raise RuntimeError(f"target output parsing failed: {[item.parser_status for item in generated]}")
            else:
                record["target_inference"] = {"status": "not_a_target_in_matrix"}
            if not hasattr(proxy, "target_loss") or spec["family"] == "dinov2":
                record["closed_set_proxy_scoring"] = "prototype_only" if spec["family"] == "dinov2" else "n/a"
            else:
                with torch.inference_mode():
                    scored = proxy.target_loss(canvas, 2, "Classify into ten ImageNet classes", "none")
                if scored.class_logits is None or scored.class_logits.shape != (2, 10):
                    raise RuntimeError("proxy clean closed-set scoring did not return ten class logits")
                record["closed_set_proxy_scoring"] = "ok"
            record["status"] = "ok"
            print(f"[model-audit] {index}/{len(names)} ok {name} depth={depth} deepest={layer}", flush=True)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[model-audit] {index}/{len(names)} FAILED {name}: {record['error']}", flush=True)
        finally:
            if generator is not None:
                del generator
            if proxy is not None:
                del proxy
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            torch.cuda.empty_cache()
        results.append(record)
    output = REPO / "outputs/a100_precommercial_v6/audits/model_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    failures = [item for item in results if item["status"] != "ok"]
    print(f"model_audit={len(results)-len(failures)}/{len(results)} output={output}", flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
