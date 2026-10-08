#!/usr/bin/env python3
"""Cache proxy and target clean predictions, then freeze deterministic cohorts."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V5_SRC = REPO / "experiments/2026-09-a100-final-v5/src"


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def batched(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    import argparse
    import sys
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/a100_precommercial_v6")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(V5_SRC))
    from common import classification_prompt, transitions, transition_dir
    from primary_ml_cka.data.manifests import read_manifest, write_manifest
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.proxies.registry import load_proxy
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
    from primary_ml_cka.config.schema import AttackConfig
    from primary_ml_cka.experiment.attack_generation import _cuda_images

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for clean screening")
    matrix = yaml.safe_load((ROOT / "experiment_matrix.yaml").read_text())
    lock = json.loads((ROOT / "model_revisions.json").read_text())
    repo = {name: item["repo_id"] for name, item in lock.items()}
    trans = transitions(yaml.safe_load((ROOT / "config/runner_template.yaml").read_text()))
    out = args.output_dir.resolve()
    candidates_by_transition = {
        item.transition_id: read_manifest(transition_dir(out, item.transition_id) / "candidates.jsonl")
        for item in trans
    }
    classes = tuple(
        read_manifest(out / "evaluation/manifests" / f"class_references_{label:02d}.jsonl")
        for label in range(1, 11)
    )
    canonical = out / "canonical_images"
    prediction_root = out / "clean_screen/predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    hf_home = Path(os.environ.get("HF_HOME", ".hf-cache"))
    prompt = classification_prompt(yaml.safe_load((ROOT / "config/runner_template.yaml").read_text()))

    single = matrix["single_proxy"]["cross_family"] + matrix["single_proxy"]["intra_family"]
    layer = matrix["embedding_layer"]["pairs"]
    ensemble_specs = []
    for block in (matrix["multiple_proxy"]["block_1"], matrix["multiple_proxy"]["block_2"]):
        for row in block:
            for proxies in row["ensembles"]:
                ensemble_specs.append((tuple(proxies), row["target"]))
    ensemble_specs = list(dict.fromkeys(ensemble_specs))
    ablations = matrix["ablation"]["model_pairs"]
    conditions = set()
    for proxy, target, _ in single:
        conditions.add(((proxy,), target))
    for proxy, target in layer:
        conditions.add(((proxy,), target))
    for proxies, target in ensemble_specs:
        conditions.add((proxies, target))
    for proxy, target in ablations:
        conditions.add(((proxy,), target))
    conditions = sorted(conditions, key=lambda item: (item[1], item[0]))
    proxy_names = sorted({name for proxies, _ in conditions for name in proxies})
    target_names = sorted({target for _, target in conditions})
    failed_models: dict[str, str] = {}

    def write_prediction(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        temporary.replace(path)

    def cached(path: Path, records) -> dict[str, int | None] | None:
        if not path.is_file():
            return None
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        result = {str(row["image_id"]): row.get("parsed_label") for row in rows}
        if set(result) == {row.image_id for row in records}:
            return result
        return None

    # Proxy-side clean prediction cache. Generative/contrastive models use the
    # same 10-way closed-set score head as final proxy-success measurement.
    for model_index, name in enumerate(proxy_names, 1):
        model_id = repo[name]
        if model_id not in MODEL_REVISIONS:
            MODEL_REVISIONS[model_id] = lock[name]["revision"]
        pending = []
        for transition in trans:
            records = candidates_by_transition[transition.transition_id]
            p = prediction_root / "proxy_closed_set" / safe(model_id) / f"{transition.transition_id}.jsonl"
            if cached(p, records) is None:
                pending.append((transition, records, p))
        if not pending:
            print(f"[clean-screen] proxy {model_index}/{len(proxy_names)} cached {name}", flush=True)
            continue
        proxy = None
        try:
            proxy = load_proxy(model_id, hf_home, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
            if name == "DINOv2-Large":
                reference_features = []
                for bank in classes:
                    tensor = _cuda_images(canonical, bank, 224)
                    feature = proxy.image_embeddings(tensor, representation_type="vision_encoder", layer=-1, pooling="mean").embeddings.float()
                    reference_features.append(functional.normalize(feature.mean(dim=0), dim=0))
                centers = functional.normalize(torch.stack(reference_features), dim=-1)
            for transition, records, path in pending:
                outputs = []
                for group in batched(records, 16):
                    images = _cuda_images(canonical, tuple(group), 224)
                    if name == "DINOv2-Large":
                        features = proxy.image_embeddings(images, representation_type="vision_encoder", layer=-1, pooling="mean").embeddings.float()
                        labels = (functional.normalize(features, dim=-1) @ centers.T).argmax(dim=1).add(1).tolist()
                    else:
                        with torch.inference_mode():
                            scored = proxy.target_loss(images, 1, prompt, "none")
                        labels = scored.class_logits.argmax(dim=-1).add(1).tolist()
                    outputs.extend({"image_id": record.image_id, "parsed_label": int(label)} for record, label in zip(group, labels, strict=True))
                write_prediction(path, outputs)
                print(f"[clean-screen] proxy {name} {transition.transition_id} {len(outputs)}/{len(records)}", flush=True)
        except Exception as exc:
            failed_models[name] = f"{type(exc).__name__}: {exc}"
            print(f"[clean-screen] proxy failed {name}: {failed_models[name]}", flush=True)
        finally:
            if proxy is not None:
                del proxy
            gc.collect()
            torch.cuda.empty_cache()

    # Target-side clean generation uses only parsed text outputs. The model is
    # loaded once per target and batches are capped to protect 80GB headroom.
    from common import class_specs
    target_class = {int(c["label"]): str(c["name"]) for c in class_specs(yaml.safe_load((ROOT / "config/runner_template.yaml").read_text()))}
    large_targets = {"Qwen3.5-27B", "Gemma 4 31B-it", "InternVL3.5-14B-HF"}
    for model_index, name in enumerate(target_names, 1):
        model_id = repo[name]
        pending = []
        for transition in trans:
            records = candidates_by_transition[transition.transition_id]
            p = prediction_root / "target_generation" / safe(model_id) / f"{transition.transition_id}.jsonl"
            if cached(p, records) is None:
                pending.append((transition, records, p))
        if not pending:
            print(f"[clean-screen] target {model_index}/{len(target_names)} cached {name}", flush=True)
            continue
        model = processor = generator = proxy = None
        try:
            if name in proxy_names and name not in failed_models and name not in {"DINOv2-Large", "CLIP ViT-L/14", "SigLIP2-So400m-patch14-384"}:
                # Reuse the same BF16 weights for closed-set proxy scoring and
                # black-box target generation in this single-model pass.
                proxy = load_proxy(model_id, hf_home, torch.device("cuda"), AttackConfig(generative_precision="bf16"))
                model, processor = proxy.model, proxy.processor
            else:
                snapshot = local_snapshot(hf_home, model_id, MODEL_REVISIONS[model_id])
                processor = load_processor(snapshot)
                model = load_target_for_generation(snapshot, torch.device("cuda"), precision="bf16")
            generator = TransformersTargetGenerator(model, processor)
            batch_size = 2 if name in large_targets else 8
            for transition, records, path in pending:
                outputs = []
                for group in batched(records, batch_size):
                    paths = [canonical / record.relative_path for record in group]
                    parsed = generator.generate_labels(paths, prompt)
                    outputs.extend({"image_id": record.image_id, "parsed_label": item.parsed_label, "parser_status": item.parser_status} for record, item in zip(group, parsed, strict=True))
                write_prediction(path, outputs)
                print(f"[clean-screen] target {name} {transition.transition_id} {len(outputs)}/{len(records)}", flush=True)
        except Exception as exc:
            failed_models[f"target::{name}"] = f"{type(exc).__name__}: {exc}"
            print(f"[clean-screen] target failed {name}: {failed_models[f'target::{name}']}", flush=True)
        finally:
            if generator is not None:
                del generator
            if model is not None:
                del model
            if processor is not None:
                del processor
            if proxy is not None:
                del proxy
            gc.collect()
            torch.cuda.empty_cache()

    cohort_root = out / "manifests/condition_cohorts"
    summary = []
    for proxies, target in conditions:
        slug = safe("+".join(proxies) + "__to__" + target)
        for transition in trans:
            candidates = candidates_by_transition[transition.transition_id]
            prediction_sets = []
            available = True
            for name in (*proxies, target):
                role = "target_generation" if name == target else "proxy_closed_set"
                path = prediction_root / role / safe(repo[name]) / f"{transition.transition_id}.jsonl"
                values = cached(path, candidates)
                if values is None:
                    available = False
                    break
                prediction_sets.append(values)
            selected = []
            valid_count = 0
            if available:
                valid = [record for record in candidates if all(values.get(record.image_id) == transition.source for values in prediction_sets)]
                valid_count = len(valid)
                selected = valid[: int(matrix["data"]["attack_images_per_transition"])]
            manifest = cohort_root / slug / f"{transition.transition_id}.jsonl"
            write_manifest(manifest, tuple(selected))
            summary.append({"condition": slug, "proxies": "+".join(proxies), "target": target, "transition_id": transition.transition_id, "clean_valid_intersection": valid_count, "selected_count": len(selected), "required": int(matrix["data"]["attack_images_per_transition"]), "status": "ready" if len(selected) == int(matrix["data"]["attack_images_per_transition"]) else "insufficient_clean_valid" if available else "screen_failed"})
    summary_path = cohort_root / "screening_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    status_path = cohort_root / "screening_failures.json"
    status_path.write_text(json.dumps(failed_models, indent=2) + "\n")
    ready = sum(row["status"] == "ready" for row in summary)
    print(f"[clean-screen] cohorts ready={ready}/{len(summary)} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
