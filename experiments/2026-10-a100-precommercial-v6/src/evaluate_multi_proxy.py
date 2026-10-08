#!/usr/bin/env python3
"""Evaluate frozen multi-proxy PNGs using targets only after attack completion."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    import sys
    import yaml
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments/2026-09-a100-final-v5/src"))
    from common import classification_prompt, transitions
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--transitions", nargs="+")
    args = parser.parse_args()
    out = REPO / "outputs/a100_precommercial_v6"
    lock = json.loads((ROOT / "model_revisions.json").read_text())
    spec = lock[args.target]
    MODEL_REVISIONS[spec["repo_id"]] = spec["revision"]
    raw = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    prompt = classification_prompt(raw)
    transitions_by_id = {item.transition_id: item for item in transitions(raw)}
    tids = args.transitions or list(transitions_by_id)
    root = out / "multiple_proxy/attacks" / safe(args.condition)
    model = processor = generator = None
    rows = []
    try:
        snapshot = local_snapshot(Path(os.environ.get("HF_HOME", ".hf-cache")), spec["repo_id"], spec["revision"])
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"), precision="bf16")
        generator = TransformersTargetGenerator(model, processor)
        batch_size = 2 if args.target in {"Qwen3.5-27B", "Gemma 4 31B-it", "InternVL3.5-14B-HF"} else 8
        for tid in tids:
            cell = root / f"{tid}.json"
            if not cell.is_file():
                continue
            state = json.loads(cell.read_text())
            if state.get("status") != "attack_complete":
                continue
            transition = transitions_by_id[tid]
            for batch_index, batch in enumerate(state.get("batches", [])):
                batch_dir = root / tid / f"batch_{batch_index:02d}"
                outputs, hits = [], []
                image_ids = batch.get("image_ids", [])
                indexed = []
                for index in range(len(image_ids)):
                    indexed.extend((batch_dir / f"{index:02d}_clean.png", batch_dir / f"{index:02d}_adv.png"))
                parsed = []
                for start in range(0, len(indexed), batch_size):
                    parsed.extend(generator.generate_labels(indexed[start:start+batch_size], prompt))
                for index in range(len(image_ids)):
                    clean, adv = parsed[2*index:2*index+2]
                    outputs.append({"clean": clean.parsed_label, "clean_status": clean.parser_status, "adv": adv.parsed_label, "adv_status": adv.parser_status})
                    hits.append(clean.parsed_label == transition.source and adv.parsed_label == transition.target)
                batch["target_evaluation"] = {"outputs": outputs, "target_hit_mask": hits, "status": "complete" if all(x["clean_status"] == "ok" and x["adv_status"] == "ok" for x in outputs) else "parse_error"}
                rows.append({"condition": args.condition, "target": args.target, "transition_id": tid, "target_hits": sum(hits), "n": len(hits), "all_proxy_success": sum(batch.get("all_proxy_success_mask", [])), "target_hits_among_all_proxy_success": sum(bool(p) and bool(t) for p,t in zip(batch.get("all_proxy_success_mask", []), hits, strict=True))})
            state["target_evaluation_status"] = "complete"
            cell.write_text(json.dumps(state, indent=2) + "\n")
    finally:
        if generator is not None: del generator
        if model is not None: del model
        if processor is not None: del processor
        gc.collect(); torch.cuda.empty_cache()
    path = out / "multiple_proxy/evaluations" / f"{safe(args.condition)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
