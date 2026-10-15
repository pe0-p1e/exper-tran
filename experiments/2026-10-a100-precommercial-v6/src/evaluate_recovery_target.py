#!/usr/bin/env python3
"""Evaluate already-frozen recovery PNGs with one BF16 target at a time."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments/2026-09-a100-final-v5/src")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs/a100_precommercial_v6_multi_repr_recovery")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()
    if os.environ.get("PRIMARY_ML_CKA_STRICT_BF16") != "1":
        raise RuntimeError("Set PRIMARY_ML_CKA_STRICT_BF16=1")
    out = args.output_dir.resolve()
    registry = json.loads((ROOT / "model_revisions.json").read_text())
    spec = registry[args.target]
    from common import classification_prompt
    from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
    from primary_ml_cka.models.backends.transformers_backend import load_processor
    from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
    from primary_ml_cka.models.common.loading import local_snapshot
    from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
    from strict_model_audit import audit_loaded_model

    MODEL_REVISIONS[spec["repo_id"]] = spec["revision"]
    raw = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    prompt = classification_prompt(raw)
    root = out / "multiple_proxy/attacks"
    ensembles = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    device = torch.device("cuda")
    model = processor = generator = None
    try:
        snapshot = local_snapshot(Path(os.environ["HF_HOME"]), spec["repo_id"], spec["revision"])
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, device, precision="bf16")
        audit_loaded_model(model, model_name=args.target, revision=spec["revision"], role="target", repo_id=spec["repo_id"], output_dir=out)
        generator = TransformersTargetGenerator(model, processor)
        rows = []
        for ensemble_dir in ensembles:
            for cell_path in sorted(ensemble_dir.glob("T*.json")):
                state = json.loads(cell_path.read_text())
                if state.get("target") != args.target or state.get("status") != "attack_complete":
                    continue
                already = out / "multiple_proxy/evaluations" / ensemble_dir.name / f"{cell_path.stem}.json"
                if already.is_file():
                    done = json.loads(already.read_text())
                    if done.get("status") == "complete" and len(done.get("images", [])) == 30:
                        rows.append(done); continue
                outputs = []
                all_paths = []
                for batch in state["batches"]:
                    all_paths.extend([(out / c, out / a, iid) for c, a, iid in zip(batch["clean_paths"], batch["adv_paths"], batch["image_ids"], strict=True)])
                for start in range(0, len(all_paths), args.batch_size):
                    pack = all_paths[start:start+args.batch_size]
                    pair_paths = [path for clean_path, adv_path, _ in pack for path in (clean_path, adv_path)]
                    with torch.inference_mode():
                        parsed = generator.generate_labels(pair_paths, prompt)
                    for i, (clean_path, adv_path, image_id) in enumerate(pack):
                        clean, adv = parsed[2*i], parsed[2*i+1]
                        outputs.append({"image_id":image_id, "clean_path":str(clean_path.relative_to(out)), "adv_path":str(adv_path.relative_to(out)), "clean_raw":clean.raw_output, "clean_label":clean.parsed_label, "clean_status":clean.parser_status, "adv_raw":adv.raw_output, "adv_label":adv.parsed_label, "adv_status":adv.parser_status})
                src, tgt = int(state["source_label"]), int(state["target_label"])
                clean_correct = [x for x in outputs if x["clean_label"] == src]
                hits = [x for x in outputs if x["adv_label"] == tgt]
                hits_clean = [x for x in clean_correct if x["adv_label"] == tgt]
                result = {"status":"complete" if len(outputs)==30 else "incomplete", "ensemble_id":state["ensemble_id"], "target":args.target, "transition_id":cell_path.stem, "source_label":src, "target_label":tgt, "images":outputs, "target_hit_numerator":len(hits), "target_hit_denominator":len(outputs), "target_ASR_all":len(hits)/len(outputs) if outputs else None, "clean_correct_numerator":len(clean_correct), "clean_correct_denominator":len(outputs), "target_clean_accuracy":len(clean_correct)/len(outputs) if outputs else None, "target_ASR_clean_correct_numerator":len(hits_clean), "target_ASR_clean_correct_denominator":len(clean_correct), "target_ASR_clean_correct":len(hits_clean)/len(clean_correct) if clean_correct else None, "clean_parse_success":sum(x["clean_status"]=="ok" for x in outputs), "adv_parse_success":sum(x["adv_status"]=="ok" for x in outputs)}
                already.parent.mkdir(parents=True, exist_ok=True)
                already.write_text(json.dumps(result, indent=2)+"\n")
                rows.append(result)
                print(f"[target-eval:{args.target}] {ensemble_dir.name}/{cell_path.stem} ASR={len(hits)}/{len(outputs)} clean-correct={len(clean_correct)}", flush=True)
        summary = out / "multiple_proxy/evaluations" / f"target_{args.target.replace('/', '_')}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(rows, indent=2)+"\n")
    finally:
        if generator is not None: del generator
        if model is not None: del model
        if processor is not None: del processor
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
