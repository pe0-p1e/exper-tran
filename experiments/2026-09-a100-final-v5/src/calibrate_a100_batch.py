#!/usr/bin/env python3
"""Measure one real attack step at candidate batch sizes and select a safe size."""

import argparse
import csv
import gc
import tempfile
from pathlib import Path

import torch
from common import classification_prompt, load_experiment, pair_specs, transition_dir, transitions
from run_a100_final import attack_batches, read_state

from primary_ml_cka.data.manifests import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/a100_final.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/a100_final_v5"))
    parser.add_argument("--pair", default="P23")
    parser.add_argument("--transition", default="T01")
    parser.add_argument("--candidates", nargs="+", type=int, default=[8, 16, 24, 32, 48, 50])
    parser.add_argument("--memory-limit-gb", type=float, default=75.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch calibration")
    raw = load_experiment(args.config)
    raw = dict(raw)
    specs = pair_specs(raw)
    spec = dict(specs[args.pair])
    transition = next(
        (item for item in transitions(raw) if item.transition_id == args.transition), None
    )
    if transition is None:
        raise ValueError(f"Unknown transition: {args.transition}")
    output_dir = args.output_dir.resolve()
    source = read_manifest(
        transition_dir(output_dir, transition.transition_id) / f"{args.pair}_attack_images.jsonl"
    )
    classes = tuple(
        read_manifest(output_dir / "evaluation/manifests" / f"class_references_{label:02d}.jsonl")
        for label in range(1, 11)
    )
    prompt = classification_prompt(raw)
    rows = []
    for candidate in args.candidates:
        row = {
            "pair_id": args.pair,
            "transition_id": args.transition,
            "batch_size": candidate,
            "status": "error",
            "peak_allocated_gb": "",
            "peak_reserved_gb": "",
            "error": "",
        }
        if candidate < 2 or candidate > len(source):
            row["error"] = "candidate must be in [2, number of source images]"
            rows.append(row)
            continue
        torch.cuda.empty_cache()
        gc.collect()
        try:
            with tempfile.TemporaryDirectory(prefix="a100-batch-calibration-") as temporary:
                temp_output = Path(temporary) / "run"
                canonical = output_dir / "canonical_images"
                if not canonical.is_dir():
                    raise FileNotFoundError(f"Canonical image directory missing: {canonical}")
                temp_output.mkdir(parents=True, exist_ok=True)
                (temp_output / "canonical_images").symlink_to(canonical, target_is_directory=True)
                trial_raw = {
                    **raw,
                    "attack_count": candidate,
                    "batch_size": candidate,
                    "steps": 1,
                    "require_proxy_free_generation": False,
                }
                trial_spec = {
                    **spec,
                    "steps": 1,
                    "lambda_cls": 0.0,
                    "lambda_cka": 1.0,
                    "selected_rho": 0.0,
                }
                attack_batches(
                    pair_id=args.pair,
                    transition=transition,
                    spec=trial_spec,
                    source=source[:candidate],
                    class_references=classes,
                    raw=trial_raw,
                    output_dir=temp_output,
                    prompt=prompt,
                    resume=False,
                    batch_size=candidate,
                )
                state_path = (
                    temp_output
                    / str(trial_raw.get("state_namespace", "states_a100_v5"))
                    / args.pair
                    / args.transition
                    / "batch_00.json"
                )
                state = read_state(state_path)
                attack = state["attack"]
                row.update(
                    {
                        "status": "ok",
                        "peak_allocated_gb": attack["peak_allocated_vram_gb"],
                        "peak_reserved_gb": attack["peak_reserved_vram_gb"],
                    }
                )
        except torch.cuda.OutOfMemoryError as exc:
            row["status"] = "oom"
            row["error"] = str(exc).splitlines()[0]
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                row["status"] = "oom"
                row["error"] = str(exc).splitlines()[0]
                torch.cuda.empty_cache()
            else:
                raise
        rows.append(row)
        print(
            f"batch={candidate} status={row['status']} reserved={row['peak_reserved_gb']}",
            flush=True,
        )
    successful = [
        r
        for r in rows
        if r["status"] == "ok" and float(r["peak_reserved_gb"]) <= args.memory_limit_gb
    ]
    chosen = max((int(r["batch_size"]) for r in successful), default=None)
    report = output_dir / "calibration" / f"{args.pair}_{args.transition}_batch_sizes.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if chosen is None:
        raise RuntimeError(f"No batch size fit below {args.memory_limit_gb:g} GB; see {report}")
    print(
        f"selected_batch_size={chosen} (reserved <= {args.memory_limit_gb:g} GB); report={report}"
    )


if __name__ == "__main__":
    main()
