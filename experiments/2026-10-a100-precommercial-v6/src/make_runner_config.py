#!/usr/bin/env python3
"""Materialize an auditable V5-compatible config from a V6 cell definition."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pull-weight", type=float, default=0.75)
    parser.add_argument("--push-weight", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--step-size", type=float, default=1 / 255)
    parser.add_argument("--attack-count", type=int, default=30)
    parser.add_argument("--objective-tag", default="pull_push_v6")
    parser.add_argument("--state-namespace", default="states_v6")
    args = parser.parse_args()
    if abs(args.pull_weight + args.push_weight - 1) > 1e-8:
        raise ValueError("Pull and Push must sum to one")
    template = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    template.update({
        "seed": 42,
        "attack_count": args.attack_count,
        "reference_count": 48,
        "candidate_count": 64,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "step_size": args.step_size,
        "momentum": 1.0,
        "random_start": True,
        "epsilon": 16 / 255,
        "generative_precision": "bf16",
        "lambda_cls": 0,
        "lambda_cka": 1,
        "state_namespace": args.state_namespace,
        "summary_filename": "v6_results.csv",
        "objective_tag": args.objective_tag,
        "semantic_mode": "linear_pull_push",
        "semantic_temperature": 0.1,
        "allow_reference_bank_smaller_than_attack": False,
    })
    spec = dict(template["pairs"][0])
    spec.update({
        "representation_type": "vision_encoder",
        "representation_layer": args.layer,
        "pooling": "mean",
        "selected_rho": 0,
        "lambda_cls": 0,
        "lambda_cka": 1,
        "semantic_mode": "linear_pull_push",
        "semantic_target_weight": 1.0,
        "semantic_source_weight": args.push_weight,
        "target_logit_weight": args.pull_weight,
        "source_logit_weight": args.push_weight,
        "steps": args.steps,
        "step_size": args.step_size,
    })
    spec.pop("representation_layer_percent", None)
    template["pairs"] = [spec]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
