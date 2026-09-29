#!/usr/bin/env python3
"""Create resumable pull-only, push-only, and combined loss configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


WEIGHTS = {
    "pull_only": (1.5, 0.0),
    "push_only": (0.0, 0.5),
    "pull_push": (1.5, 0.5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(WEIGHTS), required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target_weight, source_weight = WEIGHTS[args.arm]
    raw["state_namespace"] = f"states_a100_v5_ablation_{args.arm}"
    raw["summary_filename"] = f"ablation_{args.arm}.csv"
    raw["objective_tag"] = f"a100_v5_ablation_{args.arm}"
    raw["ablation_arm"] = args.arm
    for spec in raw["pairs"]:
        # These are the actual coefficients consumed by linear_pull_push.
        # The outer semantic_target_weight remains one for every arm.
        spec["semantic_mode"] = "linear_pull_push"
        spec["lambda_cls"] = 0.0
        spec["lambda_cka"] = 1.0
        spec["target_logit_weight"] = target_weight
        spec["source_logit_weight"] = source_weight
        spec["semantic_target_weight"] = target_weight
        spec["semantic_source_weight"] = source_weight

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(
        f"wrote={args.output} arm={args.arm} "
        f"pull={target_weight:g} push={source_weight:g}"
    )


if __name__ == "__main__":
    main()
