#!/usr/bin/env python3
"""Audit pinned proxy/target vision depths and percentage-to-block mapping."""

import argparse
import csv
from pathlib import Path

from common import layer_from_percent, load_experiment, pair_specs, vision_layer_count

from primary_ml_cka.domain.identifiers import get_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/a100_final.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/a100_final_v5/analysis/model_layer_audit.csv")
    )
    parser.add_argument("--percentages", nargs="+", type=float, default=[1, 25, 50, 75, 100])
    args = parser.parse_args()
    raw = load_experiment(args.config)
    model_ids = set()
    for pair_id in pair_specs(raw):
        pair = get_pair(pair_id)
        model_ids.update((pair.proxy_model, pair.target_model))
    rows = []
    for model_id in sorted(model_ids):
        count = vision_layer_count(model_id)
        for percentage in args.percentages:
            rows.append(
                {
                    "model_id": model_id,
                    "vision_layers": count,
                    "depth_percent": percentage,
                    "resolved_block_index": layer_from_percent(model_id, percentage),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} model/layer rows to {args.output}")


if __name__ == "__main__":
    main()
