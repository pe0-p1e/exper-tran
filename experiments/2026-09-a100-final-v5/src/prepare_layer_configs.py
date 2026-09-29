#!/usr/bin/env python3
"""Create reproducible layer-depth configs for the A100 embedding sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/a100_final.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/2026-09-a100-final-v5/config/layers")
    )
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for percent in (1, 25, 50, 75, 100):
        config = {**base, "pairs": [dict(pair) for pair in base["pairs"]]}
        config["state_namespace"] = f"states_layer_{percent:03d}"
        config["summary_filename"] = f"layer_{percent:03d}_results.csv"
        config["objective_tag"] = f"layer_{percent:03d}"
        for pair in config["pairs"]:
            pair.pop("representation_layer", None)
            pair["representation_layer_percent"] = percent
        path = args.output_dir / f"layer_{percent:03d}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
