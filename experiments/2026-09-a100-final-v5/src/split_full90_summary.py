#!/usr/bin/env python3
"""Split the selected-layer 90-transition result into pilot and held-out sets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pilot_ids = set(config["pilot_transition_ids"])
    pair_ids = {str(item["pair_id"]) for item in config["pairs"]}
    transition_ids = {str(item["id"]) for item in config["transitions"]}
    with args.summary.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    expected = {(pair_id, transition_id) for pair_id in pair_ids for transition_id in transition_ids}
    observed = {(row["pair_id"], row["transition_id"]) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError(f"Full90 summary incomplete: observed {len(rows)} of {len(expected)} cells")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, predicate in (
        ("pilot", lambda row: row["transition_id"] in pilot_ids),
        ("heldout80", lambda row: row["transition_id"] not in pilot_ids),
    ):
        subset = [dict(row, selection_set=label) for row in rows if predicate(row)]
        out = args.output_dir / f"{args.summary.stem}_{label}.csv"
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames + ["selection_set"])
            writer.writeheader()
            writer.writerows(subset)
        print(f"wrote {len(subset)} cells to {out}", flush=True)


if __name__ == "__main__":
    main()
