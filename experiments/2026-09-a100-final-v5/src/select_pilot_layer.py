#!/usr/bin/env python3
"""Select a layer from the completed 5-layer x 10-transition pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    args = parser.parse_args()

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_pairs = {str(item["pair_id"]) for item in base["pairs"]}
    expected_transitions = {str(item["id"]) for item in base["transitions"]}
    scores = {}
    for percent in (1, 25, 50, 75, 100):
        path = args.summary_dir / f"layer_{percent:03d}_results.csv"
        rows = read_rows(path)
        observed = {(row["pair_id"], row["transition_id"]) for row in rows}
        expected = {
            (pair_id, transition_id)
            for pair_id in expected_pairs
            for transition_id in expected_transitions
        }
        if observed != expected or len(rows) != len(expected):
            raise RuntimeError(
                f"Pilot layer {percent}% is incomplete: {len(rows)}/{len(expected)} rows in {path}"
            )
        hits = sum(int(row["tasr_hits"]) for row in rows)
        denominator = sum(int(row["images"]) for row in rows)
        if denominator != len(rows) * int(base["attack_count"]):
            raise RuntimeError(f"Unexpected clean-valid denominator in {path}")
        scores[percent] = {
            "tasr_hits": hits,
            "clean_valid_images": denominator,
            "tasr": hits / denominator,
        }

    selected = max(scores, key=lambda percent: (scores[percent]["tasr"], -percent))
    config = {**base, "pairs": [dict(pair) for pair in base["pairs"]]}
    config["catalog_mode"] = "full_directed_pairs"
    # Reuse the pilot's completed attack states for the chosen depth. The first
    # ten transition IDs therefore remain identical to the pilot config.
    config["state_namespace"] = f"states_layer_{selected:03d}"
    config["summary_filename"] = "single_proxy_full90.csv"
    config["objective_tag"] = f"layer_{selected:03d}"
    config["pilot_selection_layer_percent"] = selected
    config["pilot_selection_metric"] = "micro TASR over clean-valid images in 10 directed pilot transitions"
    pilot_edges = [
        (int(item["source"]), int(item["target"]), str(item["id"]))
        for item in base["transitions"]
    ]
    pilot_edge_set = {(source, target) for source, target, _ in pilot_edges}
    remaining_edges = [
        (source, target)
        for source in range(1, 11)
        for target in range(1, 11)
        if source != target and (source, target) not in pilot_edge_set
    ]
    config["transitions"] = [
        {"id": transition_id, "source": source, "target": target}
        for source, target, transition_id in pilot_edges
    ] + [
        {"id": f"T{index:02d}", "source": source, "target": target}
        for index, (source, target) in enumerate(remaining_edges, start=11)
    ]
    config["pilot_transition_ids"] = [transition_id for _, _, transition_id in pilot_edges]
    for pair in config["pairs"]:
        pair["representation_layer_percent"] = selected
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "selection_metric": config["pilot_selection_metric"],
        "pilot_transitions": sorted(expected_transitions),
        "selected_layer_percent": selected,
        "layer_scores": scores,
        "full_directed_transition_count": len(config["transitions"]),
        "confirmatory_nonpilot_transition_count": 80,
        "pilot_transition_ids_reused_for_resume": [transition_id for _, _, transition_id in pilot_edges],
        "interpretation": "Pilot transitions informed layer selection; report the other 80 transitions separately as held-out transfer evaluation.",
    }
    args.selection_report.parent.mkdir(parents=True, exist_ok=True)
    args.selection_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"selected_layer={selected}% pilot_tasr={scores[selected]['tasr']:.6f} "
        f"wrote={args.output_config}",
        flush=True,
    )


if __name__ == "__main__":
    main()
