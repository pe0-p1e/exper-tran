#!/usr/bin/env python3
"""Copy real frozen clean/adversarial PNG pairs from the common T01 cohort."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/a100_final_v5"
DESTINATION = OUTPUT / "shareable/priority3/examples"
PAIR_IDS = ("P02", "P14", "P23")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_id in PAIR_IDS:
        state = json.loads(
            (OUTPUT / "states_layer_100" / pair_id / "T01/batch_00.json").read_text()
        )
        if state.get("status") != "complete":
            raise RuntimeError(f"T01 target evaluation is incomplete for {pair_id}")
        mask = [bool(value) for value in state["target"]["target_hit_mask"]]
        proxy_mask = [bool(value) for value in state["attack"]["proxy_target_hit_mask"]]
        if len(mask) != len(proxy_mask):
            raise RuntimeError(f"Proxy/target masks differ for {pair_id}/T01")
        image_root = (
            OUTPUT / "attacks" / pair_id / "a100_v5_T01/batch_00"
            / state["objective_tag"] / f"lambda_{float(state['lambda_cka']):g}"
        )
        for outcome, desired in (("hit", True), ("miss", False)):
            indices = [
                index for index, value in enumerate(mask)
                if value is desired and proxy_mask[index]
            ]
            if not indices:
                continue
            index = indices[0]
            names = {}
            for kind in ("clean", "adv"):
                source = image_root / f"{index:02d}_{kind}.png"
                if not source.is_file():
                    raise FileNotFoundError(source)
                name = f"{pair_id}_T01_{index:02d}_{outcome}_{kind}.png"
                shutil.copy2(source, DESTINATION / name)
                names[kind] = name
            outputs = state["target"]
            rows.append(
                {
                    "pair_id": pair_id,
                    "transition_id": "T01",
                    "source_class": "goldfish",
                    "target_class": "monarch butterfly",
                    "image_index": index,
                    "target_hit": desired,
                    "proxy_target_hit": True,
                    "clean_parsed_label": outputs["clean_outputs"][index].get("parsed_label"),
                    "adversarial_parsed_label": outputs["adversarial_outputs"][index].get(
                        "parsed_label"
                    ),
                    "clean_png": names["clean"],
                    "adversarial_png": names["adv"],
                }
            )
    with (DESTINATION / "examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Real attack PNG examples: goldfish → monarch butterfly",
        "",
        "Each row uses the frozen clean and adversarial PNGs from the experiment. "
        "Every shown image succeeded on the proxy; target hit/miss is the evaluated target model's parsed class outcome.",
        "",
        "| Pair | Outcome | Clean | Adversarial |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_id']} | {row['target_hit']} | "
            f"![clean]({row['clean_png']}) | ![adv]({row['adversarial_png']}) |"
        )
    (DESTINATION / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"examples={len(rows)} files={2 * len(rows)} destination={DESTINATION}")


if __name__ == "__main__":
    main()
