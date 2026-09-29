#!/usr/bin/env python3
"""Measure actual saved-PNG perturbations for every frozen attack image."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/a100_final_v5"
INDEX = OUTPUT / "shareable/priority3/all_available_measurements/full90_all_images.csv"
DEST = OUTPUT / "shareable/priority3/all_available_measurements/png_perturbations.csv"
FIELDS = (
    "pair_id", "transition_id", "image_index", "source_image_id", "proxy_target_hit",
    "target_evaluated", "target_hit", "linf_normalized", "mean_abs_normalized",
    "rms_normalized", "fraction_changed_channels", "fraction_changed_pixels",
    "clean_png", "adversarial_png",
)


def measure(row: dict) -> dict:
    clean_path = OUTPUT / row["clean_png"]
    adv_path = OUTPUT / row["adversarial_png"]
    with Image.open(clean_path) as image:
        clean = np.asarray(image.convert("RGB"), dtype=np.int16)
    with Image.open(adv_path) as image:
        adversarial = np.asarray(image.convert("RGB"), dtype=np.int16)
    if clean.shape != adversarial.shape:
        raise ValueError(f"PNG shape differs: {clean_path} vs {adv_path}")
    difference = np.abs(adversarial - clean)
    return {
        **{key: row.get(key) for key in FIELDS[:7]},
        "linf_normalized": float(difference.max()) / 255,
        "mean_abs_normalized": float(difference.mean()) / 255,
        "rms_normalized": float(np.sqrt(np.mean(difference.astype(np.float32) ** 2))) / 255,
        "fraction_changed_channels": float(np.mean(difference > 0)),
        "fraction_changed_pixels": float(np.mean(np.any(difference > 0, axis=-1))),
        "clean_png": row["clean_png"],
        "adversarial_png": row["adversarial_png"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        with INDEX.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        temp = DEST.with_suffix(".csv.tmp")
        with ThreadPoolExecutor(max_workers=args.threads) as pool, temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for index, result in enumerate(pool.map(measure, rows), 1):
                writer.writerow(result)
                if index % 2500 == 0:
                    handle.flush()
                    print(f"measured_png_pairs={index}/{len(rows)}", flush=True)
        temp.replace(DEST)
        print(f"wrote={DEST} rows={len(rows)}", flush=True)
    aggregate = defaultdict(lambda: {"N": 0, "over_budget": 0, "max_linf": 0.0, "sum_mean_abs": 0.0, "sum_rms": 0.0, "sum_changed_pixels": 0.0})
    with DEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = aggregate[row["pair_id"]]
            item["N"] += 1
            linf = float(row["linf_normalized"])
            item["over_budget"] += int(linf > 16 / 255 + 1e-8)
            item["max_linf"] = max(item["max_linf"], linf)
            item["sum_mean_abs"] += float(row["mean_abs_normalized"])
            item["sum_rms"] += float(row["rms_normalized"])
            item["sum_changed_pixels"] += float(row["fraction_changed_pixels"])
    summary = DEST.with_name("png_perturbation_summary.csv")
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "N", "over_budget", "max_linf", "mean_abs", "mean_rms", "mean_fraction_changed_pixels"])
        writer.writeheader()
        for pair, item in sorted(aggregate.items()):
            writer.writerow({
                "pair_id": pair,
                "N": item["N"],
                "over_budget": item["over_budget"],
                "max_linf": item["max_linf"],
                "mean_abs": item["sum_mean_abs"] / item["N"],
                "mean_rms": item["sum_rms"] / item["N"],
                "mean_fraction_changed_pixels": item["sum_changed_pixels"] / item["N"],
            })
    print(f"wrote={summary} pairs={len(aggregate)} images={sum(x['N'] for x in aggregate.values())}", flush=True)


if __name__ == "__main__":
    main()
