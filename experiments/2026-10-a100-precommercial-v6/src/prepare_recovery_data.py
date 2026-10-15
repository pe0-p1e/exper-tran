#!/usr/bin/env python3
"""Freeze one model-independent 30-image cohort per forward transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = REPO / "outputs/a100_precommercial_v6"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if digest(source) != digest(destination):
            raise RuntimeError(f"Refusing to overwrite changed recovery input: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if digest(source) != digest(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copy verification failed: {source}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/a100_precommercial_v6_multi_repr_recovery")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    matrix = yaml.safe_load((ROOT / "experiment_matrix.yaml").read_text())
    runner = yaml.safe_load((ROOT / "config/runner_template.yaml").read_text())
    recovery = yaml.safe_load((ROOT / "multiple_proxy_recovery.yaml").read_text())
    candidates_root = BASE / "evaluation/manifests/transitions"
    references_root = BASE / "evaluation/manifests"
    source_images_root = BASE / "canonical_images"
    canonical_root = out / "canonical_images"
    manifest_root = out / "manifests"
    attack_root = manifest_root / "attack_cohorts"
    references_root_out = manifest_root / "class_references"
    count = int(recovery["data"]["attack_images_per_transition"])
    reference_count = int(recovery["data"]["reference_images_per_class"])
    seed = int(recovery["data"]["seed"])
    transitions = [row for row in runner["transitions"] if int(row["source"]) < int(row["target"])]
    transitions = sorted(transitions, key=lambda row: int(row["source"]))
    if len(transitions) != 5:
        raise RuntimeError(f"Expected exactly five forward transitions, found {len(transitions)}")

    all_attack_ids: set[str] = set()
    manifest_hashes = {}
    for transition in transitions:
        tid = str(transition["id"])
        src_label, tgt_label = int(transition["source"]), int(transition["target"])
        candidate_path = candidates_root / tid / "candidates.jsonl"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Missing frozen ImageNet candidate manifest {candidate_path}")
        records = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
        records = [row for row in records if int(row["human_label"]) == src_label]
        if len(records) < count:
            raise RuntimeError(f"{tid}: only {len(records)} ground-truth source candidates; need {count}")
        for row in records:
            image_path = source_images_root / row["relative_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing canonical candidate image {image_path}")
        chosen = sorted(
            records,
            key=lambda row: hashlib.sha256(f"{seed}|{tid}|{row['image_id']}".encode()).hexdigest(),
        )[:count]
        chosen.sort(key=lambda row: row["image_id"])
        ids = {row["image_id"] for row in chosen}
        if len(ids) != count or ids & all_attack_ids:
            raise RuntimeError(f"{tid}: duplicate or incomplete GT-only image selection")
        all_attack_ids |= ids
        for row in chosen:
            atomic_copy(source_images_root / row["relative_path"], canonical_root / row["relative_path"])
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in chosen)
        manifest = attack_root / f"{tid}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        if manifest.is_file() and manifest.read_text() != payload:
            raise RuntimeError(f"Existing frozen recovery cohort differs from deterministic selection: {manifest}")
        if not manifest.is_file():
            manifest.write_text(payload)
        manifest_hashes[tid] = hashlib.sha256(payload.encode()).hexdigest()
        print(f"[recovery-data] {tid}: {len(chosen)}/{count}, sha256={manifest_hashes[tid]}", flush=True)

    # Copy and verify all 48-per-class reference records/images. They must be
    # disjoint from every image selected into the fixed attack cohorts.
    reference_ids: set[str] = set()
    for label in range(1, 11):
        source_manifest = references_root / f"class_references_{label:02d}.jsonl"
        records = [json.loads(line) for line in source_manifest.read_text().splitlines() if line.strip()]
        if len(records) != reference_count:
            raise RuntimeError(f"Class {label}: expected {reference_count} reference images, found {len(records)}")
        ids = {row["image_id"] for row in records}
        if len(ids) != reference_count or ids & all_attack_ids or ids & reference_ids:
            raise RuntimeError(f"Reference bank {label} overlaps or contains duplicate image IDs")
        reference_ids |= ids
        for row in records:
            atomic_copy(source_images_root / row["relative_path"], canonical_root / row["relative_path"])
        text = source_manifest.read_text()
        target_manifest = references_root_out / source_manifest.name
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        if target_manifest.is_file() and target_manifest.read_text() != text:
            raise RuntimeError(f"Existing recovery reference manifest differs: {target_manifest}")
        if not target_manifest.is_file():
            target_manifest.write_text(text)

    metadata = {
        "selection": "lowest SHA256(seed|transition_id|image_id), then image_id sort; ImageNet ground-truth source label only",
        "seed": seed,
        "images_per_transition": count,
        "reference_images_per_class": reference_count,
        "forward_transitions": [{"transition_id": str(t["id"]), "source_label": int(t["source"]), "target_label": int(t["target"]), "manifest_sha256": manifest_hashes[str(t["id"])]} for t in transitions],
        "attack_image_ids": sorted(all_attack_ids),
        "reference_image_ids": sorted(reference_ids),
        "attack_reference_overlap": sorted(all_attack_ids & reference_ids),
        "model_output_gating": False,
        "selection_source_manifest_sha256": {str(t["id"]): digest(candidates_root / str(t["id"]) / "candidates.jsonl") for t in transitions},
    }
    if metadata["attack_reference_overlap"]:
        raise RuntimeError("Attack/reference overlap detected")
    final_path = out / "manifests/fixed_cohort_metadata.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metadata, indent=2) + "\n"
    if final_path.is_file() and final_path.read_text() != rendered:
        raise RuntimeError("Existing fixed cohort metadata differs from deterministic cohort")
    if not final_path.is_file():
        final_path.write_text(rendered)
    print(f"[recovery-data] PASS: {len(transitions)} transitions x {count}, {len(reference_ids)} disjoint refs, model-independent", flush=True)


if __name__ == "__main__":
    main()
