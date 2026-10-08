#!/usr/bin/env python3
"""Materialize a V5-compatible isolated workspace for one V6 cell."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--transitions", nargs="+", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    output = REPO / "outputs/a100_precommercial_v6"
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    canonical = workspace / "canonical_images"
    source = output / "canonical_images"
    if canonical.is_symlink() or canonical.exists():
        if not canonical.is_symlink() or canonical.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace noncanonical workspace path: {canonical}")
    else:
        canonical.symlink_to(source, target_is_directory=True)
    references = workspace / "evaluation/manifests"
    references.mkdir(parents=True, exist_ok=True)
    for label in range(1, 11):
        ref_src = output / "evaluation/manifests" / f"class_references_{label:02d}.jsonl"
        shutil.copy2(ref_src, references / ref_src.name)
    cohort = output / "manifests/condition_cohorts" / safe(args.condition)
    for transition in args.transitions:
        manifest = cohort / f"{transition}.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(rows) != 30:
            raise RuntimeError(f"{args.condition}/{transition}: expected 30 screened images, found {len(rows)}")
        transition_dir = workspace / "evaluation/manifests/transitions" / transition
        transition_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, transition_dir / "P02_attack_images.jsonl")
    print(f"workspace_ready={workspace} condition={args.condition}", flush=True)


if __name__ == "__main__":
    main()
