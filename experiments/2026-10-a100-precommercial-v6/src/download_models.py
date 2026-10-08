#!/usr/bin/env python3
"""Download only the immutable checkpoints declared by model_revisions.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Experiment-facing names; default is all")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    cache = (args.cache or Path(os.environ.get("HF_HOME", "/ephemeral/a100_precommercial_v6/model-cache/hf-cache"))).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    registry = json.loads((ROOT / "model_revisions.json").read_text())
    selected = args.models or list(registry)
    unknown = set(selected) - set(registry)
    if unknown:
        raise SystemExit(f"Unknown model names: {sorted(unknown)}")
    records = []
    for name in selected:
        item = registry[name]
        print(f"[download] {name} {item['repo_id']}@{item['revision']}", flush=True)
        path = snapshot_download(
            item["repo_id"],
            revision=item["revision"],
            cache_dir=str(cache / "hub"),
            max_workers=4,
        )
        records.append({"name": name, "repo_id": item["repo_id"], "revision": item["revision"], "snapshot": path})
        print(f"[download] complete {name}: {path}", flush=True)
    audit = ROOT.parent.parent / "outputs/a100_precommercial_v6/audits/model_snapshots.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
