#!/usr/bin/env python3
"""Download only the pinned model snapshots referenced by this run config."""

import argparse
import os
from pathlib import Path

from common import load_experiment, pair_specs
from huggingface_hub import snapshot_download

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, get_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/a100_final.yaml",
    )
    parser.add_argument(
        "--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", ".hf-cache"))
    )
    args = parser.parse_args()
    raw = load_experiment(args.config)
    model_ids = set()
    for pair_id in pair_specs(raw):
        pair = get_pair(pair_id)
        model_ids.update((pair.proxy_model, pair.target_model))
    for model_id in sorted(model_ids):
        revision = MODEL_REVISIONS[model_id]
        location = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=args.hf_home,
        )
        print(f"ready model={model_id} revision={revision} snapshot={location}", flush=True)


if __name__ == "__main__":
    main()
