#!/usr/bin/env python3
"""Use the V5 deterministic clean screen with a V6-pinned model pair."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V5_SRC = REPO / "experiments/2026-09-a100-final-v5/src"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy-repo", required=True)
    parser.add_argument("--target-repo", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(V5_SRC))
    import primary_ml_cka.domain.identifiers as ids
    from primary_ml_cka.domain.identifiers import ExperimentType, ModelPair

    lock = json.loads((ROOT / "model_revisions.json").read_text())
    by_repo = {item["repo_id"]: item for item in lock.values()}
    for repo in (args.proxy_repo, args.target_repo):
        if repo not in by_repo:
            raise ValueError(f"Model is not pinned in V6 lock: {repo}")
        ids.MODEL_REVISIONS[repo] = by_repo[repo]["revision"]
    relation = ExperimentType.INTRA_FAMILY if args.proxy_repo.split("/")[0] == args.target_repo.split("/")[0] else ExperimentType.CROSS_FAMILY
    ids.MODEL_PAIRS = tuple(item for item in ids.MODEL_PAIRS if item.pair_id != "P02") + (
        ModelPair("P02", relation, args.proxy_repo, args.target_repo),
    )
    sys.argv = [
        str(V5_SRC / "screen_transitions.py"),
        "--config", str(args.config.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--resume",
    ]
    runpy.run_path(str(V5_SRC / "screen_transitions.py"), run_name="__main__")


if __name__ == "__main__":
    main()
