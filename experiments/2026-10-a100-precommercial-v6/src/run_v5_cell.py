#!/usr/bin/env python3
"""Run a V5 single-proxy cell using a V6-pinned, runtime-injected model pair."""

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
    parser.add_argument("--transitions", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attack-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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
    families = {args.proxy_repo.split("/")[0], args.target_repo.split("/")[0]}
    relation = ExperimentType.INTRA_FAMILY if args.proxy_repo.split("/")[0] == args.target_repo.split("/")[0] else ExperimentType.CROSS_FAMILY
    v6_pair = ModelPair("P02", relation, args.proxy_repo, args.target_repo)
    ids.MODEL_PAIRS = tuple(item for item in ids.MODEL_PAIRS if item.pair_id != "P02") + (v6_pair,)
    sys.argv = [
        str(V5_SRC / "run_a100_final.py"),
        "--config", str(args.config.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--pairs", "P02",
        "--transitions", *args.transitions,
        "--resume",
        "--batch-size", str(args.batch_size),
    ]
    if args.attack_only:
        sys.argv.append("--attack-only")
    if args.evaluate_only:
        sys.argv.append("--evaluate-only")
    if args.smoke:
        sys.argv.append("--smoke")
    runpy.run_path(str(V5_SRC / "run_a100_final.py"), run_name="__main__")


if __name__ == "__main__":
    main()
