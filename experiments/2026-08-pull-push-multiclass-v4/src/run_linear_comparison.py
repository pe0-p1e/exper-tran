#!/usr/bin/env python3
"""Search explicit cosine pull/push on reserve8, then compare frozen recipes."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from common import EXPERIMENT_ROOT
from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.loader import load_config


def make_arms(plan, pair):
    spec = plan["pairs"][pair]
    shared = dict(steps=plan["steps"], step_size=plan["step_size"],
                  semantic_temperature=spec["tau"], target_logit_weight=1.0)
    arms = [dict(shared, name="cls_only", semantic_mode="linear_pull_push",
                 rho=0., source_logit_weight=0.),
            dict(shared, name="infonce_baseline", semantic_mode="prototype",
                 rho=spec["rho"], source_logit_weight=spec["push"])]
    for multiplier in plan["rho_multipliers"]:
        for push in plan["push_weights"]:
            rho = spec["rho"] * multiplier
            arms.append(dict(shared, name=f"linear_r{rho:g}_p{push:g}",
                             semantic_mode="linear_pull_push", rho=rho,
                             source_logit_weight=push))
    return arms


def select_arm(output, pair, arms, transitions, *, pull_only=False):
    candidates = []
    for arm in arms:
        if not arm["name"].startswith("linear_"):
            continue
        if pull_only and arm["source_logit_weight"] != 0:
            continue
        if not pull_only and arm["source_logit_weight"] == 0:
            continue
        hits = 0
        for transition in transitions:
            state = json.loads((output / "states" / pair / transition /
                                f"{arm['name']}.json").read_text())
            if state["status"] != "complete" or state["clean_valid_count"] != 8:
                raise RuntimeError("Selection requires all tuning evaluations complete with N=8")
            hits += state["tasr_hits"]
        # Primary selection = TASR. Ties prefer less auxiliary and less push.
        candidates.append((-hits, arm["rho"], arm["source_logit_weight"], arm["name"], arm))
    return min(candidates)[-1]


def summarize(output, plan):
    rows = []
    tuning = set(plan["tuning_transitions"])
    for path in sorted((output / "states").glob("*/*/*.json")):
        s = json.loads(path.read_text())
        if s.get("status") != "complete":
            continue
        a = s["attack"]
        tc, ta = a["target_similarity_clean"], a["target_similarity_adversarial"]
        sc, sa = a["source_similarity_clean"], a["source_similarity_adversarial"]
        rows.append(dict(
            pair=s["pair_id"], transition=s["transition_id"], arm=s["arm"],
            split="smoke" if s["arm"].startswith("smoke_") else
                  "tuning" if s["transition_id"] in tuning else "validation",
            semantic_mode=s["semantic_mode"], rho=s["rho"],
            effective_lambda=a["effective_lambda_cka"],
            pull_weight=s["target_logit_weight"], push_weight=s["source_logit_weight"],
            steps=s["steps"], layer=s["representation_layer"],
            proxy_hits=a["proxy_target_hit_count"],
            tasr_hits=s["tasr_hits"], asr_hits=s["asr_hits"], n=s["clean_valid_count"],
            target_similarity_clean=tc, target_similarity_adv=ta,
            source_similarity_clean=sc, source_similarity_adv=sa,
            pull_loss_clean=1-tc, pull_loss_adv=1-ta,
            push_loss_clean=1+sc, push_loss_adv=1+sa,
            target_gain=ta-tc, source_drop=sc-sa, gap_gain=(ta-tc)+(sc-sa),
        ))
    if not rows:
        return
    directory = output / "summaries"
    directory.mkdir(exist_ok=True)
    with (directory / "linear_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["pair"], r["split"], r["arm"]), []).append(r)
    lines = ["# Explicit pull+push versus InfoNCE (8 images per transition)", "",
             "Partial groups are shown with their actual denominators. Selection uses only tuning cells.", "",
             "| Pair | Split | Arm | Cells | TASR | ASR | Proxy | Target gain | Source drop |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for (pair, split, arm), group in sorted(groups.items()):
        n = sum(r["n"] for r in group)
        values = [sum(r[k] for r in group) for k in ("tasr_hits", "asr_hits", "proxy_hits")]
        gain = sum(r["target_gain"] for r in group)/len(group)
        drop = sum(r["source_drop"] for r in group)/len(group)
        lines.append(f"| {pair} | {split} | {arm} | {len(group)} | {values[0]}/{n} | "
                     f"{values[1]}/{n} | {values[2]}/{n} | {gain:.4f} | {drop:.4f} |")
    (directory / "comparison.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=EXPERIMENT_ROOT / "config/linear_pull_push_search.yaml")
    parser.add_argument("--stage", choices=["all", "smoke", "tune", "validate", "summary"], default="all")
    args = parser.parse_args()
    plan = load_config(args.plan)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "plan.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != plan:
        raise RuntimeError("Frozen plan changed; use a new output directory")
    write_json(snapshot, plan)
    if args.stage == "summary":
        summarize(output, plan)
        return
    subprocess.run([sys.executable, str(EXPERIMENT_ROOT / "src/prepare_reserve_tuning.py"),
                    "--output-dir", str(output)], check=True)

    def run(pair, arms, transitions):
        # JSON is accepted by the YAML loader; persist the exact launched arms.
        for arm in arms:
            config = output / "configs" / pair / f"{arm['name']}.json"
            write_json(config, {"arms": [arm]})
            try:
                subprocess.run([
                    sys.executable, str(EXPERIMENT_ROOT / "src/run_compare.py"),
                    "--output-dir", str(output), "--pairs", pair,
                    "--transitions", *transitions, "--arms", arm["name"],
                    "--arm-config", str(config), "--resume", "--strict-resume", "--fail-on-error",
                ], check=True)
            finally:
                summarize(output, plan)

    if args.stage in {"all", "smoke"}:
        for pair in plan["pairs"]:
            base = make_arms(plan, pair)
            smoke = [dict(base[0], name="smoke_cls", steps=1),
                     dict(base[1], name="smoke_infonce", steps=1),
                     dict(base[-1], name="smoke_linear", steps=1)]
            run(pair, smoke, [plan["tuning_transitions"][0]])
    if args.stage in {"all", "tune", "validate"}:
        for pair in plan["pairs"]:
            arms = make_arms(plan, pair)
            if args.stage in {"all", "tune"}:
                run(pair, arms, plan["tuning_transitions"])
            if args.stage in {"all", "validate"}:
                winner = select_arm(output, pair, arms, plan["tuning_transitions"])
                pull = select_arm(output, pair, arms, plan["tuning_transitions"], pull_only=True)
                matched = dict(winner, name="infonce_matched_linear", semantic_mode="prototype")
                write_json(output / "selection" / f"{pair}.json", {
                    "selected_on": plan["tuning_transitions"], "linear": winner,
                    "pull_only": pull, "infonce_matched": matched,
                    "criterion": "TASR; ties lower rho then lower push", 
                })
                run(pair, [arms[0], arms[1], winner, pull, matched], plan["validation_transitions"])


if __name__ == "__main__":
    main()
