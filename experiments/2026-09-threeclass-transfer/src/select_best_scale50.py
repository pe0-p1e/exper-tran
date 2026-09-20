#!/usr/bin/env python3
"""Select the best 8-image arm per pair and materialize the 50-image config."""

import argparse
import glob
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = []
    for path in glob.glob(str(args.states / "P*" / "G*" / "*.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("status") != "complete":
            continue
        if data["pair_id"] not in {"P14", "P16", "P19"}:
            continue
        payloads.append(data)
    chosen = {}
    for pair_id in ("P14", "P16", "P19"):
        candidates = {}
        prefix = f"{pair_id.lower()}_"
        for data in payloads:
            arm = str(data["arm"])
            if pair_id in {"P14", "P19"} and not arm.startswith("p14_p19_"):
                continue
            if pair_id == "P16" and arm != "p16_second_target_pull_half":
                continue
            item = candidates.setdefault(arm, {"tasr": 0, "proxy": 0, "asr": 0, "example": data})
            item["tasr"] += int(data["tasr_hits"])
            item["proxy"] += int(data["attack"]["proxy_target_hit_count"])
            item["asr"] += int(data["asr_hits"])
        if not candidates:
            raise RuntimeError(f"No completed tuning arm found for {pair_id}")
        arm, score = max(
            candidates.items(), key=lambda item: (item[1]["tasr"], item[1]["proxy"], item[1]["asr"])
        )
        chosen[pair_id] = (arm, score, score["example"])
        print(pair_id, arm, score, flush=True)

    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    by_pair = {str(item["pair_id"]): item for item in config["pairs"]}
    for pair_id, (arm, _, example) in chosen.items():
        spec = by_pair[pair_id]
        for key in (
            "lambda_cls", "lambda_cka", "rho", "target_logit_weight",
            "source_logit_weight", "steps", "step_size", "semantic_mode",
        ):
            source_key = {"rho": "rho"}.get(key, key)
            if key in example:
                spec[key if key != "rho" else "selected_rho"] = example[source_key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
