#!/usr/bin/env python3
"""Run full-90 embedding precompute beside evaluation, keeping four GPU jobs or fewer."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/a100_final_v5"
CONFIG = EXPERIMENT / "config/full90_selected_layer_priority3.yaml"
RUNNER = EXPERIMENT / "src/analyze_joint_geometry.py"
PAIRS = ("P14", "P02", "P23")
WEIGHT_GB = {"P14": 8.68, "P02": 14.89, "P23": 8.82}
ANALYSIS = OUTPUT / "analysis/single_proxy_full90_priority3"
PLOT_IDS = tuple(
    item["id"]
    for item in yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["transitions"]
    if int(item["source"]) < int(item["target"])
)


def gpu_workers() -> list[tuple[int, str]]:
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if not parts or "python" not in Path(parts[0].decode(errors="ignore")).name:
            continue
        command = " ".join(part.decode(errors="ignore") for part in parts)
        if "run_a100_final.py" in command or "analyze_joint_geometry.py" in command:
            workers.append((int(entry.name), command))
    return workers


def gpu_memory_gb() -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(output.splitlines()[0].strip()) / 1024


def plots_done(pair: str) -> bool:
    return len(PLOT_IDS) == 45 and all(
        (ANALYSIS / pair / kind / f"{pair}_{transition_id}.png").is_file()
        for transition_id in PLOT_IDS
        for kind in ("pca", "tsne")
    )


def main() -> None:
    log_dir = OUTPUT / "logs/priority3_precompute"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    attempts = {pair: 0 for pair in PAIRS}
    while True:
        for pair, (process, log) in tuple(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.write(f"precompute_exit pair={pair} code={code}\n")
            log.close()
            running.pop(pair)
            if code or not plots_done(pair):
                print(f"precompute_failed pair={pair} code={code}", flush=True)
            else:
                print(f"precompute_done pair={pair}", flush=True)
        if all(plots_done(pair) for pair in PAIRS):
            print("priority3_precompute_done pairs=P14,P02,P23", flush=True)
            return

        workers = gpu_workers()
        occupied_pairs = {
            pair
            for pair in PAIRS
            if any("analyze_joint_geometry.py" in command and f"--pair {pair}" in command for _, command in workers)
        }
        for pair in PAIRS:
            if plots_done(pair) or pair in occupied_pairs:
                continue
            if len(workers) >= 4:
                break
            if gpu_memory_gb() + WEIGHT_GB[pair] + 6 > 70:
                continue
            if attempts[pair] >= 2:
                raise RuntimeError(f"Precompute failed twice for {pair}")
            attempts[pair] += 1
            log = (log_dir / f"{pair}.log").open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable, str(RUNNER),
                "--config", str(CONFIG), "--output-dir", str(OUTPUT),
                "--pair", pair, "--batch-size", "50", "--plot-transition", "T01",
                "--analysis-subdir", f"single_proxy_full90_priority3/{pair}",
                "--resume", "--precompute-only",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            running[pair] = (process, log)
            print(f"precompute_start pair={pair} pid={process.pid}", flush=True)
            time.sleep(10)
            workers = gpu_workers()
        time.sleep(20)


if __name__ == "__main__":
    main()
