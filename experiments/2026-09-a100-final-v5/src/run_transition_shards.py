#!/usr/bin/env python3
"""Resume one pair's unfinished attack transitions on up to four disjoint workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import load_experiment, transitions


def free_gpu_memory_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--gpu-memory-budget-gb", type=float, default=70.0)
    parser.add_argument("--memory-reserve-gb", type=float, default=6.0)
    parser.add_argument("--hf-home", type=Path, default=Path(".hf-cache"))
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 4:
        raise ValueError("This runner intentionally caps attack concurrency at four workers")

    raw = load_experiment(args.config)
    state_root = args.output_dir / str(raw.get("state_namespace", "states_a100_v5")) / args.pair
    pending = []
    for transition in transitions(raw):
        state_path = state_root / transition.transition_id / "batch_00.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if state.get("status") not in {"attack_complete", "complete"}:
            pending.append(transition.transition_id)
    if not pending:
        print(f"no_pending_transitions pair={args.pair}", flush=True)
        return

    calibration_path = args.output_dir / "calibration" / f"{args.pair}_T01_batch_sizes.csv"
    with calibration_path.open(newline="", encoding="utf-8") as handle:
        calibration = list(csv.DictReader(handle))
    usable = [
        row for row in calibration
        if row["status"] == "ok"
        and int(row["batch_size"]) <= min(50, int(raw["attack_count"]))
        and float(row["peak_reserved_gb"]) <= 75
    ]
    if not usable:
        raise RuntimeError(f"No safe calibrated batch size: {calibration_path}")
    selected = max(usable, key=lambda row: int(row["batch_size"]))
    batch_size = int(selected["batch_size"])
    per_worker_peak = float(selected["peak_reserved_gb"])
    planned = min(args.max_workers, len(pending))
    estimated_peak = planned * per_worker_peak + args.memory_reserve_gb
    if estimated_peak > args.gpu_memory_budget_gb:
        raise RuntimeError(
            f"Refusing {planned} workers: calibrated peak plus reserve "
            f"is {estimated_peak:.2f} GiB > {args.gpu_memory_budget_gb:.2f} GiB"
        )

    experiment = Path(__file__).resolve().parents[1]
    runner = experiment / "src/run_a100_final.py"
    log_dir = args.output_dir / "logs" / "queue" / "transition_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    shards = [pending[index::planned] for index in range(planned)]
    processes = []
    for worker, shard in enumerate(shards, 1):
        command = [
            sys.executable, str(runner),
            "--config", str(args.config), "--output-dir", str(args.output_dir),
            "--pairs", args.pair, "--transitions", *shard,
            "--batch-size", str(batch_size), "--resume", "--attack-only",
        ]
        log_path = log_dir / f"{args.config.stem}_{args.pair}_worker{worker}.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(
                f"worker_start pair={args.pair} worker={worker}/{planned} "
                f"batch={batch_size} calibrated_peak_gb={per_worker_peak:.2f} "
                f"reserved_plan_gb={estimated_peak:.2f} transitions={','.join(shard)}\n"
            )
        log = log_path.open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
        processes.append((process, log, log_path, worker))
        print(
            f"worker_start pair={args.pair} worker={worker}/{planned} pid={process.pid} "
            f"transitions={','.join(shard)} calibrated_peak_gb={per_worker_peak:.2f} "
            f"planned_gb={estimated_peak:.2f}",
            flush=True,
        )
        # Stagger model loading and verify the measured allocation before adding workers.
        time.sleep(8)
        measured = free_gpu_memory_mib() / 1024
        print(f"worker_memory_check active={worker} gpu_used_gb={measured:.2f}", flush=True)
        if measured > args.gpu_memory_budget_gb:
            raise RuntimeError(
                f"GPU usage {measured:.2f} GiB exceeds budget; stopping newly launched worker"
            )

    failures = []
    try:
        while processes:
            for item in tuple(processes):
                process, log, log_path, worker = item
                code = process.poll()
                if code is None:
                    continue
                log.write(f"worker_exit worker={worker} code={code}\n")
                log.close()
                processes.remove(item)
                print(f"worker_exit worker={worker} code={code}", flush=True)
                if code:
                    failures.append((worker, code, str(log_path)))
            if failures:
                for process, log, _log_path, _worker in processes:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    log.close()
                raise RuntimeError(f"Transition workers failed: {failures}")
            if processes:
                measured = free_gpu_memory_mib() / 1024
                if measured > args.gpu_memory_budget_gb:
                    for process, _log, _path, _worker in processes:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    raise RuntimeError(
                        f"GPU usage {measured:.2f} GiB exceeded budget "
                        f"{args.gpu_memory_budget_gb:.2f} GiB"
                    )
                time.sleep(5)
    except KeyboardInterrupt:
        for process, log, _path, _worker in processes:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            log.close()
        raise
    print(f"shards_done pair={args.pair} transitions={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
