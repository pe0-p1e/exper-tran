#!/usr/bin/env python3
"""Run memory-budgeted pair jobs concurrently in attack-only/evaluation-only waves."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import load_experiment, pair_specs

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, get_pair
from primary_ml_cka.models.common.loading import local_snapshot

CALIBRATION_PAIR = {
    "P02": "P02",
    "P14": "P14",
    "P16": "P16",
    "P19": "P19",
    "P20": "P02",
    "P21": "P21",
    "P22": "P22",
    "P23": "P14",
}


def calibration_rows(output_dir: Path, pair_id: str) -> list[dict[str, str]]:
    calibration_id = CALIBRATION_PAIR[pair_id]
    report = output_dir / "calibration" / f"{calibration_id}_T01_batch_sizes.csv"
    with report.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calibrated_peak(output_dir: Path, pair_id: str) -> float:
    rows = calibration_rows(output_dir, pair_id)
    usable = [
        row
        for row in rows
        if row["status"] == "ok" and float(row["peak_reserved_gb"]) <= 75
    ]
    if not usable:
        calibration_id = CALIBRATION_PAIR[pair_id]
        raise RuntimeError(
            f"No successful batch calibration for {pair_id}: "
            f"{output_dir / 'calibration' / f'{calibration_id}_T01_batch_sizes.csv'}"
        )
    selected = max(usable, key=lambda row: int(row["batch_size"]))
    return float(selected["peak_reserved_gb"])


def calibrated_batch(output_dir: Path, pair_id: str, cap: int) -> int:
    rows = calibration_rows(output_dir, pair_id)
    usable = [
        int(row["batch_size"])
        for row in rows
        if row["status"] == "ok" and float(row["peak_reserved_gb"]) <= 75
    ]
    if not usable:
        raise RuntimeError(f"No successful batch calibration for {pair_id}")
    return min(max(usable), cap)


def memory_estimate(
    pair_id: str,
    stage: str,
    output_dir: Path,
    hf_home: Path,
    reserve_gb: float,
) -> tuple[float, str]:
    pair = get_pair(pair_id)
    model_id = pair.proxy_model if stage == "attack" else pair.target_model
    if stage == "attack":
        return calibrated_peak(output_dir, pair_id) + reserve_gb, model_id
    snapshot = local_snapshot(hf_home, model_id, MODEL_REVISIONS[model_id])
    weight_bytes = sum(path.stat().st_size for path in snapshot.glob("*.safetensors"))
    # BF16 target weights plus a conservative generation workspace reservation.
    return weight_bytes / 2**30 + reserve_gb, model_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("attack", "evaluate"), required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--max-batch-size", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--gpu-memory-budget-gb", type=float, default=70.0)
    parser.add_argument("--memory-reserve-gb", type=float, default=6.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hf-home", type=Path, default=Path(".hf-cache"))
    args = parser.parse_args()
    raw = load_experiment(args.config)
    specs = pair_specs(raw)
    if set(args.pairs) - set(specs):
        raise ValueError("Unknown pair requested")
    if args.max_workers < 1 or args.gpu_memory_budget_gb <= 0:
        raise ValueError("Worker count and memory budget must be positive")

    runner = Path(__file__).with_name("run_a100_final.py")
    worker_logs = args.output_dir / "logs" / "queue" / "workers"
    worker_logs.mkdir(parents=True, exist_ok=True)
    pending = list(args.pairs)
    running: dict[str, tuple[subprocess.Popen, object, float, str]] = {}
    failures = []

    while pending or running:
        for pair_id in tuple(pending):
            if len(running) >= args.max_workers:
                break
            estimate, model_id = memory_estimate(
                pair_id,
                args.stage,
                args.output_dir,
                args.hf_home,
                args.memory_reserve_gb,
            )
            active_memory = sum(item[2] for item in running.values())
            active_models = {item[3] for item in running.values()}
            if model_id in active_models or active_memory + estimate > args.gpu_memory_budget_gb:
                continue
            batch_size = calibrated_batch(
                args.output_dir,
                pair_id,
                min(args.max_batch_size, int(raw.get("attack_count", 50))),
            )
            command = [
                sys.executable,
                str(runner),
                "--config",
                str(args.config),
                "--output-dir",
                str(args.output_dir),
                "--pairs",
                pair_id,
                "--batch-size",
                str(batch_size),
                "--resume",
                "--attack-only" if args.stage == "attack" else "--evaluate-only",
            ]
            if args.smoke:
                command.append("--smoke")
            log_path = worker_logs / f"{args.config.stem}_{args.stage}_{pair_id}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            log_handle.write(
                f"job_start stage={args.stage} pair={pair_id} model={model_id} "
                f"memory_estimate_gb={estimate:.2f} batch={batch_size}\n"
            )
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
            running[pair_id] = (process, log_handle, estimate, model_id)
            pending.remove(pair_id)
            print(
                f"job_start stage={args.stage} pair={pair_id} pid={process.pid} "
                f"estimated_gb={estimate:.2f} active={len(running)}",
                flush=True,
            )

        completed = []
        for pair_id, (process, log_handle, _estimate, _model_id) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.write(f"job_exit stage={args.stage} pair={pair_id} code={return_code}\n")
            log_handle.close()
            completed.append(pair_id)
            print(f"job_exit stage={args.stage} pair={pair_id} code={return_code}", flush=True)
            if return_code:
                failures.append((pair_id, return_code))
        for pair_id in completed:
            running.pop(pair_id)
        if failures:
            for process, log_handle, _estimate, _model_id in running.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                log_handle.close()
            raise RuntimeError(f"Pair jobs failed: {failures}")
        if pending or running:
            time.sleep(1)

    if args.stage == "evaluate":
        from run_a100_final import summarize

        summarize(args.output_dir, raw)
    print(f"wave_done stage={args.stage} pairs={','.join(args.pairs)}", flush=True)


if __name__ == "__main__":
    main()
