#!/usr/bin/env python3
"""Run one schedulable benchmark job for a single model, epoch, and seed on a server."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppm_drift.utils.io import ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def run_cmd(cmd, logger):
    logger.info("RUN %s", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one benchmark sub-job on a server.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--work_root", default="reproducibility_run")
    ap.add_argument("--model_name", required=True, choices=["lstm", "transformer", "tft"])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    work_root = ensure_dir(args.work_root)
    logs_dir = ensure_dir(work_root / "logs")
    bench_root = work_root / "benchmarks"
    logger = setup_logger(logs_dir, filename=f"server_benchmark_job_{args.model_name}_e{args.epoch}_s{args.seed}.log")

    with timed_stage(logger, logs_dir, work_root, f"benchmark_{args.model_name}_e{args.epoch}_s{args.seed}", {"current_model": args.model_name}):
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
            "--data_root", args.data_root,
            "--output_root", str(bench_root),
            "--model_name", args.model_name,
            "--epochs", str(args.epoch),
            "--seeds", str(args.seed),
            "--single_job",
        ]
        if args.skip_existing:
            cmd.append("--skip_existing")
        run_cmd(cmd, logger)


if __name__ == "__main__":
    main()
