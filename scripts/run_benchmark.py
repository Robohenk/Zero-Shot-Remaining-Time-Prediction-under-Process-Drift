#!/usr/bin/env python3
"""Run the epoch-sensitivity benchmark for one architecture and export summary results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.config import BenchmarkConfig
from ppm_drift.training.runner import run_epoch_grid
from ppm_drift.utils.io import discover_runs, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an epoch-sensitivity benchmark for one architecture.")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--output_root", type=str, default="outputs")
    ap.add_argument("--model_name", type=str, choices=["lstm", "transformer", "tft"], default="lstm")
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--single_job", action="store_true", help="Treat the provided one epoch/one seed as one schedulable benchmark job.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if the target benchmark output already exists.")
    args = ap.parse_args()

    cfg = BenchmarkConfig()
    cfg.data.data_root = args.data_root
    cfg.data.output_root = args.output_root
    cfg.train.model_name = args.model_name
    runs = discover_runs(cfg.data.data_root)
    if args.single_job and (len(args.epochs) != 1 or len(args.seeds) != 1):
        raise SystemExit("--single_job expects exactly one epoch and one seed.")
    logs_dir = ensure_dir(Path(cfg.data.output_root) / 'logs')
    logger = setup_logger(logs_dir, filename='benchmark.log')
    if args.single_job:
        target = Path(cfg.data.output_root) / f"epoch_grid_{cfg.train.model_name}" / f"epochs_{args.epochs[0]}_seed_{args.seeds[0]}" / cfg.train.model_name / "run_metrics.csv"
        if args.skip_existing and target.exists():
            logger.info("Skipping benchmark job because outputs already exist: %s", target)
            print(target)
            return

    with timed_stage(logger, logs_dir, cfg.data.output_root, f"run_benchmark_{cfg.train.model_name}", {"current_model": cfg.train.model_name}):
        summary = run_epoch_grid(runs, cfg, args.epochs, args.seeds, logger=logger, work_root=cfg.data.output_root)
    print(summary)


if __name__ == "__main__":
    main()
