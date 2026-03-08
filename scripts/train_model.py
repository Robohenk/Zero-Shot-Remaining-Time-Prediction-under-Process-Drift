#!/usr/bin/env python3
"""Train one model architecture and export evaluation predictions plus run-level metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.config import BenchmarkConfig
from ppm_drift.training.runner import train_one_model
from ppm_drift.utils.io import discover_runs, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one remaining-time model and export per-run predictions.")
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config. If omitted, CLI flags are used.")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--output_root", type=str, default="outputs")
    ap.add_argument("--model_name", type=str, choices=["lstm", "transformer", "tft"], default="lstm")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_existing", action="store_true", help="Skip training if final outputs for this model/seed/epoch already exist.")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_json(args.config) if args.config else BenchmarkConfig()
    cfg.data.data_root = args.data_root
    cfg.data.output_root = args.output_root
    cfg.train.model_name = args.model_name
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.seed = args.seed

    logs_dir = ensure_dir(Path(cfg.data.output_root) / 'logs')
    logger = setup_logger(logs_dir, filename='train.log')
    runs = discover_runs(cfg.data.data_root)
    model_root_guess = Path(cfg.data.output_root) / cfg.train.model_name
    final_metrics = model_root_guess / "run_metrics.csv"
    final_predictions = model_root_guess / "predictions"
    if args.skip_existing and final_metrics.exists() and final_predictions.exists() and any(final_predictions.glob("*.parquet")):
        logger.info("Skipping %s because outputs already exist at %s", cfg.train.model_name, model_root_guess)
        print(model_root_guess)
        return

    with timed_stage(logger, logs_dir, cfg.data.output_root, f"train_model_{cfg.train.model_name}", {"current_model": cfg.train.model_name}):
        model_root = train_one_model(runs, cfg, logger=logger, work_root=cfg.data.output_root)
    print(model_root)


if __name__ == "__main__":
    main()
