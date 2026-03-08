#!/usr/bin/env python3
"""Run one main-comparison model job on a server: train, predict, and export all model-specific artifacts."""
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
    ap = argparse.ArgumentParser(description="Run one model job for server parallelization.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--work_root", default="reproducibility_run")
    ap.add_argument("--model_name", required=True, choices=["lstm", "transformer", "tft"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    work_root = ensure_dir(args.work_root)
    logs_dir = ensure_dir(work_root / "logs")
    logger = setup_logger(logs_dir, filename=f"server_model_job_{args.model_name}.log")
    model_output_root = work_root / "predictions" / args.model_name
    pred_dir = model_output_root / args.model_name / "predictions"
    out_dir = work_root / "artifacts" / args.model_name

    with timed_stage(logger, logs_dir, work_root, f"train_and_predict_{args.model_name}", {"current_model": args.model_name}):
        run_cmd([
            sys.executable, str(ROOT / "scripts" / "train_model.py"),
            "--data_root", args.data_root,
            "--output_root", str(model_output_root),
            "--model_name", args.model_name,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--skip_existing",
        ] if args.skip_existing else [
            sys.executable, str(ROOT / "scripts" / "train_model.py"),
            "--data_root", args.data_root,
            "--output_root", str(model_output_root),
            "--model_name", args.model_name,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
        ], logger)

    with timed_stage(logger, logs_dir, work_root, f"export_paper_artifacts_{args.model_name}", {"current_model": args.model_name}):
        base_cmd = [sys.executable, str(ROOT / "scripts" / "export_paper_artifacts.py"), "--data_root", args.data_root, "--pred_dir", str(pred_dir), "--out_dir", str(out_dir)]
        if args.skip_existing:
            base_cmd.append("--skip_existing")
        run_cmd(base_cmd, logger)

        plot_cmd = [sys.executable, str(ROOT / "scripts" / "plot_representative_runs.py"), "--pred_dir", str(pred_dir), "--out_dir", str(out_dir)]
        if args.skip_existing:
            plot_cmd.append("--skip_existing")
        run_cmd(plot_cmd, logger)

        inv_cmd = [sys.executable, str(ROOT / "scripts" / "summarize_prediction_inventory.py"), "--pred_dir", str(pred_dir), "--out_dir", str(out_dir)]
        if args.skip_existing:
            inv_cmd.append("--skip_existing")
        run_cmd(inv_cmd, logger)


if __name__ == "__main__":
    main()
