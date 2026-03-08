#!/usr/bin/env python3
"""Aggregate server-produced outputs into the compact manuscript tables and benchmark summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ppm_drift.utils.io import ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage
from ppm_drift.paper_exports.pgfplots import export_cross_model_summary_tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate cross-model and benchmark outputs after server jobs complete.")
    ap.add_argument("--work_root", default="reproducibility_run")
    args = ap.parse_args()

    work_root = ensure_dir(args.work_root)
    logs_dir = ensure_dir(work_root / "logs")
    logger = setup_logger(logs_dir, filename="server_stage4_aggregate.log")

    with timed_stage(logger, logs_dir, work_root, "export_cross_model_summary_tables"):
        export_cross_model_summary_tables(work_root / "predictions", work_root)

    # Optional benchmark aggregation across per-job outputs
    bench_root = work_root / "benchmarks"
    rows = []
    for p in bench_root.rglob("epoch_grid_summary.csv"):
        try:
            rows.append(pd.read_csv(p))
        except Exception:
            logger.exception("Failed to read %s", p)
    if rows:
        out = pd.concat(rows, ignore_index=True).drop_duplicates()
        ensure_dir(work_root / "figures")
        out.to_csv(work_root / "figures" / "benchmark_summary_all.csv", index=False)
        logger.info("Wrote benchmark summary with %d rows", len(out))


if __name__ == "__main__":
    main()
