#!/usr/bin/env python3
"""Export concise cross-model accuracy and compute-cost tables for the manuscript."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.paper_exports.pgfplots import export_cross_model_summary_tables
from ppm_drift.utils.io import ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser(description="Export concise cross-model comparison tables from the full pipeline outputs.")
    ap.add_argument("--predictions_root", required=True)
    ap.add_argument("--work_root", required=True)
    args = ap.parse_args()

    work_root = ensure_dir(args.work_root)
    logs_dir = ensure_dir(Path(work_root) / "logs")
    logger = setup_logger(logs_dir, filename="export_model_summary_tables.log")
    with timed_stage(logger, logs_dir, work_root, "export_cross_model_summary_tables"):
        export_cross_model_summary_tables(args.predictions_root, work_root)


if __name__ == "__main__":
    main()
