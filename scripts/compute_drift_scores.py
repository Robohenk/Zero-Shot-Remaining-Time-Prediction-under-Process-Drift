#!/usr/bin/env python3
"""Compute run-level distribution-shift scores against the source-training reference logs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.paper_exports.pgfplots import export_fig5_artifacts
from ppm_drift.utils.io import discover_runs, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--source_test_runs', type=int, nargs='+', default=[19, 20])
    args = ap.parse_args()
    runs = discover_runs(args.data_root)
    logs_dir = ensure_dir(Path(args.out_dir) / 'logs')
    logger = setup_logger(logs_dir, filename='compute_drift_scores.log')
    with timed_stage(logger, logs_dir, args.out_dir, 'compute_drift_scores'):
        export_fig5_artifacts(args.pred_dir, runs, set(args.source_test_runs), args.out_dir)

if __name__ == '__main__':
    main()
