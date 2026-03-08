#!/usr/bin/env python3
"""Compute case-start coverage statistics and export start-event artifacts used in the manuscript."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.paper_exports.pgfplots import export_start_event_artifacts
from ppm_drift.utils.io import discover_runs, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--source_test_runs', type=int, nargs='+', default=[19, 20])
    args = ap.parse_args()
    runs = discover_runs(args.data_root)
    logs_dir = ensure_dir(Path(args.out_dir) / 'logs')
    logger = setup_logger(logs_dir, filename='case_start_coverage.log')
    with timed_stage(logger, logs_dir, args.out_dir, 'case_start_coverage'):
        export_start_event_artifacts(runs, set(args.source_test_runs), args.out_dir)

if __name__ == '__main__':
    main()
