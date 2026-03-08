#!/usr/bin/env python3
"""Select representative runs and export predicted-vs-true diagnostic scatter plots for the paper."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.visualization.diagnostics import pick_representative_run, plot_pred_vs_true
from ppm_drift.utils.io import ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser(description="Export representative predicted-vs-true diagnostic figures.")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_existing", action="store_true", help="Skip if the representative diagnostic figures already exist.")
    ap.add_argument("--source_test_runs", type=int, nargs="+", default=[19, 20])
    args = ap.parse_args()
    source_test_runs = set(args.source_test_runs)
    out_dir = ensure_dir(Path(args.out_dir) / 'figures')
    logs_dir = ensure_dir(Path(args.out_dir) / 'logs')
    logger = setup_logger(logs_dir, filename='plot_representative_runs.log')

    jobs = [
        ('source-test', 'median', 'error_diag_source_test_bestmedian.pdf', 'Source-test (median MAE run)'),
        ('target', 'median', 'error_diag_target_bestmedian.pdf', 'Target (median MAE run)'),
        ('source-test', 'worst', 'error_diag_source_test_worstMAE.pdf', 'Source-test (worst MAE run)'),
        ('target', 'worst', 'error_diag_target_worstMAE.pdf', 'Target (worst MAE run)'),
    ]
    for group, mode, fname, title in jobs:
        with timed_stage(logger, logs_dir, args.out_dir, f'plot_{group}_{mode}'):
            path = pick_representative_run(args.pred_dir, source_test_runs, group, mode=mode)
            plot_pred_vs_true(path, out_dir / fname, title=title)
            jpg = (out_dir / fname).with_suffix('.jpg')
            plot_pred_vs_true(path, jpg, title=title)


if __name__ == '__main__':
    main()
