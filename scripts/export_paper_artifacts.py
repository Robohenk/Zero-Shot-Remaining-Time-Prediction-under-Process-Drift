#!/usr/bin/env python3
"""Generate manuscript-ready tables, TeX snippets, and companion image files from predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.paper_exports.pgfplots import (
    export_fig4_artifacts,
    export_fig5_artifacts,
    export_group_summary_artifacts,
    export_run_metric_artifacts,
    export_run_size_table,
    export_start_event_artifacts,
)
from ppm_drift.utils.io import discover_runs, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser(description="Export manuscript-ready LaTeX snippets/tables plus PDF/JPG figures.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--source_test_runs", type=int, nargs="+", default=[19, 20])
    ap.add_argument("--skip_existing", action="store_true", help="Skip export if the main artifact files already exist.")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    logs_dir = ensure_dir(Path(out_dir) / 'logs')
    logger = setup_logger(logs_dir, filename='export_paper_artifacts.log')
    if args.skip_existing and (Path(out_dir) / "figures" / "fig3_severity_ladder_pgfplots_data.tex").exists():
        logger.info("Skipping artifact export because main figure assets already exist in %s", out_dir)
        return
    runs = discover_runs(args.data_root)
    source_test_runs = set(args.source_test_runs)

    with timed_stage(logger, logs_dir, out_dir, 'export_start_event_artifacts'):
        export_start_event_artifacts(runs, source_test_runs, out_dir)
    with timed_stage(logger, logs_dir, out_dir, 'export_run_metric_artifacts'):
        run_metrics = export_run_metric_artifacts(args.pred_dir, source_test_runs, out_dir)
    with timed_stage(logger, logs_dir, out_dir, 'export_run_size_table'):
        export_run_size_table(run_metrics, out_dir)
    with timed_stage(logger, logs_dir, out_dir, 'export_group_summary_artifacts'):
        export_group_summary_artifacts(run_metrics, out_dir)
    with timed_stage(logger, logs_dir, out_dir, 'export_fig4_artifacts'):
        export_fig4_artifacts(args.pred_dir, source_test_runs, out_dir)
    with timed_stage(logger, logs_dir, out_dir, 'export_fig5_artifacts'):
        export_fig5_artifacts(args.pred_dir, runs, source_test_runs, out_dir)


if __name__ == "__main__":
    main()
