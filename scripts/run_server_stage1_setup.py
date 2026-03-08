#!/usr/bin/env python3
"""Run only the shared setup stages for server execution: dataset validation and case-start coverage."""
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
    ap = argparse.ArgumentParser(description="Run only the shared setup stages on a server.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--work_root", default="reproducibility_run")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    work_root = ensure_dir(args.work_root)
    logs_dir = ensure_dir(work_root / "logs")
    logger = setup_logger(logs_dir, filename="server_stage1_setup.log")

    dataset_summary = work_root / "dataset_checks" / "dataset_validation_summary.json"
    if not (args.skip_existing and dataset_summary.exists()):
        with timed_stage(logger, logs_dir, work_root, "validate_dataset"):
            run_cmd([sys.executable, str(ROOT / "scripts" / "validate_dataset.py"), "--data_root", args.data_root, "--out_dir", str(work_root / "dataset_checks")], logger)
    else:
        logger.info("Skipping validate_dataset because %s already exists", dataset_summary)

    start_event_file = work_root / "figures" / "start_event_boxplot_snippet.tex"
    if not (args.skip_existing and start_event_file.exists()):
        with timed_stage(logger, logs_dir, work_root, "case_start_coverage"):
            run_cmd([sys.executable, str(ROOT / "scripts" / "analyze_case_start_coverage.py"), "--data_root", args.data_root, "--out_dir", str(work_root)], logger)
    else:
        logger.info("Skipping case_start_coverage because %s already exists", start_event_file)


if __name__ == "__main__":
    main()
