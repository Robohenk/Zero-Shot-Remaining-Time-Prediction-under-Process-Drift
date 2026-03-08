#!/usr/bin/env python3
"""Summarize how many prediction rows, prefixes, and cases were exported per run and group."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.constants import label_group
from ppm_drift.utils.io import parquet_files, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize number of prefixes and cases per prediction file.")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_existing", action="store_true", help="Skip if the inventory summary already exists.")
    ap.add_argument("--source_test_runs", type=int, nargs="+", default=[19, 20])
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    if args.skip_existing and (Path(out_dir) / "prediction_inventory_summary.csv").exists():
        print(out_dir)
        return
    logs_dir = ensure_dir(Path(args.out_dir) / 'logs')
    logger = setup_logger(logs_dir, filename='summarize_prediction_inventory.log')
    rows = []
    source_test_runs = set(args.source_test_runs)
    with timed_stage(logger, logs_dir, args.out_dir, 'summarize_prediction_inventory'):
        for path in parquet_files(args.pred_dir):
            df = pd.read_parquet(path, columns=["run_id", "exp", "run", "case_id"])
            rows.append({
                "run_id": str(df["run_id"].iloc[0]),
                "exp": int(df["exp"].iloc[0]),
                "run": int(df["run"].iloc[0]),
                "group": label_group(int(df["exp"].iloc[0]), int(df["run"].iloc[0]), source_test_runs),
                "n_prefixes": len(df),
                "n_cases": df["case_id"].nunique(),
            })
        pd.DataFrame(rows).sort_values(["exp", "run"]).to_csv(Path(out_dir) / 'prediction_inventory.csv', index=False)


if __name__ == "__main__":
    main()
