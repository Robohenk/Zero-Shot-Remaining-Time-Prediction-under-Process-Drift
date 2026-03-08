#!/usr/bin/env python3
"""Validate the 540 raw log files, their naming convention, and the expected sample schema."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppm_drift.data.loading import load_log
from ppm_drift.utils.io import discover_runs, dump_json, ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage

EXPECTED_COLS = {
    "uniqueID", "productID", "productIDStr", "productType", "productNr", "event",
    "timeStamp", "vehicleType", "vehicle", "currentDecayLevel", "processingStation",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the 540 raw Exp{exp}Run{run}.txt files.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    logger = setup_logger(out_dir, filename="validate_dataset.log")
    summary = {}

    with timed_stage(logger, out_dir, out_dir.parent if out_dir.name else out_dir, "discover_and_validate_inventory"):
        runs = discover_runs(args.data_root)
        expected = {(e, r) for e in range(1, 28) for r in range(1, 21)}
        found = {(rid.exp, rid.run) for rid in runs}
        missing = sorted(expected - found)
        extras = sorted(found - expected)
        logger.info("Discovered %d/540 expected runs", len(found))
        summary.update({
            "data_root": str(Path(args.data_root).resolve()),
            "n_found": len(found),
            "n_expected": 540,
            "missing": missing,
            "extras": extras,
        })

    with timed_stage(logger, out_dir, out_dir.parent if out_dir.name else out_dir, "validate_sample_schema"):
        schema = {"columns": [], "missing_expected_columns": [], "sample_file": None}
        if runs:
            first_path = next(iter(runs.values()))
            df = load_log(first_path)
            cols = list(df.columns)
            schema = {
                "sample_file": str(first_path),
                "columns": cols,
                "missing_expected_columns": sorted(EXPECTED_COLS - set(cols)),
            }
        summary["schema"] = schema

    dump_json(out_dir / "dataset_validation_summary.json", summary)
    logger.info("Wrote %s", out_dir / "dataset_validation_summary.json")
    if summary["n_found"] != 540:
        raise SystemExit("Dataset inventory is incomplete. Inspect dataset_validation_summary.json.")
    if summary["schema"]["missing_expected_columns"]:
        raise SystemExit("Dataset schema mismatch. Inspect dataset_validation_summary.json.")


if __name__ == "__main__":
    main()
