from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import label_group
from ..utils.io import parquet_files


def _load_all_predictions(pred_dir: str | Path, source_test_runs: set[int]) -> pd.DataFrame:
    parts = []
    for path in parquet_files(pred_dir):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "group" not in df.columns:
            df["group"] = [label_group(int(e), int(r), source_test_runs) for e, r in zip(df["exp"], df["run"])]
        if "abs_err" not in df.columns:
            df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
        if "signed_err" not in df.columns:
            df["signed_err"] = df["y_pred"] - df["y_true"]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def binned_error_curves(pred_dir: str | Path, source_test_runs: set[int], n_rt_bins: int = 12, n_progress_bins: int = 10):
    df = _load_all_predictions(pred_dir, source_test_runs)
    rt_bins = pd.qcut(df["y_true"], q=n_rt_bins, duplicates="drop")
    rt = df.groupby(["group", rt_bins], observed=True).agg(
        x=("y_true", "median"),
        mae=("abs_err", "mean"),
        signed=("signed_err", "mean"),
        n=("y_true", "size"),
    ).reset_index(drop=True)

    progress_edges = np.linspace(0.0, 1.0, n_progress_bins + 1)
    progress_bin = pd.cut(df["progress"], bins=progress_edges, include_lowest=True)
    progress = df.groupby(["group", progress_bin], observed=True).agg(
        x=("progress", "mean"),
        mae=("abs_err", "mean"),
        n=("progress", "size"),
    ).reset_index(drop=True)
    return rt, progress
