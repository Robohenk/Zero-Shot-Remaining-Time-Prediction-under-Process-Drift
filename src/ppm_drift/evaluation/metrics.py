from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import label_group
from ..utils.io import parquet_files


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


def run_metrics_from_predictions(pred_dir: str | Path, source_test_runs: set[int]):
    rows = []
    for path in parquet_files(pred_dir):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "abs_err" not in df.columns:
            df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
        if "signed_err" not in df.columns:
            df["signed_err"] = df["y_pred"] - df["y_true"]
        exp = int(df["exp"].iloc[0])
        run = int(df["run"].iloc[0])
        rows.append({
            "run_id": str(df["run_id"].iloc[0]),
            "exp": exp,
            "run": run,
            "group": label_group(exp, run, source_test_runs),
            "n_prefixes": len(df),
            "n_cases": df["case_id"].nunique(),
            "mae": float(df["abs_err"].mean()),
            "rmse": float(np.sqrt(np.mean((df["y_pred"] - df["y_true"]) ** 2))),
            "bias": float(df["signed_err"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["exp", "run"]).reset_index(drop=True)


def summarize_groups(run_metrics: pd.DataFrame):
    rows = []
    for group, gdf in run_metrics.groupby("group"):
        mae_lo, mae_hi = bootstrap_ci(gdf["mae"].to_numpy())
        rmse_lo, rmse_hi = bootstrap_ci(gdf["rmse"].to_numpy())
        rows.append({
            "group": group,
            "n_runs": len(gdf),
            "mae_mean": float(gdf["mae"].mean()),
            "mae_ci_lo": mae_lo,
            "mae_ci_hi": mae_hi,
            "rmse_mean": float(gdf["rmse"].mean()),
            "rmse_ci_lo": rmse_lo,
            "rmse_ci_hi": rmse_hi,
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
