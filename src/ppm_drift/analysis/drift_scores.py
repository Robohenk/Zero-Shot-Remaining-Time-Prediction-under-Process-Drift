from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..constants import RunId, label_group
from ..data.loading import load_log
from ..data.preprocess import build_prefix_table


def safe_prob(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    return p / p.sum()


def psi_numeric(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if len(train) == 0 or len(test) == 0:
        return np.nan
    edges = np.unique(np.quantile(train, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    train_counts, _ = np.histogram(train, bins=edges)
    test_counts, _ = np.histogram(test, bins=edges)
    p = safe_prob(train_counts + 1e-6)
    q = safe_prob(test_counts + 1e-6)
    return float(np.sum((q - p) * np.log(q / p)))


def js_divergence_categorical(train_vals: pd.Series, test_vals: pd.Series) -> float:
    cats = sorted(set(train_vals.astype(str).unique()).union(set(test_vals.astype(str).unique())))
    p = safe_prob(np.array([(train_vals.astype(str) == c).sum() for c in cats], dtype=float) + 1e-6)
    q = safe_prob(np.array([(test_vals.astype(str) == c).sum() for c in cats], dtype=float) + 1e-6)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def build_reference_prefix_table(train_ids: list[RunId], runs: dict[RunId, Path]) -> pd.DataFrame:
    return pd.concat([build_prefix_table(load_log(runs[rid]), rid) for rid in train_ids], ignore_index=True)


def compute_drift_scores(reference_prefixes: pd.DataFrame, eval_ids: list[RunId], runs: dict[RunId, Path], source_test_runs: set[int], include_label_shift: bool = True) -> pd.DataFrame:
    rows = []
    for rid in eval_ids:
        pt = build_prefix_table(load_log(runs[rid]), rid)
        comps = {
            "psi_progress": psi_numeric(reference_prefixes["progress"].to_numpy(), pt["progress"].to_numpy()),
            "psi_delta_t": psi_numeric(reference_prefixes["delta_t"].to_numpy(), pt["delta_t"].to_numpy()),
            "psi_decay": psi_numeric(reference_prefixes["currentDecayLevel"].fillna(0).to_numpy(), pt["currentDecayLevel"].fillna(0).to_numpy()),
            "js_event": js_divergence_categorical(reference_prefixes["event"], pt["event"]),
            "js_vehicle": js_divergence_categorical(reference_prefixes["vehicleType"], pt["vehicleType"]),
        }
        if include_label_shift:
            comps["psi_y_true"] = psi_numeric(reference_prefixes["y_true"].to_numpy(), pt["y_true"].to_numpy())
        rows.append({
            "run_id": rid.key,
            "exp": rid.exp,
            "run": rid.run,
            "group": label_group(rid.exp, rid.run, source_test_runs),
            **comps,
            "drift_score": float(np.nanmean(list(comps.values()))),
        })
    return pd.DataFrame(rows)


def spearman_rho(x, y):
    if len(x) < 2:
        return np.nan
    return float(stats.spearmanr(x, y, nan_policy="omit").statistic)
