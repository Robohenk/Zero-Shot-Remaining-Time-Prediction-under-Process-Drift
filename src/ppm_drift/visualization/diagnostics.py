from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import label_group
from ..utils.io import parquet_files
from .style import set_paper_style


def pick_representative_run(pred_dir: str | Path, source_test_runs: set[int], group: str, mode: str = "median") -> Path:
    rows = []
    files = parquet_files(pred_dir)
    for path in files:
        df = pd.read_parquet(path, columns=["exp", "run", "run_id", "y_true", "y_pred"])
        exp = int(df["exp"].iloc[0])
        run = int(df["run"].iloc[0])
        g = label_group(exp, run, source_test_runs)
        if g != group:
            continue
        mae = float(np.mean(np.abs(df["y_pred"] - df["y_true"])))
        rows.append((path, mae))
    rows = sorted(rows, key=lambda x: x[1])
    if not rows:
        raise ValueError(f"No runs found for group={group}")
    return rows[len(rows)//2][0] if mode == "median" else rows[-1][0]


def plot_pred_vs_true(path: str | Path, out_path: str | Path, title: str, max_points: int = 6000, seed: int = 7) -> None:
    set_paper_style()
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    if len(df) > max_points:
        idx = rng.choice(len(df), size=max_points, replace=False)
        df = df.iloc[idx].copy()
    lim = np.quantile(np.r_[df["y_true"].to_numpy(), df["y_pred"].to_numpy()], 0.995)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(df["y_true"], df["y_pred"], s=5, alpha=0.25)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True remaining time (s)")
    ax.set_ylabel("Predicted remaining time (s)")
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
