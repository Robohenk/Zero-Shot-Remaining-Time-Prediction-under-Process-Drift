#!/usr/bin/env python3
"""
make_fig2_heatmap.py

Create heatmaps (MAE/RMSE) from outputs/fig2/fig2_run_metrics.csv

Input CSV format:
exp,run,run_id,group,MAE,RMSE

Outputs (per metric):
- fig2_heatmap_<METRIC>.png
- fig2_heatmap_<METRIC>.pdf
- fig2_heatmap_<METRIC>_pivot.csv   (wide table used for the heatmap)

Changes requested:
- SWAP axes: x = experiments, y = runs
- Experiment tick labels: no "Exp"; show "<number> (<target/source-test/alt>)"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GROUP_ORDER = [
    "target (Exp1–9)",
    "source-test (Exp10–18)",
    "alt (Exp19–27)",
]


def configure_fonts(usetex: bool, cm: bool, base_size: int = 9) -> None:
    """Call before creating any figures."""
    import matplotlib as mpl

    common = {
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
    }

    if usetex:
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                **common,
            }
        )
    elif cm:
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                **common,
            }
        )
    else:
        mpl.rcParams.update(common)


def fix_group_encoding(s: pd.Series) -> pd.Series:
    """Fix common Windows CSV encoding issue for en-dash."""
    return s.astype(str).str.replace("â€“", "–", regex=False).str.strip()


def add_group_exp_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = {"group", "exp", "run"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")

    df["group"] = fix_group_encoding(df["group"])

    g2k = {g: i for i, g in enumerate(GROUP_ORDER)}
    df["group_k"] = df["group"].map(g2k).fillna(99).astype(int)

    df["exp"] = pd.to_numeric(df["exp"], errors="coerce")
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df = df.dropna(subset=["exp", "run"])
    df["exp"] = df["exp"].astype(int)
    df["run"] = df["run"].astype(int)

    return df.sort_values(["group_k", "exp", "run"])


def _exp_label(exp: int, group: str) -> str:
    # group is like "target (Exp1–9)" -> we want "target" / "source-test" / "alt"
    short = str(group).split(" ")[0]  # target | source-test | alt
    return f"{exp} ({short})"


def make_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_pdf: Path,
    out_png: Path,
    out_csv: Path,
    robust: bool = True,
    dpi: int = 600,
) -> None:
    if metric not in ("MAE", "RMSE"):
        raise ValueError("metric must be 'MAE' or 'RMSE'")

    df = df.copy()

    # Build stable exp -> (group_k, group, label) mapping
    # (Each experiment belongs to one group in your design.)
    exp_map = (
        df.groupby("exp", as_index=False)
          .agg(group_k=("group_k", "min"), group=("group", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]))
    )
    exp_map["label"] = exp_map.apply(lambda r: _exp_label(int(r["exp"]), r["group"]), axis=1)

    # Column order: target exps, then source-test exps, then alt exps (each sorted)
    exp_map = exp_map.sort_values(["group_k", "exp"])
    ordered_exps = exp_map["exp"].tolist()

    # Pivot: rows = runs (y-axis), cols = experiments (x-axis)
    pivot = (
        df.pivot_table(index="run", columns="exp", values=metric, aggfunc="mean")
    )

    # Ensure runs 1..20 exist and are ordered
    run_index = list(range(1, 21))
    pivot = pivot.reindex(index=run_index)

    # Ensure experiments exist and are ordered
    pivot = pivot.reindex(columns=ordered_exps)

    # Export pivot for external plotting (columns are labels, not "ExpX")
    col_labels = [exp_map.loc[exp_map["exp"] == e, "label"].iloc[0] for e in ordered_exps]
    pivot_out = pivot.copy()
    pivot_out.columns = col_labels
    pivot_out.reset_index().rename(columns={"run": "Run"}).to_csv(out_csv, index=False)

    # Robust color scale (5th–95th percentile)
    vmin = vmax = None
    if robust:
        vals = pivot.to_numpy(dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            vmin = float(np.percentile(vals, 5))
            vmax = float(np.percentile(vals, 95))

    Z = pivot.to_numpy(dtype=float)  # shape: (runs, exps)

    # Figure sizing similar to your other heatmap
    fig_w = max(7.0, 0.35 * Z.shape[1] + 2.2)  # experiments on x
    fig_h = max(4.5, 0.30 * Z.shape[0] + 2.0)  # runs on y
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0.9, 0.9, 0.9, 1.0))

    im = ax.imshow(
        Z,
        aspect="auto",
        interpolation="nearest",
        origin="lower",  # Run1 at bottom (matches your other script)
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Labels (NO TITLE)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Run")

    # X ticks = experiments (numbers + group short name; NO "Exp")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=90, va="top")

    # Y ticks = runs
    ax.set_yticks(np.arange(len(run_index)))
    ax.set_yticklabels([str(r) for r in run_index], fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Minor grid (white cell borders)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(run_index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Vertical separators between groups (target | source-test | alt)
    # We look at group_k sequence along ordered_exps.
    exp_to_gk = {int(r.exp): int(r.group_k) for r in exp_map.itertuples(index=False)}
    gks = [exp_to_gk[e] for e in ordered_exps]
    for j in range(1, len(gks)):
        if gks[j] != gks[j - 1]:
            ax.vlines(
                j - 0.5,
                -0.5,
                len(run_index) - 0.5,
                colors="white",
                linewidth=2.5,
            )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Path to outputs/fig2/fig2_run_metrics.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory (e.g., outputs/fig2)")
    ap.add_argument("--metric", default="both", choices=["MAE", "RMSE", "both"])
    ap.add_argument("--robust", action="store_true", help="Clip color scale to 5th–95th percentiles")
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--cm", action="store_true")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    configure_fonts(args.usetex, args.cm)

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Safe encoding fallback
    try:
        df = pd.read_csv(in_csv, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(in_csv, engine="python", encoding="cp1252")

    df = add_group_exp_sort(df)

    metrics = ["MAE", "RMSE"] if args.metric == "both" else [args.metric]
    for m in metrics:
        make_heatmap(
            df=df,
            metric=m,
            out_pdf=out_dir / f"fig2_heatmap_{m}.pdf",
            out_png=out_dir / f"fig2_heatmap_{m}.png",
            out_csv=out_dir / f"fig2_heatmap_{m}_pivot.csv",
            robust=args.robust,
            dpi=args.dpi,
        )
        print(f"Wrote: {out_dir / f'fig2_heatmap_{m}.pdf'}")


if __name__ == "__main__":
    main()
