#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagonal scatter (Predicted vs True remaining time) for representative runs.

Modes:

A) Manual (as before):
  python plot_diag_pred_vs_true.py ^
    --pred_dir outputs/predictions ^
    --source_run_id Exp10Run19 ^
    --target_run_id Exp1Run1 ^
    --out_dir outputs/figures ^
    --combined

B) Auto-pick representative runs by run-level MAE:
  python plot_diag_pred_vs_true.py ^
    --pred_dir outputs/predictions ^
    --pick median ^
    --out_dir outputs/figures ^
    --combined

  python plot_diag_pred_vs_true.py ^
    --pred_dir outputs/predictions ^
    --pick worst ^
    --out_dir outputs/figures ^
    --combined

Outputs:
- PDFs (separate or combined)
- CSV files of the *actual plotted points* (sampled if needed)
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------
# Config / group definition
# -------------------------

SOURCE_EXPS = set(range(10, 19))   # 10..18
TARGET_EXPS = set(range(1, 10))    # 1..9
ALT_EXPS    = set(range(19, 28))   # 19..27

RUNID_RE = re.compile(r"^Exp(?P<exp>\d+)Run(?P<run>\d+)$", re.IGNORECASE)


def parse_run_id(run_id: str) -> Tuple[int, int]:
    m = RUNID_RE.match(run_id.strip())
    if not m:
        raise ValueError(f"Bad run id '{run_id}'. Expected like Exp10Run19.")
    return int(m.group("exp")), int(m.group("run"))


def group_of(exp: int, run: int, source_test_runs: set[int]) -> str:
    if exp in TARGET_EXPS:
        return "target"
    if exp in SOURCE_EXPS:
        return "source-test" if run in source_test_runs else "source-train"
    if exp in ALT_EXPS:
        return "alt"
    return "other"


# -------------------------
# Plot font config
# -------------------------

def configure_fonts(prefer_usetex: bool) -> bool:
    """Return True if usetex enabled, else False (fallback)."""
    latex_exe = shutil.which("latex")
    use_tex = bool(prefer_usetex and latex_exe)

    if use_tex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{lmodern}",
        })
        return True

    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return False


# -------------------------
# IO + metrics
# -------------------------

def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_parquet(path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()
    return df


def ensure_err_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = {"y_true", "y_pred"}
    if not need.issubset(df.columns):
        raise KeyError(f"Parquet must include {need}. Found: {list(df.columns)}")
    df = df.copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["y_true", "y_pred"])
    if "abs_err" not in df.columns:
        df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    if "signed_err" not in df.columns:
        df["signed_err"] = df["y_pred"] - df["y_true"]
    return df


def compute_run_mae(path: Path) -> float:
    """
    Compute run-level MAE for a parquet file.
    Prefer abs_err if present; else compute from y_true/y_pred.
    """
    df = read_parquet_safe(path)
    if "abs_err" in df.columns:
        x = pd.to_numeric(df["abs_err"], errors="coerce").dropna().to_numpy()
        return float(np.mean(x)) if len(x) else float("nan")
    df = ensure_err_cols(df)
    return float(df["abs_err"].mean()) if len(df) else float("nan")


def sample_points(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def robust_limits(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.995) -> float:
    vmax = float(np.nanmax([
        np.nanquantile(y_true, q),
        np.nanquantile(y_pred, q),
    ]))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    return vmax


def metrics_text(df: pd.DataFrame) -> str:
    mae = float(df["abs_err"].mean())
    rmse = float(np.sqrt(np.mean(np.square(df["signed_err"]))))

    bias = float(df["signed_err"].mean())
    n = int(len(df))
    return f"$n$={n:,}\nMAE={mae:.2f}s\nRMSE={rmse:.2f}s\nBias={bias:.2f}s"


# -------------------------
# Auto-pick helpers
# -------------------------

def discover_pred_files(pred_dir: Path) -> List[Path]:
    files = sorted(pred_dir.glob("Exp*Run*.parquet"))
    return [p for p in files if RUNID_RE.match(p.stem)]


def pick_representative_run(
    files: List[Path],
    criterion: str,
) -> Path:
    """
    criterion: 'median' or 'worst'
    Picks based on run-level MAE.
    """
    maes: List[Tuple[float, Path]] = []
    for p in files:
        mae = compute_run_mae(p)
        if np.isfinite(mae):
            maes.append((mae, p))

    if not maes:
        raise RuntimeError("No valid runs found (could not compute MAE on any candidate files).")

    maes.sort(key=lambda t: (t[0], t[1].stem))

    if criterion == "worst":
        return maes[-1][1]

    if criterion == "median":
        med = float(np.median([m for m, _ in maes]))
        # choose closest-to-median; tie-break by stem
        maes2 = sorted(maes, key=lambda t: (abs(t[0] - med), t[1].stem))
        return maes2[0][1]

    raise ValueError(f"Unknown criterion: {criterion}")


def select_paths_auto(
    pred_dir: Path,
    pick: str,
    source_test_runs: set[int],
) -> Tuple[Path, Path]:
    """
    Auto-select one source-test and one target run based on MAE.
    """
    all_files = discover_pred_files(pred_dir)
    if not all_files:
        raise RuntimeError(f"No Exp*Run*.parquet found under: {pred_dir}")

    src_files = []
    tgt_files = []

    for p in all_files:
        exp, run = parse_run_id(p.stem)
        g = group_of(exp, run, source_test_runs)
        if g == "source-test":
            src_files.append(p)
        elif g == "target":
            tgt_files.append(p)

    if not src_files:
        raise RuntimeError(
            f"No source-test candidate files found. Expected Exp10–18 runs {sorted(source_test_runs)}."
        )
    if not tgt_files:
        raise RuntimeError("No target candidate files found. Expected Exp1–9 runs 1–20.")

    print(f"[auto] Candidates: source-test={len(src_files)}  target={len(tgt_files)}")

    src_path = pick_representative_run(src_files, pick)
    tgt_path = pick_representative_run(tgt_files, pick)

    return src_path, tgt_path


# -------------------------
# Plotting
# -------------------------

def plot_diag(ax: plt.Axes, df: pd.DataFrame, title: str,
              max_q: float, use_hexbin: bool,
              s: float, alpha: float):
    y = df["y_true"].to_numpy()
    yp = df["y_pred"].to_numpy()

    vmax = robust_limits(y, yp, q=max_q)

    if use_hexbin:
        ax.hexbin(y, yp, gridsize=60, mincnt=1, cmap="Greys")
    else:
        ax.scatter(y, yp, s=s, alpha=alpha, color="black", linewidths=0)

    ax.plot([0, vmax], [0, vmax], color="black", lw=1.0, alpha=0.8)

    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("True remaining time (s)")
    ax.set_ylabel("Predicted remaining time (s)")
    ax.set_title(title)

    txt = metrics_text(df)
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6, alpha=0.9),
    )

    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_dir", type=str, default="", help="Directory with per-run parquet predictions.")

    # manual selection (as before)
    ap.add_argument("--source_run_id", type=str, default="", help="Manual: e.g., Exp10Run19")
    ap.add_argument("--target_run_id", type=str, default="", help="Manual: e.g., Exp1Run1")

    # auto selection
    ap.add_argument(
        "--pick",
        type=str,
        default="",
        choices=["", "median", "worst"],
        help="Auto-pick representative runs by MAE within source-test and target (requires --pred_dir).",
    )
    ap.add_argument(
        "--source_test_runs",
        type=str,
        default="19,20",
        help="Comma-separated run numbers defining source-test within Exp10–18 (default: 19,20).",
    )

    # explicit file paths (override)
    ap.add_argument("--in_source", type=str, default="", help="Explicit parquet path for source plot.")
    ap.add_argument("--in_target", type=str, default="", help="Explicit parquet path for target plot.")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--combined", action="store_true", help="Also write a combined 2-panel PDF.")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX text rendering if 'latex' is on PATH.")

    ap.add_argument("--max_points", type=int, default=200_000, help="Max points per plot (random sample).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_q", type=float, default=0.995, help="Quantile for axis limits (robust).")

    ap.add_argument("--hexbin", action="store_true", help="Use hexbin instead of scatter (faster / cleaner).")
    ap.add_argument("--marker_size", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.07)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    using_tex = configure_fonts(args.usetex)
    if args.usetex and not using_tex:
        print("[WARN] --usetex requested, but 'latex' not found on PATH. Falling back (no external LaTeX).")

    source_test_runs = {int(x.strip()) for x in args.source_test_runs.split(",") if x.strip()}

    # Resolve parquet paths in priority order:
    # 1) explicit --in_source/--in_target
    # 2) auto --pick
    # 3) manual --source_run_id/--target_run_id
    src_path: Optional[Path] = None
    tgt_path: Optional[Path] = None

    if args.in_source and args.in_target:
        src_path = Path(args.in_source)
        tgt_path = Path(args.in_target)
    else:
        if not args.pred_dir:
            raise SystemExit("Provide --pred_dir OR --in_source/--in_target.")

        pred_dir = Path(args.pred_dir)

        if args.pick:
            src_path, tgt_path = select_paths_auto(pred_dir, args.pick, source_test_runs)
            print(f"[auto] Picked source-test: {src_path.stem}")
            print(f"[auto] Picked target     : {tgt_path.stem}")
        else:
            if not args.source_run_id or not args.target_run_id:
                raise SystemExit("Manual mode needs --source_run_id and --target_run_id, or use --pick median|worst.")
            src_path = pred_dir / f"{args.source_run_id}.parquet"
            tgt_path = pred_dir / f"{args.target_run_id}.parquet"

    assert src_path is not None and tgt_path is not None

    src_name = src_path.stem
    tgt_name = tgt_path.stem

    # Load + sample
    df_src = ensure_err_cols(read_parquet_safe(src_path))
    df_tgt = ensure_err_cols(read_parquet_safe(tgt_path))

    df_src_plot = sample_points(df_src, args.max_points, args.seed)
    df_tgt_plot = sample_points(df_tgt, args.max_points, args.seed)

    # Export plotted points
    src_csv = out_dir / f"diag_points_{src_name}.csv"
    tgt_csv = out_dir / f"diag_points_{tgt_name}.csv"
    df_src_plot[["y_true", "y_pred", "abs_err", "signed_err"]].to_csv(src_csv, index=False)
    df_tgt_plot[["y_true", "y_pred", "abs_err", "signed_err"]].to_csv(tgt_csv, index=False)

    # Separate PDFs
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    plot_diag(ax, df_src_plot, f"Source-test ({src_name})", args.max_q, args.hexbin, args.marker_size, args.alpha)
    fig.tight_layout()
    out_src_pdf = out_dir / "error_diag_source_test.pdf"
    fig.savefig(out_src_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    plot_diag(ax, df_tgt_plot, f"Target ({tgt_name})", args.max_q, args.hexbin, args.marker_size, args.alpha)
    fig.tight_layout()
    out_tgt_pdf = out_dir / "error_diag_target.pdf"
    fig.savefig(out_tgt_pdf)
    plt.close(fig)

    # Combined 2-panel PDF (optional)
    if args.combined:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
        plot_diag(axes[0], df_src_plot, f"Source-test ({src_name})", args.max_q, args.hexbin, args.marker_size, args.alpha)
        plot_diag(axes[1], df_tgt_plot, f"Target ({tgt_name})", args.max_q, args.hexbin, args.marker_size, args.alpha)
        fig.tight_layout()
        out_comb = out_dir / "error_diag_compare.pdf"
        fig.savefig(out_comb)
        plt.close(fig)

    print("[OK] Wrote:")
    print(" -", out_src_pdf.resolve())
    print(" -", out_tgt_pdf.resolve())
    if args.combined:
        print(" -", (out_dir / "error_diag_compare.pdf").resolve())
    print("[OK] Exported plotted points:")
    print(" -", src_csv.resolve())
    print(" -", tgt_csv.resolve())


if __name__ == "__main__":
    main()
