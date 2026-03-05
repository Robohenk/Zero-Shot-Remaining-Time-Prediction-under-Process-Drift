#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def read_csv_flex(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="latin1")


def norm_group(s: str) -> str:
    s = str(s)
    s = s.replace("â€“", "–").replace("â€”", "—")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"Missing columns. Tried {candidates}. Available: {list(df.columns)}")


def maybe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


CANON_ORDER = [
    "source-test (Exp10–18)",
    "target (Exp1–9)",
    "alt (Exp19–27)",
]

STYLE = {
    "source-test (Exp10–18)": dict(color="black", linestyle="-",  marker="o", markersize=3.5, linewidth=1.2),
    "target (Exp1–9)":        dict(color="black", linestyle="--", marker="s", markersize=3.5, linewidth=1.2),
    "alt (Exp19–27)":         dict(color="black", linestyle=":",  marker="D", markersize=3.5, linewidth=1.2),
}


def normalize_to_canonical_group(g: str) -> str:
    g0 = norm_group(g)
    gl = g0.lower()
    if "source-test" in gl:
        return "source-test (Exp10–18)"
    if "target" in gl:
        return "target (Exp1–9)"
    if "alt" in gl:
        return "alt (Exp19–27)"
    return g0


def configure_fonts(prefer_usetex: bool) -> bool:
    """
    Returns True if usetex is enabled, False if we fell back.
    """
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

    # Fallback: no external LaTeX needed; still serif + mathtext
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


def plot_lines_with_optional_ci(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    groupcol: str,
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    xlim=None,
    ylim=None,
    show_legend: bool = False,   # IMPORTANT: default False to avoid double legends
):
    df = df.copy()
    df[groupcol] = df[groupcol].astype(str).map(normalize_to_canonical_group)

    # optional CI columns for MAE-style panels
    ylo = maybe_col(df, ["mae_ci_lo", "MAE_ci_lo", "ci_lo", "lower", "lo"])
    yhi = maybe_col(df, ["mae_ci_hi", "MAE_ci_hi", "ci_hi", "upper", "hi"])

    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    if ylo and yhi:
        df[ylo] = pd.to_numeric(df[ylo], errors="coerce")
        df[yhi] = pd.to_numeric(df[yhi], errors="coerce")

    groups = [g for g in CANON_ORDER if g in set(df[groupcol])]
    groups += [g for g in sorted(df[groupcol].unique()) if g not in groups]

    for g in groups:
        sub = df[df[groupcol] == g].sort_values(xcol)
        if sub.empty:
            continue
        st = STYLE.get(g, dict(color="black", linestyle="-", marker="o", markersize=3.5, linewidth=1.2))
        ax.plot(sub[xcol], sub[ycol], label=g, **st)

        if ylo and yhi and sub[ylo].notna().any() and sub[yhi].notna().any():
            ax.fill_between(sub[xcol], sub[ylo], sub[yhi], color=st.get("color", "black"), alpha=0.12, linewidth=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, axis="y", linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(frameon=False, loc="best")


def add_outside_legend_and_layout(fig: plt.Figure, ax: plt.Axes, right: float = 0.78):
    """
    Put legend outside on the right without shrinking axes unexpectedly.
    """
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0)
    fig.subplots_adjust(right=right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_a", required=True)
    ap.add_argument("--in_b", required=True)
    ap.add_argument("--in_c", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--usetex", action="store_true", help="Try to use external LaTeX (requires 'latex' on PATH).")
    ap.add_argument("--right", type=float, default=0.78, help="Right margin for outside legend (0-1).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    using_tex = configure_fonts(prefer_usetex=args.usetex)
    if args.usetex and not using_tex:
        print("[WARN] --usetex requested, but 'latex' not found on PATH. Falling back to mathtext (no external LaTeX).")

    # ---- Fig4a ----
    df_a = read_csv_flex(Path(args.in_a))
    df_a["group"] = df_a["group"].map(norm_group)

    xcol_a = find_col(df_a, ["rt_mid", "y_true_mid", "true_rt_mid", "rt_center"])
    ycol_a = find_col(df_a, ["mae", "MAE", "abs_error_mean", "mean_abs_err"])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    plot_lines_with_optional_ci(
        df=df_a, xcol=xcol_a, ycol=ycol_a, groupcol="group", ax=ax,
        xlabel="True remaining time (s)",
        ylabel="MAE (s)",
        show_legend=False,
    )
    add_outside_legend_and_layout(fig, ax, right=args.right)
    fig.savefig(out_dir / "fig4a_mae_vs_true_rt.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig4b ----
    df_b = read_csv_flex(Path(args.in_b))
    df_b["group"] = df_b["group"].map(norm_group)

    xcol_b = find_col(df_b, ["rt_mid", "y_true_mid", "true_rt_mid", "rt_center"])
    ycol_b = find_col(df_b, ["mean_signed", "signed_err_mean", "signed_error_mean", "signed_err"])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    plot_lines_with_optional_ci(
        df=df_b, xcol=xcol_b, ycol=ycol_b, groupcol="group", ax=ax,
        xlabel="True remaining time (s)",
        ylabel="Mean signed error (s)",
        show_legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
    add_outside_legend_and_layout(fig, ax, right=args.right)
    fig.savefig(out_dir / "fig4b_signed_vs_true_rt.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig4c ----
    df_c = read_csv_flex(Path(args.in_c))
    df_c["group"] = df_c["group"].map(norm_group)

    xcol_c = find_col(df_c, ["prog_mid", "progress_mid", "progress_center"])
    ycol_c = find_col(df_c, ["mae", "MAE", "abs_error_mean", "mean_abs_err"])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    plot_lines_with_optional_ci(
        df=df_c, xcol=xcol_c, ycol=ycol_c, groupcol="group", ax=ax,
        xlabel="Prefix progress (fraction of case completed)",
        ylabel="MAE (s)",
        xlim=(0.0, 1.0),
        show_legend=False,
    )
    add_outside_legend_and_layout(fig, ax, right=args.right)
    fig.savefig(out_dir / "fig4c_mae_vs_progress.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Wrote PDFs to:", out_dir.resolve())
    print(" - fig4a_mae_vs_true_rt.pdf")
    print(" - fig4b_signed_vs_true_rt.pdf")
    print(" - fig4c_mae_vs_progress.pdf")


if __name__ == "__main__":
    main()
