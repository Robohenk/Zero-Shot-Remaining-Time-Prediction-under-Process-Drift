import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP_ORDER = ["source-test (Exp10–18)", "target (Exp1–9)", "alt (Exp19–27)"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_dir_csv", required=True)
    ap.add_argument("--n_bins_rt", type=int, default=20)
    ap.add_argument("--n_bins_prog", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, encoding="utf-8")
    # defensive repairs
    df = df.rename(columns={"abs_err":"abs_error","signed_err":"signed_error"})
    if "abs_error" not in df.columns:
        df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    if "signed_error" not in df.columns:
        df["signed_error"] = (df["y_pred"] - df["y_true"])
    if "group" in df.columns:
        df["group"] = df["group"].astype(str).str.replace("â€“","–", regex=False)

    # Use bins based on source-test y_true quantiles for comparability
    src = df[df["group"].astype(str).str.contains("source-test", na=False)]
    y_ref = src["y_true"].to_numpy()
    # guard: if source-test missing, fall back to all
    if len(y_ref) < 1000:
        y_ref = df["y_true"].to_numpy()

    qs = np.linspace(0, 1, args.n_bins_rt + 1)
    edges_rt = np.unique(np.quantile(y_ref, qs))
    if len(edges_rt) < 3:  # degenerate
        edges_rt = np.linspace(df["y_true"].min(), df["y_true"].max(), args.n_bins_rt + 1)

    edges_prog = np.linspace(0, 1, args.n_bins_prog + 1)

    def bin_mid(edges):
        return (edges[:-1] + edges[1:]) / 2.0

    df["rt_bin"] = np.digitize(df["y_true"], edges_rt[1:-1], right=True)
    df["prog_bin"] = np.digitize(df["progress"], edges_prog[1:-1], right=True)

    # Fig4a: MAE vs true remaining time
    g1 = (df.groupby(["group","rt_bin"])
            .agg(mae=("abs_error","mean"), n=("abs_error","size"))
            .reset_index())
    g1["rt_mid"] = bin_mid(edges_rt)[g1["rt_bin"].to_numpy()]

    # Fig4b: mean signed error vs true remaining time
    g2 = (df.groupby(["group","rt_bin"])
            .agg(mean_signed=("signed_error","mean"), n=("signed_error","size"))
            .reset_index())
    g2["rt_mid"] = bin_mid(edges_rt)[g2["rt_bin"].to_numpy()]

    # Fig4c: MAE vs progress
    g3 = (df.groupby(["group","prog_bin"])
            .agg(mae=("abs_error","mean"), n=("abs_error","size"))
            .reset_index())
    g3["prog_mid"] = bin_mid(edges_prog)[g3["prog_bin"].to_numpy()]

    out_dir = args.out_dir_csv
    import os
    os.makedirs(out_dir, exist_ok=True)
    g1.to_csv(os.path.join(out_dir, "fig4a_mae_vs_true_rt.csv"), index=False)
    g2.to_csv(os.path.join(out_dir, "fig4b_signed_vs_true_rt.csv"), index=False)
    g3.to_csv(os.path.join(out_dir, "fig4c_mae_vs_progress.csv"), index=False)

    # Plot
    def plot_lines(ax, data, x, y, ylabel, title):
        for grp in GROUP_ORDER:
            sub = data[data["group"] == grp]
            if len(sub) == 0:
                continue
            ax.plot(sub[x], sub[y], marker="o", linewidth=1.5, label=grp)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    plot_lines(axes[0], g1, "rt_mid", "mae", "MAE (s)", "(a) MAE vs. true remaining time")
    plot_lines(axes[1], g2, "rt_mid", "mean_signed", "Mean signed error (s)", "(b) Signed error vs. true remaining time")
    plot_lines(axes[2], g3, "prog_mid", "mae", "MAE (s)", "(c) MAE vs. prefix progress")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(args.out_pdf, bbox_inches="tight")
    plt.close()
    print("Wrote:", args.out_pdf)
    print("CSV points in:", out_dir)

if __name__ == "__main__":
    main()
