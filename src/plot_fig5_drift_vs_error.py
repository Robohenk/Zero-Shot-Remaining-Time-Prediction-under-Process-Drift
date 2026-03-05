import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
RUN_RE = re.compile(r"Exp(\d+)Run(\d+)\.parquet$", re.IGNORECASE)

def parse_exp_run(path: Path):
    m = RUN_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def group_from_exp_run(exp: int, run: int):
    if 1 <= exp <= 9:
        return "target (Exp1–9)"
    if 10 <= exp <= 18:
        if run in (19, 20):
            return "source-test (Exp10–18)"
        else:
            return "source-train (Exp10–18)"
    if 19 <= exp <= 27:
        return "alt (Exp19–27)"
    return "unknown"

def safe_prob(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    p = p / p.sum()
    return p

def psi_from_counts(act_counts, exp_counts, eps=1e-6):
    # PSI = sum((p - q) * ln(p / q))
    p = safe_prob(act_counts, eps)
    q = safe_prob(exp_counts, eps)
    return float(np.sum((p - q) * np.log(p / q)))

def js_divergence(p, q, eps=1e-12):
    p = safe_prob(p, eps)
    q = safe_prob(q, eps)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

def spearman_r(x, y):
    # scipy-free Spearman
    x = np.asarray(x)
    y = np.asarray(y)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def get_numeric_bins(ref: pd.Series, n_bins=10):
    ref = ref.dropna().to_numpy()
    if len(ref) < 100:
        # fallback fixed bins
        mn, mx = np.nanmin(ref), np.nanmax(ref)
        return np.linspace(mn, mx, n_bins + 1)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        mn, mx = np.nanmin(ref), np.nanmax(ref)
        edges = np.linspace(mn, mx, n_bins + 1)
    return edges

def binned_counts(values: np.ndarray, edges: np.ndarray):
    # returns counts per bin (len = len(edges)-1)
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.zeros(len(edges) - 1, dtype=float)
    idx = np.digitize(values, edges[1:-1], right=True)
    return np.bincount(idx, minlength=len(edges) - 1).astype(float)

def cat_probs(series: pd.Series, categories: np.ndarray):
    vc = series.value_counts(dropna=False)
    return np.array([vc.get(c, 0) for c in categories], dtype=float)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory with ExpNRunM.parquet files")
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_bins_psi", type=int, default=10)
    ap.add_argument("--include_ytrue_psi", action="store_true",
                    help="Include PSI(y_true) in drift score (offline-only)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    files = sorted([p for p in pred_dir.glob("Exp*Run*.parquet") if parse_exp_run(p) is not None])
    if not files:
        raise FileNotFoundError(f"No Exp*Run*.parquet files found in {pred_dir}")

    # Identify source-training reference files (Exp10–18, runs 1–18)
    ref_files = []
    eval_files = []
    for f in files:
        exp, run = parse_exp_run(f)
        grp = group_from_exp_run(exp, run)
        if grp == "source-train (Exp10–18)":
            ref_files.append(f)
        else:
            eval_files.append(f)

    if not ref_files:
        raise RuntimeError("No reference files found (expected Exp10–18 runs 1–18).")

    # Build reference distributions (we avoid storing the full ref df if possible, but we need bins + categories)
    # We'll concatenate only needed columns for bin edge computation and category vocab.
    num_cols = ["progress", "delta_t", "currentDecayLevel"]
    cat_cols = ["event", "vehicleType"]
    if args.include_ytrue_psi:
        num_cols = num_cols + ["y_true"]

    ref_chunks = []
    for f in ref_files:
        df = pd.read_parquet(f, columns=[c for c in (num_cols + cat_cols) if c is not None])
        ref_chunks.append(df)
    ref = pd.concat(ref_chunks, ignore_index=True)

    # Numeric bin edges from reference
    bin_edges = {col: get_numeric_bins(ref[col], n_bins=args.n_bins_psi) for col in num_cols}

    # Categorical vocab from reference (fixed support for JS)
    cat_vocab = {}
    for col in cat_cols:
        cat_vocab[col] = ref[col].astype(str).fillna("NA").unique()
        cat_vocab[col].sort()

    # Reference hist/prob for numeric/cat
    ref_hist = {col: binned_counts(ref[col].to_numpy(dtype=float), bin_edges[col]) for col in num_cols}
    ref_cat = {col: cat_probs(ref[col].astype(str).fillna("NA"), cat_vocab[col]) for col in cat_cols}

    # Now compute run-level MAE + drift components per evaluation file
    rows = []
    for f in eval_files:
        exp, run = parse_exp_run(f)
        grp = group_from_exp_run(exp, run)

        df = pd.read_parquet(f)  # read full, because we need y_true/y_pred too
        # defensive: compute errors even if abs_err naming differs
        y_true = df["y_true"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=float)
        abs_err = np.abs(y_pred - y_true)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        # PSI components
        psi_vals = {}
        for col in num_cols:
            vals = df[col].to_numpy(dtype=float)
            act_hist = binned_counts(vals, bin_edges[col])
            psi_vals[f"psi_{col}"] = psi_from_counts(act_hist, ref_hist[col])

        # JS components (categorical)
        js_vals = {}
        for col in cat_cols:
            vals = df[col].astype(str).fillna("NA")
            act = cat_probs(vals, cat_vocab[col])
            js_vals[f"js_{col}"] = js_divergence(act, ref_cat[col])

        # Drift scores: label-free and (optional) with y_true
        label_free_components = [psi_vals["psi_progress"], psi_vals["psi_delta_t"], psi_vals["psi_currentDecayLevel"],
                                 js_vals["js_event"], js_vals["js_vehicleType"]]
        drift_score_label_free = float(np.mean(label_free_components))

        if args.include_ytrue_psi:
            drift_score = float(np.mean(label_free_components + [psi_vals["psi_y_true"]]))
        else:
            drift_score = drift_score_label_free

        rows.append({
            "exp": exp,
            "run": run,
            "run_id": f"Exp{exp}Run{run}",
            "group": grp,
            "mae": mae,
            "rmse": rmse,
            "drift_score": drift_score,
            "drift_score_label_free": drift_score_label_free,
            **psi_vals,
            **js_vals,
            "n_prefix_rows": int(len(df)),
        })

    out_df = pd.DataFrame(rows)

    # Correlation (use label-free by default for reporting)
    r = spearman_r(out_df["drift_score_label_free"], out_df["mae"])

    # Export raw points
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print("Wrote points CSV:", out_csv)

    # Plot
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "source-test (Exp10–18)": "#1f77b4",
        "target (Exp1–9)": "#d62728",
        "alt (Exp19–27)": "#2ca02c",
        "unknown": "#7f7f7f",
    }

    plt.figure(figsize=(7.6, 5.2))
    for grp, sub in out_df.groupby("group"):
        plt.scatter(sub["drift_score_label_free"], sub["mae"],
                    s=26, alpha=0.75, label=grp, color=colors.get(grp, "#7f7f7f"))

    plt.xlabel("Drift score (label-free: mean of PSI(progress, delta_t, decay) + JS(event, vehicleType))")
    plt.ylabel("Run-level MAE (s)")
    title = f"Drift magnitude vs MAE (Spearman r = {r:.3f})"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("Wrote figure:", out_pdf)

if __name__ == "__main__":
    main()
