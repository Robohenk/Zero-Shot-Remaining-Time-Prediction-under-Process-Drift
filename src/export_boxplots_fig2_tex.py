#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

GROUP_ORDER = [
    "source-test (Exp10–18)",
    "target (Exp1–9)",
    "alt (Exp19–27)",
]

BOX_STYLE = {
    "source-test (Exp10–18)": "sourceBox",
    "target (Exp1–9)": "targetBox",
    "alt (Exp19–27)": "altBox",
}
OUT_STYLE = {
    "source-test (Exp10–18)": "sourceOutliers",
    "target (Exp1–9)": "targetOutliers",
    "alt (Exp19–27)": "altOutliers",
}

def fix_group_encoding(s: pd.Series) -> pd.Series:
    # Fix the common CSV en-dash mojibake
    return s.astype(str).str.replace("â€“", "–", regex=False)

def tukey_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return dict(lw=np.nan, q1=np.nan, med=np.nan, q3=np.nan, uw=np.nan, outliers=[])

    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr

    inliers = x[(x >= lo_fence) & (x <= hi_fence)]
    lw = float(inliers.min()) if inliers.size else float(x.min())
    uw = float(inliers.max()) if inliers.size else float(x.max())

    outliers = np.sort(x[(x < lo_fence) | (x > hi_fence)]).tolist()
    return dict(lw=lw, q1=float(q1), med=float(med), q3=float(q3), uw=uw, outliers=outliers)

def write_boxplot_snippet(stats_by_group: dict, out_path: Path, metric: str):
    lines = []
    for pos, g in enumerate(GROUP_ORDER, start=1):
        st = stats_by_group.get(g, None)
        if st is None:
            raise RuntimeError(f"Missing group '{g}' in run-metrics CSV. Available: {list(stats_by_group)}")

        box_style = BOX_STYLE[g]
        out_style = OUT_STYLE[g]

        out_coords = " ".join([f"(0,{v:.6g})" for v in st["outliers"]])

        lines.append(
            r"\addplot+[%s, boxplot/every outlier/.style={%s}, boxplot prepared={"
            r"lower whisker=%.6g, lower quartile=%.6g, median=%.6g, upper quartile=%.6g, upper whisker=%.6g}, "
            r"boxplot prepared/draw position=%d] coordinates {%s};"
            % (box_style, out_style, st["lw"], st["q1"], st["med"], st["q3"], st["uw"], pos, out_coords)
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} for metric={metric}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_run_metrics", required=True, help="Path to fig2_run_metrics.csv")
    ap.add_argument("--out_dir", required=True, help="Where to write .tex snippets")
    args = ap.parse_args()

    in_path = Path(args.in_run_metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, engine="python")
    if "group" not in df.columns:
        raise RuntimeError("Expected 'group' column in run-metrics CSV.")
    df["group"] = fix_group_encoding(df["group"])

    # Keep only the three evaluation groups
    df = df[df["group"].isin(GROUP_ORDER)].copy()

    if not {"MAE", "RMSE"}.issubset(df.columns):
        raise RuntimeError("Expected MAE and RMSE columns in fig2_run_metrics.csv")

    stats_mae = {g: tukey_stats(df.loc[df["group"] == g, "MAE"].values) for g in GROUP_ORDER}
    stats_rmse = {g: tukey_stats(df.loc[df["group"] == g, "RMSE"].values) for g in GROUP_ORDER}

    write_boxplot_snippet(stats_mae, out_dir / "fig2_boxplot_mae.tex", metric="MAE")
    write_boxplot_snippet(stats_rmse, out_dir / "fig2_boxplot_rmse.tex", metric="RMSE")

if __name__ == "__main__":
    main()
