#!/usr/bin/env python3
r"""
Convert Fig4 CSV outputs into PGFPlots-friendly inputs.

Expected inputs (based on your snippets):
- Fig4a: columns include group, rt_mid, mae (and optionally n)
- Fig4b: columns include group, rt_mid, mean_signed (and optionally n)
- Fig4c: columns include group, prog_mid, mae (and optionally n)

Outputs (per group):
- out_dir/fig4a_<slug>.csv with columns: x,y,n
- out_dir/fig4b_<slug>.csv with columns: x,y,n
- out_dir/fig4c_<slug>.csv with columns: x,y,n
- out_dir/fig4_pgf_tables.tex (helper file with \pgfplotstableread)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_flex(path: Path) -> pd.DataFrame:
    # Try utf-8 first, then fall back to common Windows encodings
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="latin1")


def norm_group(s: str) -> str:
    s = str(s)
    # Fix mojibake dash
    s = s.replace("â€“", "–").replace("â€”", "—")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def group_slug(g: str) -> str:
    gl = g.lower()
    # canonical slugs
    if "source" in gl and "test" in gl:
        return "source_test"
    if "source" in gl and "train" in gl:
        return "source_train"
    if "target" in gl:
        return "target"
    if "alt" in gl:
        return "alt"
    # fallback slugify
    x = re.sub(r"[^\w]+", "_", gl).strip("_")
    return x[:60] if x else "group"


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


def write_group_files(df: pd.DataFrame, prefix: str, x_col: str, y_col: str, out_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Writes per-group CSVs:
      fig4a_alt.csv etc with columns x,y,n (n optional -> filled with NA)
    Returns list of (panel, slug, path) for latex helper generation.
    """
    df = df.copy()
    df["group"] = df["group"].map(norm_group)

    n_col = None
    if "n" in [c.lower() for c in df.columns]:
        n_col = pick_col(df, ["n"])

    written = []
    for g, sub in df.groupby("group"):
        slug = group_slug(g)

        out = pd.DataFrame({
            "x": pd.to_numeric(sub[x_col], errors="coerce"),
            "y": pd.to_numeric(sub[y_col], errors="coerce"),
        })
        if n_col is not None:
            out["n"] = pd.to_numeric(sub[n_col], errors="coerce")
        else:
            out["n"] = pd.NA

        out = out.dropna(subset=["x", "y"]).sort_values("x")

        out_path = out_dir / f"{prefix}_{slug}.csv"
        out.to_csv(out_path, index=False)
        written.append((prefix, slug, out_path))

    return written


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_a", required=True, help="Fig4a CSV (MAE vs true remaining time)")
    ap.add_argument("--in_b", required=True, help="Fig4b CSV (signed error vs true remaining time)")
    ap.add_argument("--in_c", required=True, help="Fig4c CSV (MAE vs progress)")
    ap.add_argument("--out_dir", required=True, help="Output directory for PGFPlots inputs")

    args = ap.parse_args()

    in_a = Path(args.in_a)
    in_b = Path(args.in_b)
    in_c = Path(args.in_c)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Read
    a = read_csv_flex(in_a)
    b = read_csv_flex(in_b)
    c = read_csv_flex(in_c)

    # Normalize group column name if needed
    for df, name in [(a, "Fig4a"), (b, "Fig4b"), (c, "Fig4c")]:
        if "group" not in df.columns:
            # tolerate "Group"
            gcol = None
            for col in df.columns:
                if col.lower() == "group":
                    gcol = col
                    break
            if gcol is None:
                raise KeyError(f"{name}: missing 'group' column. Columns: {list(df.columns)}")
            df.rename(columns={gcol: "group"}, inplace=True)

    # Pick x/y columns per your schema (with fallbacks)
    a_x = pick_col(a, ["rt_mid", "y_true_mid", "y_mid", "rt_center"])
    a_y = pick_col(a, ["mae", "MAE", "abs_err_mean"])
    b_x = pick_col(b, ["rt_mid", "y_true_mid", "y_mid", "rt_center"])
    b_y = pick_col(b, ["mean_signed", "signed_err_mean", "signed_mean", "mean_signed_error"])
    c_x = pick_col(c, ["prog_mid", "progress_mid", "progress_center"])
    c_y = pick_col(c, ["mae", "MAE", "abs_err_mean"])

    written = []
    written += write_group_files(a, "fig4a", a_x, a_y, out_dir)
    written += write_group_files(b, "fig4b", b_x, b_y, out_dir)
    written += write_group_files(c, "fig4c", c_x, c_y, out_dir)

    # Write a small helper tex file with pgfplotstable reads
    tex_path = out_dir / "fig4_pgf_tables.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by make_pgfplots_fig4_inputs.py\n")
        f.write("% Usage: \\input{figures/fig4_points/fig4_pgf_tables.tex}\n\n")
        for prefix, slug, path in written:
            macro = f"\\{prefix}_{slug}"
            rel = path.as_posix()
            f.write(f"\\pgfplotstableread[col sep=comma]{{{rel}}}{macro}\n")
        f.write("\n")

    print(f"[OK] Wrote {len(written)} per-group CSVs to: {out_dir}")
    print(f"[OK] Wrote helper LaTeX table loader: {tex_path}")


if __name__ == "__main__":
    main()
