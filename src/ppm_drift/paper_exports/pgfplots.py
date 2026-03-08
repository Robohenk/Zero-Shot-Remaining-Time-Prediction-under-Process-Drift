from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.case_start_coverage import compute_case_start_coverage
from ..analysis.drift_scores import build_reference_prefix_table, compute_drift_scores, spearman_rho
from ..analysis.error_anatomy import binned_error_curves
from ..constants import ALT_EXPS, SOURCE_EXPS, TARGET_EXPS, label_group
from ..evaluation.metrics import run_metrics_from_predictions, summarize_groups
from ..utils.io import ensure_dir
from ..visualization.style import set_paper_style


def _group_style(group: str) -> tuple[str, str]:
    if group == "target":
        return "targetBox", "mark=*"
    if group == "source-test":
        return "sourceBox", "mark=square*"
    return "altBox", "mark=diamond*"


def _normalize_start_event_group(group: str) -> str:
    if group in {"source-train", "source-test", "source"}:
        return "source"
    if group == "target":
        return "target"
    if group == "alt":
        return "alt"
    return group


def _latex_num(x: float) -> str:
    if pd.isna(x):
        return "nan"
    if abs(x) >= 1000:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if abs(x) >= 10:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.10f}".rstrip("0").rstrip(".")


def _quartiles(vals: pd.Series) -> dict:
    vals = vals.dropna().astype(float).sort_values().to_numpy()
    q1 = np.quantile(vals, 0.25)
    med = np.quantile(vals, 0.5)
    q3 = np.quantile(vals, 0.75)
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    non_out = vals[(vals >= lo_fence) & (vals <= hi_fence)]
    low_wh = non_out.min() if len(non_out) else vals.min()
    hi_wh = non_out.max() if len(non_out) else vals.max()
    outliers = vals[(vals < lo_fence) | (vals > hi_fence)]
    return {
        "lower_whisker": low_wh,
        "lower_quartile": q1,
        "median": med,
        "upper_quartile": q3,
        "upper_whisker": hi_wh,
        "outliers": outliers,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_start_event_artifacts(runs, source_test_runs: set[int], out_dir: str | Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    per_run, _ = compute_case_start_coverage(runs, source_test_runs)
    per_run.to_csv(figures_dir / "start_event_per_run.csv", index=False)

    lines = [
        "% Auto-generated Tukey boxplots (whiskers=1.5*IQR) + outliers",
        "% Requires: \\usepgfplotslibrary{statistics}",
        "",
    ]
    outlier_lines = {"target": [], "source": [], "alt": []}

    for exp in range(1, 28):
        gdf = per_run.loc[per_run["exp"] == exp, ["start_event_rate", "group"]].copy()
        if gdf.empty:
            continue

        raw_group = gdf["group"].iloc[0]
        group = _normalize_start_event_group(raw_group)

        style = "targetBox" if group == "target" else ("sourceBox" if group == "source" else "altBox")
        stats = _quartiles(gdf["start_event_rate"])
        group_comment = "target" if exp in TARGET_EXPS else ("source" if exp in SOURCE_EXPS else "alt")

        lines.extend(
            [
                f"% Exp{exp} ({group_comment})",
                r"\addplot+[",
                f"  springerBoxBase, {style}, boxplot prepared={{",
                f"    draw position={exp},",
                f"    lower whisker={_latex_num(stats['lower_whisker'])},",
                f"    lower quartile={_latex_num(stats['lower_quartile'])},",
                f"    median={_latex_num(stats['median'])},",
                f"    upper quartile={_latex_num(stats['upper_quartile'])},",
                f"    upper whisker={_latex_num(stats['upper_whisker'])}",
                "  },",
                "  solid",
                r"] coordinates {(0,0)};",
                "",
            ]
        )

        if group not in outlier_lines:
            outlier_lines[group] = []
        for y in stats["outliers"]:
            outlier_lines[group].append(f"  ({exp},{_latex_num(float(y))})")

    lines.append("% ---- Outliers (Tukey) ----")
    for group, mark in [("target", "*"), ("source", "square*"), ("alt", "triangle*")]:
        lines.append(f"% Outliers: {group}")
        lines.append(rf"\addplot+[only marks, mark={mark}, mark size=0.9pt] coordinates {{")
        lines.extend(outlier_lines[group] if outlier_lines[group] else ["% none"])
        lines.append(r"};")
        lines.append("")
    lines.append("% End of snippet.")
    _write_text(figures_dir / "start_event_boxplot_snippet.tex", "\n".join(lines))

    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    positions = []
    data = []
    for exp in range(1, 28):
        vals = per_run.loc[per_run["exp"] == exp, "start_event_rate"].to_numpy()
        positions.append(exp)
        data.append(vals)
    ax.boxplot(data, positions=positions, widths=0.35, whis=1.5, patch_artist=False, manage_ticks=False)
    ax.set_xlim(0.5, 27.5)
    ax.set_ylim(0.96, 0.995)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Start event rate")
    ax.set_xticks(list(range(1, 28)))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axvline(9.5, linestyle="--", linewidth=1)
    ax.axvline(18.5, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(figures_dir / "start_event_rate_tukey_boxplot.pdf")
    fig.savefig(figures_dir / "start_event_rate_tukey_boxplot.jpg", dpi=220)
    plt.close(fig)
    return per_run


def _build_boxplot_tex(metric_df: pd.DataFrame, metric: str) -> str:
    lines = []
    for pos, group in [(1, "source-test"), (2, "target"), (3, "alt")]:
        gdf = metric_df.loc[metric_df["group"] == group, metric]
        stats = _quartiles(gdf)
        style, mark = _group_style(group)
        coords = " ".join(f"({pos},{_latex_num(float(v))})" for v in stats["outliers"])
        lines.extend(
            [
                r"\addplot+[",
                f"  {style},",
                "  boxplot prepared={",
                f"    draw position={pos},",
                f"    lower whisker={_latex_num(stats['lower_whisker'])},",
                f"    lower quartile={_latex_num(stats['lower_quartile'])},",
                f"    median={_latex_num(stats['median'])},",
                f"    upper quartile={_latex_num(stats['upper_quartile'])},",
                f"    upper whisker={_latex_num(stats['upper_whisker'])}",
                "  },",
                f"  {mark},",
                "  mark size=1.8pt,",
                "  mark options={draw=black, line width=0.45pt, fill=black, fill opacity=0.35}",
                rf"] coordinates {{{coords}}};",
                "",
            ]
        )
    return "\n".join(lines)


def export_run_metric_artifacts(pred_dir: str | Path, source_test_runs: set[int], out_dir: str | Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    run_metrics = run_metrics_from_predictions(pred_dir, source_test_runs)
    run_metrics.to_csv(figures_dir / "run_metrics.csv", index=False)
    _write_text(figures_dir / "fig2_boxplot_mae.tex", _build_boxplot_tex(run_metrics, "mae"))
    _write_text(figures_dir / "fig2_boxplot_rmse.tex", _build_boxplot_tex(run_metrics, "rmse"))

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    for ax, metric, title in zip(axes, ["mae", "rmse"], ["(a) Run-level MAE", "(b) Run-level RMSE"]):
        data = [run_metrics.loc[run_metrics["group"] == g, metric].to_numpy() for g in ["source-test", "target", "alt"]]
        ax.boxplot(data, labels=["Source-test", "Target", "Alt"], whis=1.5)
        ax.set_title(title)
        ax.set_ylabel(f"{metric.upper()} (s)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_boxplots_mae_rmse.pdf")
    fig.savefig(figures_dir / "fig2_boxplots_mae_rmse.jpg", dpi=220)
    plt.close(fig)
    return run_metrics


def export_run_size_table(run_metrics: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    rows = []
    labels = [("source-test", "Source-test (18 runs)"), ("target", "Target (180 runs)"), ("alt", "Alt (180 runs)")]
    for g, label in labels:
        gdf = run_metrics.loc[run_metrics["group"] == g]
        rows.append(
            {
                "group": g,
                "label": label,
                "prefix_median": int(round(gdf["n_prefixes"].median())),
                "prefix_q1": int(round(gdf["n_prefixes"].quantile(0.25))),
                "prefix_q3": int(round(gdf["n_prefixes"].quantile(0.75))),
                "cases_median": int(round(gdf["n_cases"].median())),
                "cases_q1": int(round(gdf["n_cases"].quantile(0.25))),
                "cases_q3": int(round(gdf["n_cases"].quantile(0.75))),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(figures_dir / "run_size_summary.csv", index=False)
    tex = [
        r"\begin{table}[t!]",
        r"\centering",
        r"\caption{Evaluation run size with median [first quartile (Q1), third quartile (Q3)] across runs per group.}",
        r"\label{tab:run_size_summary}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"\textbf{Group} & \textbf{Prefixes/run} & \textbf{Cases/run} \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        row = f"{r['label']} & {r['prefix_median']:,} [{r['prefix_q1']:,}, {r['prefix_q3']:,}] & {r['cases_median']:,} [{r['cases_q1']:,}, {r['cases_q3']:,}] \\".replace(",", "{,}")
        tex.append(row)
    tex.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    _write_text(figures_dir / "run_size_summary.tex", "\n".join(tex))
    return df


def export_group_summary_artifacts(run_metrics: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    group_summary = summarize_groups(run_metrics)
    order = {"source-test": 1, "target": 2, "alt": 3}
    group_summary["x"] = group_summary["group"].map(order)
    group_summary = group_summary.sort_values("x")
    lines = ["% Auto-generated by export_paper_artifacts.py", "% Columns: x y ylo yhi  (ylo/yhi are CI bounds)", r"\pgfplotstableread{", "x y ylo yhi"]
    for _, r in group_summary.iterrows():
        lines.append(f"{int(r['x'])} {_latex_num(r['mae_mean'])} {_latex_num(r['mae_ci_lo'])} {_latex_num(r['mae_ci_hi'])}")
    lines.append(r"}{\FigThreeMAE}")
    lines.append("")
    lines.append(r"\pgfplotstableread{")
    lines.append("x y ylo yhi")
    for _, r in group_summary.iterrows():
        lines.append(f"{int(r['x'])} {_latex_num(r['rmse_mean'])} {_latex_num(r['rmse_ci_lo'])} {_latex_num(r['rmse_ci_hi'])}")
    lines.append(r"}{\FigThreeRMSE}")
    _write_text(figures_dir / "fig3_severity_ladder_pgfplots_data.tex", "\n".join(lines))
    group_summary.to_csv(figures_dir / "fig3_group_summary.csv", index=False)

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    labels = ["Source-test", "Target", "Alt"]
    x = np.arange(3)
    for ax, metric, lo, hi, title in [
        (axes[0], "mae_mean", "mae_ci_lo", "mae_ci_hi", "(a) Mean MAE"),
        (axes[1], "rmse_mean", "rmse_ci_lo", "rmse_ci_hi", "(b) Mean RMSE"),
    ]:
        y = group_summary[metric].to_numpy()
        yerr = np.vstack([y - group_summary[lo].to_numpy(), group_summary[hi].to_numpy() - y])
        ax.bar(x, y, yerr=yerr, capsize=4)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel(metric.split("_")[0].upper() + " (s)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_severity_ladder.pdf")
    fig.savefig(figures_dir / "fig3_severity_ladder.jpg", dpi=220)
    plt.close(fig)
    return group_summary


def export_fig4_artifacts(pred_dir: str | Path, source_test_runs: set[int], out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    rt, progress = binned_error_curves(pred_dir, source_test_runs)
    rt.to_csv(figures_dir / "fig4_remaining_time_curves.csv", index=False)
    progress.to_csv(figures_dir / "fig4_progress_curves.csv", index=False)

    set_paper_style()
    for metric, fname, ylabel in [("mae", "fig4a_mae_vs_true_rt", "MAE (s)"), ("signed", "fig4b_signed_vs_true_rt", "Mean signed error (s)")]:
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        for group, gdf in rt.groupby("group"):
            ax.plot(gdf["x"], gdf[metric], marker="o", label=group)
        ax.set_xlabel("True remaining time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{fname}.pdf")
        fig.savefig(figures_dir / f"{fname}.jpg", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for group, gdf in progress.groupby("group"):
        ax.plot(gdf["x"], gdf["mae"], marker="o", label=group)
    ax.set_xlabel("Prefix progress")
    ax.set_ylabel("MAE (s)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4c_mae_vs_progress.pdf")
    fig.savefig(figures_dir / "fig4c_mae_vs_progress.jpg", dpi=220)
    plt.close(fig)
    return rt, progress


def export_fig5_artifacts(pred_dir: str | Path, runs, source_test_runs: set[int], out_dir: str | Path) -> pd.DataFrame:
    figures_dir = ensure_dir(Path(out_dir) / "figures")
    points_dir = ensure_dir(figures_dir / "fig5_points")
    source_train_ids = [rid for rid in runs if rid.exp in SOURCE_EXPS and rid.run not in source_test_runs]
    eval_ids = [rid for rid in runs if label_group(rid.exp, rid.run, source_test_runs) in {"source-test", "target", "alt"}]
    ref = build_reference_prefix_table(source_train_ids, runs)
    drift = compute_drift_scores(ref, eval_ids, runs, source_test_runs, include_label_shift=True)
    metrics = run_metrics_from_predictions(pred_dir, source_test_runs)[["run_id", "mae"]]
    out = drift.merge(metrics, on="run_id", how="left")
    out.to_csv(points_dir / "fig5_points_all.csv", index=False)
    for group, fname in [("source-test", "fig5_points_source_test.csv"), ("target", "fig5_points_target.csv"), ("alt", "fig5_points_alt.csv")]:
        out.loc[out["group"] == group, ["drift_score", "mae"]].to_csv(points_dir / fname, index=False)
    rho = spearman_rho(out["drift_score"].to_numpy(), out["mae"].to_numpy())
    _write_text(points_dir / "fig5_stats.tex", f"\\def\\FigFiveSpearman{{{_latex_num(rho)}}}")
    _write_text(
        points_dir / "fig5_tables.tex",
        "\n".join(
            [
                "% Auto-generated by export_paper_artifacts.py",
                r"\pgfplotstableread[col sep=comma]{figures/fig5_points/fig5_points_source_test.csv}\FigFiveSource",
                r"\pgfplotstableread[col sep=comma]{figures/fig5_points/fig5_points_target.csv}\FigFiveTarget",
                r"\pgfplotstableread[col sep=comma]{figures/fig5_points/fig5_points_alt.csv}\FigFiveAlt",
                r"\input{figures/fig5_points/fig5_stats.tex}",
            ]
        ),
    )

    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    markers = {"source-test": "o", "target": "s", "alt": "D"}
    labels = {"source-test": "Source-test", "target": "Target", "alt": "Alt"}
    for group, gdf in out.groupby("group"):
        ax.scatter(gdf["drift_score"], gdf["mae"], s=18, marker=markers[group], label=labels[group], alpha=0.8)
    ax.set_xlabel("Drift score (avg. PSI/JS components)")
    ax.set_ylabel("MAE (s)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_drift_vs_error.pdf")
    fig.savefig(figures_dir / "fig5_drift_vs_error.jpg", dpi=220)
    plt.close(fig)
    return out