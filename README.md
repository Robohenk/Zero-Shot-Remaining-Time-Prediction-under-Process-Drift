> [!WARNING]
> **Temporary Repository** — This repository was created to support the review process of the corresponding scientific manuscript. Upon acceptance, the codebase will be refactored and released as a permanent, fully documented repository with a DOI (e.g., via Zenodo). Until then, contents, structure, and documentation may change without notice.

# Zero-shot Remaining Time Prediction under Resource Drift

This repository contains the code, paper sources, and exported plot points for the manuscript **"Zero-Shot Remaining-Time Prediction under Process Drift."** It supports reproducing the figures and analyses reported in the manuscript.

## Dataset convention

The code expects **540 raw tab-separated files** named exactly like:

- `Exp1Run1.txt`
- `Exp16Run7.txt`
- ...
- `Exp27Run20.txt`

Place them in `data/` inside the repository, or point `--data_root` to the folder that contains them.

## Main one-command pipeline

From the repository root:

```bash
python scripts/run_full_pipeline.py --data_root .\data --work_root .\reproducibility_run --models lstm transformer tft --run_benchmarks
```

This runs:

1. dataset validation
2. case-start coverage / start-event artifacts
3. training and prediction for each requested model
4. manuscript-ready artifact export (LaTeX snippets/tables + PDF/JPG figures)
5. representative diagnostic scatter plots
6. prediction inventory summaries
7. compact cross-model summary tables (accuracy + compute cost)
8. optional epoch-sensitivity benchmarks

## Checking progress live

In another console while the pipeline is running:

```bash
python scripts/check_progress.py --work_root .\reproducibility_run --watch
```

This shows:

- current stage and status
- current model (if applicable)
- elapsed runtime of the currently active stage
- counts of generated files in `dataset_checks`, `logs`, `predictions`, `artifacts`, `figures`, `benchmarks`
- recent completed timings and cumulative timed compute by model
- recent lines from `logs/pipeline.log`

## Important logs and timing files

The pipeline writes:

- `reproducibility_run/logs/pipeline.log`
- `reproducibility_run/logs/train.log`
- `reproducibility_run/logs/benchmark.log`
- `reproducibility_run/logs/timings.jsonl`
- per-script logs inside artifact/model folders

`timings.jsonl` contains machine-readable elapsed times for heavy stages such as encoding, model fit, prediction export, artifact export, and benchmarking.

## Running individual steps

### Validate the dataset

```bash
python scripts/validate_dataset.py --data_root .\data --out_dir .\reproducibility_run\dataset_checks
```

### Train one model

```bash
python scripts/train_model.py --data_root .\data --output_root .\reproducibility_run\predictions\lstm --model_name lstm
```

### Export manuscript assets for one model

```bash
python scripts/export_paper_artifacts.py --data_root .\data --pred_dir .\reproducibility_run\predictions\lstm\lstm\predictions --out_dir .\reproducibility_run\artifacts\lstm
python scripts/plot_representative_runs.py --pred_dir .\reproducibility_run\predictions\lstm\lstm\predictions --out_dir .\reproducibility_run\artifacts\lstm
```

### Run epoch sensitivity benchmarks

```bash
python scripts/run_benchmark.py --data_root .\data --output_root .\reproducibility_run\benchmarks --model_name lstm --epochs 1 2 5 10 20 --seeds 7 11 13
```

## Manuscript-shaped outputs

The artifact export writes files under `.../artifacts/<model>/figures/` that match the paper structure, including:

- `start_event_boxplot_snippet.tex`
- `start_event_rate_tukey_boxplot.pdf` and `.jpg`
- `run_size_summary.tex`
- `fig2_boxplot_mae.tex`
- `fig2_boxplot_rmse.tex`
- `fig2_boxplots_mae_rmse.pdf` and `.jpg`
- `fig3_severity_ladder_pgfplots_data.tex`
- `fig3_severity_ladder.pdf` and `.jpg`
- `error_diag_source_test_bestmedian.pdf` / `.jpg`
- `error_diag_target_bestmedian.pdf` / `.jpg`
- `error_diag_source_test_worstMAE.pdf` / `.jpg`
- `error_diag_target_worstMAE.pdf` / `.jpg`
- `fig4a_mae_vs_true_rt.pdf` / `.jpg`
- `fig4b_signed_vs_true_rt.pdf` / `.jpg`
- `fig4c_mae_vs_progress.pdf` / `.jpg`
- `fig5_points/fig5_tables.tex`
- `fig5_points/fig5_points_source_test.csv`
- `fig5_points/fig5_points_target.csv`
- `fig5_points/fig5_points_alt.csv`
- `fig5_points/fig5_stats.tex`
- `fig5_drift_vs_error.pdf` and `.jpg`

## Spyder / IPython note

In Spyder, run commands on **one line**. Do not use shell backslashes for line continuation.

Correct:

```bash
!python scripts/run_full_pipeline.py --data_root ".\\data" --work_root ".\\reproducibility_run" --models lstm transformer tft --run_benchmarks
```

## Caveat about TFT

The included TFT implementation is a compact TFT-style comparator for benchmarking. It is not intended as a claim of a fully canonical literature-faithful TFT reproduction.

## Compact cross-model tables

After the main three-model run finishes, the full pipeline also writes concise manuscript tables to `reproducibility_run/figures/`:

- `model_comparison_compact.tex`
- `model_compute_compact.tex`
- `model_comparison_summary.csv`
- `model_compute_summary.csv`

You can also regenerate them directly with:

```bash
python scripts/export_model_summary_tables.py --predictions_root .\reproducibility_run\predictions --work_root .\reproducibility_run
```

## Running on a Linux server with parallel workers

The repository now includes server-friendly stages so you can follow the same strategy as in your earlier Aurora workflow:

1. run the shared setup once
2. run independent model jobs in parallel
3. optionally run benchmark jobs in parallel
4. aggregate the final tables once

### Stage 1: shared setup

```bash
python scripts/run_server_stage1_setup.py --data_root ./data --work_root ./reproducibility_run --skip_existing
```

### Stage 2: one main-comparison job per model

```bash
python scripts/run_server_stage2_model_job.py --data_root ./data --work_root ./reproducibility_run --model_name lstm --skip_existing
python scripts/run_server_stage2_model_job.py --data_root ./data --work_root ./reproducibility_run --model_name transformer --skip_existing
python scripts/run_server_stage2_model_job.py --data_root ./data --work_root ./reproducibility_run --model_name tft --skip_existing
```

These jobs are independent, so they can be launched in parallel from one shell using `xargs -P`, `nohup`, or a `tmux` session.

### Stage 3: one benchmark job per (model, epoch, seed)

```bash
python scripts/run_server_stage3_benchmark_job.py --data_root ./data --work_root ./reproducibility_run --model_name lstm --epoch 10 --seed 7 --skip_existing
```

This is designed for job-array style parallelism on a server.

### Stage 4: final aggregation

```bash
python scripts/run_server_stage4_aggregate.py --work_root ./reproducibility_run
```

### Auto-generate a launcher script

```bash
python scripts/build_server_launcher.py --data_root ./data --work_root ./reproducibility_run --models lstm transformer tft --benchmark_epochs 1 2 5 10 20 --benchmark_seeds 7 --parallel_models 3 --parallel_benchmarks 8 --output server_launchers.sh
```

Then inspect and run `server_launchers.sh` inside a `tmux` session.

### Resume-safe full pipeline

The monolithic pipeline can now be resumed safely:

```bash
python scripts/run_full_pipeline.py --data_root ./data --work_root ./reproducibility_run --models lstm transformer tft --run_benchmarks --skip_existing
```

You can also restart later stages only:

```bash
python scripts/run_full_pipeline.py --data_root ./data --work_root ./reproducibility_run --models lstm transformer tft --start_at_stage aggregate --skip_existing
```

This mirrors the earlier safe pattern: shared setup once, heavy independent jobs in parallel, aggregation once.

## Dataset

This project uses the simulation event logs described in the paper. If the dataset cannot be redistributed here, place it under a local `data/` directory (or configure the path via script arguments).

The underlying dataset is:

> Bemthuis, R., Mes, M.R.K., Iacob, M.E., & Havinga, P.J.M. (2021). *Data underlying the paper: Using agent-based simulation for emergent behavior detection in cyber-physical systems.* 4TU.ResearchData. CC BY 4.0. [DOI: 10.4121/14743263.v1](https://doi.org/10.4121/14743263.v1)

## Note on AI-Assisted Code Polishing

Parts of the codebase and documentation were *polished* with assistance from **ChatGPT (GPT-5.2)**, primarily for refactoring, clarity, and robustness. Generated changes were reviewed and verified against the intended experimental protocol.

## License

| Component | License |
|---|---|
| Code (`src/`, `tools/`) | [MIT](LICENSE) |
| Dataset (third-party) | CC BY 4.0 — see dataset DOI above |

## Citation

If you use this repository, please cite the manuscript as follows (placeholder until published):

```bibtex
@inproceedings{ivanyi2026zeroshot,
  title     = {Zero-Shot Remaining-Time Prediction under Process Drift},
  author    = {Iványi, Zsombor and Bemthuis, Rob and Monti, Flavia and Mecella, Massimo},
  booktitle = {<Conference Name>},
  year      = {2026},
  note      = {Submitted}
}
```

**Plain text:** Z. Iványi, R. Bemthuis, F. Monti, and M. Mecella, "Zero-Shot Remaining-Time Prediction under Process Drift," in *\<Conference Name\>*, 2026. Submitted.