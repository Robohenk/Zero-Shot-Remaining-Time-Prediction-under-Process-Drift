#!/usr/bin/env python3
"""Run the full reproducibility pipeline from raw event logs to manuscript-ready artifacts."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / 'src'))
from ppm_drift.utils.io import ensure_dir
from ppm_drift.utils.logging_utils import setup_logger, timed_stage, update_status


def run_cmd(cmd, logger):
    logger.info('RUN %s', ' '.join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run the full reproducibility pipeline from raw logs to manuscript assets.')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--work_root', default='reproducibility_run')
    ap.add_argument('--models', nargs='+', default=['lstm', 'transformer', 'tft'])
    ap.add_argument('--run_benchmarks', action='store_true')
    ap.add_argument('--skip_existing', action='store_true', help='Skip stages whose main outputs already exist.')
    ap.add_argument('--start_at_stage', default='validate', choices=['validate', 'models', 'aggregate', 'benchmarks'], help='Start at a later pipeline stage for resumed runs.')
    ap.add_argument('--benchmark_models', nargs='+', default=None)
    ap.add_argument('--benchmark_epochs', nargs='+', type=int, default=[1,2,5,10,20])
    ap.add_argument('--benchmark_seeds', nargs='+', type=int, default=[7])
    args = ap.parse_args()

    work_root = ensure_dir(args.work_root)
    stage_order = {'validate': 0, 'models': 1, 'aggregate': 2, 'benchmarks': 3}
    logs_dir = ensure_dir(work_root / 'logs')
    logger = setup_logger(logs_dir, filename='pipeline.log')
    predictions_root = ensure_dir(work_root / 'predictions')
    artifacts_root = ensure_dir(work_root / 'artifacts')
    benchmarks_root = ensure_dir(work_root / 'benchmarks')
    ensure_dir(work_root / 'dataset_checks')
    ensure_dir(work_root / 'figures')
    update_status(work_root, 'init', 'running', {'models': args.models})

    if stage_order[args.start_at_stage] <= stage_order['validate']:
        dataset_summary = work_root / 'dataset_checks' / 'dataset_validation_summary.json'
        if not (args.skip_existing and dataset_summary.exists()):
            with timed_stage(logger, logs_dir, work_root, 'validate_dataset'):
                run_cmd([sys.executable, str(ROOT / 'scripts' / 'validate_dataset.py'), '--data_root', args.data_root, '--out_dir', str(work_root / 'dataset_checks')], logger)
        else:
            logger.info('Skipping validate_dataset because %s already exists', dataset_summary)

        start_event_tex = work_root / 'figures' / 'start_event_boxplot_snippet.tex'
        if not (args.skip_existing and start_event_tex.exists()):
            with timed_stage(logger, logs_dir, work_root, 'case_start_coverage'):
                run_cmd([sys.executable, str(ROOT / 'scripts' / 'analyze_case_start_coverage.py'), '--data_root', args.data_root, '--out_dir', str(work_root)], logger)
        else:
            logger.info('Skipping case_start_coverage because %s already exists', start_event_tex)

    if stage_order[args.start_at_stage] <= stage_order['models']:
        for model in args.models:
            with timed_stage(logger, logs_dir, work_root, f'train_and_predict_{model}', {'current_model': model}):
                model_output_root = predictions_root / model
                train_cmd = [
                    sys.executable, str(ROOT / 'scripts' / 'train_model.py'),
                    '--data_root', args.data_root,
                    '--output_root', str(model_output_root),
                    '--model_name', model,
                ]
                if args.skip_existing:
                    train_cmd.append('--skip_existing')
                run_cmd(train_cmd, logger)

            with timed_stage(logger, logs_dir, work_root, f'export_paper_artifacts_{model}', {'current_model': model}):
                pred_dir = model_output_root / model / 'predictions'
                out_dir = artifacts_root / model
                export_cmd = [
                    sys.executable, str(ROOT / 'scripts' / 'export_paper_artifacts.py'),
                    '--data_root', args.data_root,
                    '--pred_dir', str(pred_dir),
                    '--out_dir', str(out_dir),
                ]
                if args.skip_existing:
                    export_cmd.append('--skip_existing')
                run_cmd(export_cmd, logger)
                plot_cmd = [
                    sys.executable, str(ROOT / 'scripts' / 'plot_representative_runs.py'),
                    '--pred_dir', str(pred_dir),
                    '--out_dir', str(out_dir),
                ]
                if args.skip_existing:
                    plot_cmd.append('--skip_existing')
                run_cmd(plot_cmd, logger)
                inv_cmd = [
                    sys.executable, str(ROOT / 'scripts' / 'summarize_prediction_inventory.py'),
                    '--pred_dir', str(pred_dir),
                    '--out_dir', str(out_dir),
                ]
                if args.skip_existing:
                    inv_cmd.append('--skip_existing')
                run_cmd(inv_cmd, logger)


    if stage_order[args.start_at_stage] <= stage_order['aggregate']:
        with timed_stage(logger, logs_dir, work_root, 'export_cross_model_tables'):
            run_cmd([
                sys.executable, str(ROOT / 'scripts' / 'export_model_summary_tables.py'),
                '--predictions_root', str(predictions_root),
                '--work_root', str(work_root),
            ], logger)

    if args.run_benchmarks and stage_order[args.start_at_stage] <= stage_order['benchmarks']:
        bench_models = args.benchmark_models or args.models
        for model in bench_models:
            with timed_stage(logger, logs_dir, work_root, f'benchmark_{model}', {'current_model': model}):
                bench_cmd = [
                    sys.executable, str(ROOT / 'scripts' / 'run_benchmark.py'),
                    '--data_root', args.data_root,
                    '--output_root', str(benchmarks_root),
                    '--model_name', model,
                    '--epochs', *map(str, args.benchmark_epochs),
                    '--seeds', *map(str, args.benchmark_seeds),
                ]
                if args.skip_existing:
                    bench_cmd.append('--skip_existing')
                run_cmd(bench_cmd, logger)

    update_status(work_root, 'pipeline', 'completed')
    logger.info('Full pipeline completed successfully.')


if __name__ == '__main__':
    main()
