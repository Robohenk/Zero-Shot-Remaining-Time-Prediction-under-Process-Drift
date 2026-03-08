#!/usr/bin/env python3
"""Generate a Bash launcher script for tmux/nohup-style server execution with parallel workers."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a server launcher script for setup, model jobs, benchmark jobs, and final aggregation.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--work_root", default="reproducibility_run")
    ap.add_argument("--models", nargs="+", default=["lstm", "transformer", "tft"])
    ap.add_argument("--benchmark_epochs", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    ap.add_argument("--benchmark_seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--parallel_models", type=int, default=3)
    ap.add_argument("--parallel_benchmarks", type=int, default=8)
    ap.add_argument("--output", default="server_launchers.sh")
    args = ap.parse_args()

    model_lines = " ".join(f'"{m}"' for m in args.models)
    benchmark_rows = [f'"{m} {e} {s}"' for m in args.models for e in args.benchmark_epochs for s in args.benchmark_seeds]
    benchmark_lines = " ".join(benchmark_rows)

    script = f'''#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
DATA_ROOT="{args.data_root}"
WORK_ROOT="{args.work_root}"

# Stage 1: shared setup
python scripts/run_server_stage1_setup.py --data_root "$DATA_ROOT" --work_root "$WORK_ROOT" --skip_existing

# Stage 2: main model jobs
printf "%s\n" {model_lines} \
| xargs -I{{}} -P {args.parallel_models} bash -lc 'python scripts/run_server_stage2_model_job.py --data_root "$DATA_ROOT" --work_root "$WORK_ROOT" --model_name {{}} --skip_existing > "$WORK_ROOT/logs/model_{{}}.out" 2>&1'

# Stage 3: benchmark jobs
printf "%s\n" {benchmark_lines} \
| xargs -I{{}} -P {args.parallel_benchmarks} bash -lc 'set -- {{}}; python scripts/run_server_stage3_benchmark_job.py --data_root "$DATA_ROOT" --work_root "$WORK_ROOT" --model_name "$1" --epoch "$2" --seed "$3" --skip_existing > "$WORK_ROOT/logs/benchmark_${{1}}_e${{2}}_s${{3}}.out" 2>&1'

# Stage 4: final aggregation
python scripts/run_server_stage4_aggregate.py --work_root "$WORK_ROOT"

echo "Progress monitor: python scripts/check_progress.py --work_root $WORK_ROOT --watch"
'''

    out = Path(args.output)
    out.write_text(script, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
