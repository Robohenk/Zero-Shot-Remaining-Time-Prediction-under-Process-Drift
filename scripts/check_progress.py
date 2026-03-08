#!/usr/bin/env python3
"""Show a live status summary for the end-to-end experiment pipeline."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd


def count_files(p: Path) -> int:
    return sum(1 for x in p.rglob('*') if x.is_file()) if p.exists() else 0


def tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def load_timings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return pd.DataFrame(rows)


def fmt_seconds(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 'n/a'
    x = int(round(float(x)))
    h, rem = divmod(x, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def snapshot(work_root: Path) -> str:
    status_path = work_root / 'pipeline_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    timings = load_timings(work_root / 'logs' / 'timings.jsonl')
    lines = [f"work_root: {work_root}"]
    if status:
        lines.append(f"stage: {status.get('stage')} | status: {status.get('status')} | last_elapsed_seconds: {status.get('last_elapsed_seconds')}")
        if 'current_model' in status:
            lines.append(f"current_model: {status.get('current_model')}")
        started = status.get('current_stage_started_ts')
        if started:
            lines.append(f"current_stage_runtime: {fmt_seconds(time.time() - float(started))}")
    for sub in ['dataset_checks', 'logs', 'predictions', 'artifacts', 'figures', 'benchmarks']:
        lines.append(f"{sub}: {count_files(work_root / sub)} files")
    if not timings.empty:
        recent = timings.tail(8).copy()
        lines.append('--- recent completed timings ---')
        for _, r in recent.iterrows():
            model = f" | model={r['model']}" if 'model' in r and pd.notna(r['model']) else ''
            lines.append(f"{r['stage']}: {fmt_seconds(r['elapsed_seconds'])}{model}")
        if 'model' in timings.columns:
            agg = timings.dropna(subset=['model']).groupby('model', as_index=False)['elapsed_seconds'].sum()
            if not agg.empty:
                lines.append('--- cumulative timed compute by model ---')
                for _, r in agg.iterrows():
                    lines.append(f"{r['model']}: {fmt_seconds(r['elapsed_seconds'])}")
    log_file = work_root / 'logs' / 'pipeline.log'
    if log_file.exists():
        lines.append('--- recent log lines ---')
        lines.append(tail(log_file, n=12))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description='Show live pipeline progress summary.')
    ap.add_argument('--work_root', required=True)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=float, default=5.0)
    args = ap.parse_args()
    work_root = Path(args.work_root)
    if not args.watch:
        print(snapshot(work_root))
        return
    while True:
        print('\033c', end='')
        print(snapshot(work_root))
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
