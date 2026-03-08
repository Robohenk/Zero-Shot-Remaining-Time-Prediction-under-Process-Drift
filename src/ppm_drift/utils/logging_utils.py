"""Logging, timing, and live-status helpers for long experiment pipelines."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .io import ensure_dir


def setup_logger(log_dir: str | Path, name: str = "ppm_drift", filename: str = "pipeline.log") -> logging.Logger:
    """Create a file+console logger for one pipeline stage or script."""
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger(f"{name}:{Path(filename).stem}:{log_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(Path(log_dir) / filename, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def append_timing(log_dir: str | Path, stage: str, elapsed_seconds: float, extra: dict | None = None) -> None:
    """Append one completed-stage timing row to a JSONL log."""
    log_dir = ensure_dir(log_dir)
    row = {"stage": stage, "elapsed_seconds": round(float(elapsed_seconds), 6), "ts": time.time()}
    if extra:
        row.update(extra)
    with (Path(log_dir) / "timings.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def update_status(work_root: str | Path, stage: str, status: str, extra: dict | None = None) -> None:
    """Write a lightweight JSON status file consumed by the live progress monitor."""
    path = ensure_dir(work_root) / "pipeline_status.json"
    payload = {"stage": stage, "status": status, "updated_ts": time.time()}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) | payload
        except Exception:
            pass
    if status == "running":
        payload["current_stage_started_ts"] = time.time()
        payload["current_stage"] = stage
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@contextmanager
def timed_stage(logger: logging.Logger, log_dir: str | Path, work_root: str | Path, stage: str, extra: dict | None = None) -> Iterator[None]:
    """Context manager that logs start/end timestamps and updates pipeline status."""
    logger.info("START %s", stage)
    update_status(work_root, stage, "running", extra)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("END %s | elapsed_seconds=%.3f", stage, elapsed)
        append_timing(log_dir, stage, elapsed, extra)
        payload = {"last_elapsed_seconds": round(elapsed, 3), **(extra or {})}
        payload["current_stage_started_ts"] = None
        update_status(work_root, stage, "completed", payload)
