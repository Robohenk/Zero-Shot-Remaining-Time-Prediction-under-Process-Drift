from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..constants import RunId


REQUIRED_COLUMNS = {"productIDStr", "timeStamp"}


def parse_timestamp_series(ts: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(ts, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(ts, errors="coerce"))
    return parsed


def clean_and_sort_log(df: pd.DataFrame, cfg: DataConfig | None = None) -> pd.DataFrame:
    cfg = cfg or DataConfig()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work[cfg.case_col] = work[cfg.case_col].astype(str)
    work = work[work[cfg.case_col].notna()]
    work = work[work[cfg.case_col].str.strip().ne("")]
    work = work[work[cfg.case_col].str.lower().ne("na")]

    work[cfg.time_col] = parse_timestamp_series(work[cfg.time_col])
    work = work[work[cfg.time_col].notna()].copy()

    if cfg.unique_id_col in work.columns:
        work[cfg.unique_id_col] = pd.to_numeric(work[cfg.unique_id_col], errors="coerce")
    else:
        work[cfg.unique_id_col] = np.arange(len(work), dtype=float)

    for c in (cfg.event_col, cfg.vehicle_col, "vehicle", "processingStation"):
        if c not in work.columns:
            work[c] = "NA"
        work[c] = work[c].fillna("NA").astype(str)

    if cfg.decay_col not in work.columns:
        work[cfg.decay_col] = np.nan
    work[cfg.decay_col] = pd.to_numeric(work[cfg.decay_col], errors="coerce")

    work = work.sort_values([cfg.case_col, cfg.time_col, cfg.unique_id_col], kind="mergesort")
    return work


def build_prefix_table(df: pd.DataFrame, rid: RunId, cfg: DataConfig | None = None) -> pd.DataFrame:
    cfg = cfg or DataConfig()
    work = clean_and_sort_log(df, cfg)

    start_times = work.groupby(cfg.case_col)[cfg.time_col].min().rename("case_start_time")
    end_times = work.groupby(cfg.case_col)[cfg.time_col].max().rename("case_end_time")
    case_lens = work.groupby(cfg.case_col).size().rename("case_len")

    work = work.join(start_times, on=cfg.case_col)
    work = work.join(end_times, on=cfg.case_col)
    work = work.join(case_lens, on=cfg.case_col)

    work["prev_time"] = work.groupby(cfg.case_col)[cfg.time_col].shift(1)
    work["delta_t"] = (work[cfg.time_col] - work["prev_time"]).dt.total_seconds().fillna(0.0)
    work["prefix_len"] = work.groupby(cfg.case_col).cumcount() + 1
    work["y_true"] = (work["case_end_time"] - work[cfg.time_col]).dt.total_seconds().clip(lower=0)
    work["progress"] = work["prefix_len"] / work["case_len"].clip(lower=1)
    work["time_since_start"] = (work[cfg.time_col] - work["case_start_time"]).dt.total_seconds().fillna(0.0)

    prefix = work[[
        cfg.case_col,
        cfg.event_col,
        cfg.vehicle_col,
        cfg.decay_col,
        cfg.time_col,
        "case_start_time",
        "case_end_time",
        "case_len",
        "prefix_len",
        "progress",
        "delta_t",
        "time_since_start",
        "y_true",
    ]].copy()
    prefix.rename(columns={cfg.case_col: "case_id", cfg.event_col: "event", cfg.vehicle_col: "vehicleType", cfg.decay_col: "currentDecayLevel", cfg.time_col: "timeStamp"}, inplace=True)
    prefix.insert(0, "run", rid.run)
    prefix.insert(0, "exp", rid.exp)
    prefix.insert(0, "run_id", rid.key)
    return prefix
