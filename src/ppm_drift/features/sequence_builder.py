from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .encoding import Encoding


def case_to_event_rows(prefix_tbl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {cid: cdf.copy() for cid, cdf in prefix_tbl.groupby("case_id", sort=False)}


def make_prefix_samples_for_case(case_df: pd.DataFrame, enc: Encoding):
    case_df = case_df.sort_values(["prefix_len", "timeStamp"], kind="mergesort").reset_index(drop=True)
    event_idx = np.array([enc.event_to_idx.get(v, 0) for v in case_df["event"].astype(str)], dtype=int)
    veh_idx = np.array([enc.vehicle_to_idx.get(v, 0) for v in case_df["vehicleType"].astype(str)], dtype=int)

    decay = case_df["currentDecayLevel"].astype(float)
    decay = decay.fillna(decay.median() if not decay.dropna().empty else 0.0)

    num = np.vstack([
        case_df["delta_t"].astype(float).values,
        case_df["time_since_start"].astype(float).values,
        decay.values,
        case_df["progress"].astype(float).values,
    ]).T
    num = (num - enc.mean_array()) / enc.std_array()

    e = len(enc.event_to_idx)
    v = len(enc.vehicle_to_idx)
    event_oh = np.zeros((len(case_df), e), dtype=np.float32)
    vehicle_oh = np.zeros((len(case_df), v), dtype=np.float32)
    event_oh[np.arange(len(case_df)), event_idx] = 1.0
    vehicle_oh[np.arange(len(case_df)), veh_idx] = 1.0
    base = np.hstack([num.astype(np.float32), event_oh, vehicle_oh]).astype(np.float32)

    max_len = enc.max_len
    n_prefixes = len(case_df)
    x = np.zeros((n_prefixes, max_len, enc.feature_dim), dtype=np.float32)
    y = case_df["y_true"].astype(float).values.astype(np.float32)
    for k in range(1, n_prefixes + 1):
        seq = base[:k]
        if len(seq) > max_len:
            seq = seq[-max_len:]
        x[k - 1, max_len - len(seq):, :] = seq
    meta = case_df[[
        "exp", "run", "run_id", "case_id", "case_len", "prefix_len", "progress", "timeStamp",
        "event", "vehicleType", "delta_t", "currentDecayLevel", "time_since_start", "y_true",
    ]].copy()
    return x, y, meta


def prefix_generator(train_ids: Iterable, runs: dict, enc: Encoding, load_log_fn, build_prefix_table_fn):
    for rid in train_ids:
        pt = build_prefix_table_fn(load_log_fn(runs[rid]), rid)
        for _, cdf in case_to_event_rows(pt).items():
            x, y, _ = make_prefix_samples_for_case(cdf, enc)
            for i in range(len(y)):
                yield x[i], y[i]
