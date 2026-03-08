from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import RunId
from ..data.loading import load_log
from ..data.preprocess import build_prefix_table
from ..utils.io import load_json, dump_json


@dataclass
class Encoding:
    event_to_idx: dict[str, int]
    vehicle_to_idx: dict[str, int]
    max_len: int
    num_mean: list[float]
    num_std: list[float]
    n_train_prefix_rows: int = 0

    @property
    def numeric_dim(self) -> int:
        return 4

    @property
    def feature_dim(self) -> int:
        return self.numeric_dim + len(self.event_to_idx) + len(self.vehicle_to_idx)

    def mean_array(self) -> np.ndarray:
        return np.asarray(self.num_mean, dtype=float)

    def std_array(self) -> np.ndarray:
        return np.asarray(self.num_std, dtype=float)

    def to_json(self, path: str | Path) -> None:
        dump_json(path, asdict(self))

    @staticmethod
    def from_json(path: str | Path) -> "Encoding":
        return Encoding(**load_json(path))


def _combine_stats(n1: int, mean1: np.ndarray, m2_1: np.ndarray, n2: int, mean2: np.ndarray, m2_2: np.ndarray):
    if n1 == 0:
        return n2, mean2, m2_2
    if n2 == 0:
        return n1, mean1, m2_1
    delta = mean2 - mean1
    n = n1 + n2
    mean = mean1 + delta * (n2 / n)
    m2 = m2_1 + m2_2 + (delta ** 2) * (n1 * n2 / n)
    return n, mean, m2


def fit_encoding_stream(train_ids: list[RunId], runs: dict[RunId, Path], max_len_cap: int = 200) -> Encoding:
    events: set[str] = set()
    vehicles: set[str] = set()
    max_len = 0
    n_rows = 0
    n = 0
    mean = np.zeros(4, dtype=float)
    m2 = np.zeros(4, dtype=float)

    for rid in train_ids:
        pt = build_prefix_table(load_log(runs[rid]), rid)
        n_rows += len(pt)
        events.update(pt["event"].astype(str).unique())
        vehicles.update(pt["vehicleType"].astype(str).unique())
        max_len = max(max_len, int(pt["case_len"].max()))

        decay = pt["currentDecayLevel"].astype(float)
        decay = decay.fillna(decay.median() if not decay.dropna().empty else 0.0)
        num = np.vstack([
            pt["delta_t"].astype(float).values,
            pt["time_since_start"].astype(float).values,
            decay.values,
            pt["progress"].astype(float).values,
        ]).T
        num = num[~np.isnan(num).any(axis=1)]
        if len(num) == 0:
            continue
        n2 = len(num)
        mean2 = num.mean(axis=0)
        m2_2 = num.var(axis=0, ddof=0) * n2
        n, mean, m2 = _combine_stats(n, mean, m2, n2, mean2, m2_2)

    std = np.sqrt(np.maximum(m2 / max(n, 1), 1e-12))
    std[std == 0] = 1.0
    event_to_idx = {v: i for i, v in enumerate(sorted(events or {"NA"}))}
    vehicle_to_idx = {v: i for i, v in enumerate(sorted(vehicles or {"NA"}))}
    return Encoding(
        event_to_idx=event_to_idx,
        vehicle_to_idx=vehicle_to_idx,
        max_len=min(max_len_cap, max_len if max_len > 0 else max_len_cap),
        num_mean=mean.tolist(),
        num_std=std.tolist(),
        n_train_prefix_rows=n_rows,
    )
