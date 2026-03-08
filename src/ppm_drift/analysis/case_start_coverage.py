from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import START_EVENT_NAME, RunId, label_group
from ..data.loading import load_log
from ..data.preprocess import clean_and_sort_log


def compute_case_start_coverage(runs: dict[RunId, Path], source_test_runs: set[int], start_event_name: str = START_EVENT_NAME) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_rows = []
    for rid, path in runs.items():
        df = clean_and_sort_log(load_log(path))
        first = df.groupby("productIDStr", as_index=False).first()[["productIDStr", "event"]]
        first["run_id"] = rid.key
        first["exp"] = rid.exp
        first["run"] = rid.run
        first["group"] = label_group(rid.exp, rid.run, source_test_runs)
        first_rows.append(first)
    first_rows = pd.concat(first_rows, ignore_index=True)
    per_run = first_rows.groupby(["run_id", "exp", "run", "group"], as_index=False).agg(
        n_cases=("productIDStr", "nunique"),
        n_start=("event", lambda s: int((s == start_event_name).sum())),
    )
    per_run["start_event_rate"] = per_run["n_start"] / per_run["n_cases"].clip(lower=1)
    first_event_dist = first_rows.groupby(["group", "event"], as_index=False).agg(n=("productIDStr", "size"))
    first_event_dist["share"] = first_event_dist.groupby("group")["n"].transform(lambda x: x / x.sum())
    return per_run, first_event_dist
