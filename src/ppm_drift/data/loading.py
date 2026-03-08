from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="	", encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df
