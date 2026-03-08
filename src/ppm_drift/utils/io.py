from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..constants import RunId

FILENAME_RE = re.compile(r"Exp(?P<exp>\d+)Run(?P<run>\d+)\.txt$", re.IGNORECASE)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def discover_runs(data_root: str | Path) -> dict[RunId, Path]:
    data_root = Path(data_root)
    found: dict[RunId, Path] = {}
    for path in data_root.rglob("Exp*Run*.txt"):
        match = FILENAME_RE.search(path.name)
        if match:
            rid = RunId(exp=int(match.group("exp")), run=int(match.group("run")))
            found[rid] = path
    return dict(sorted(found.items(), key=lambda kv: (kv[0].exp, kv[0].run)))


def read_table_flex(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **kwargs)
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, **kwargs, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def dump_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parquet_files(path: str | Path) -> list[Path]:
    return sorted(Path(path).glob("*.parquet"))


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
