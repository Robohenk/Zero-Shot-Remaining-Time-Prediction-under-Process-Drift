from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .constants import DEFAULT_SOURCE_TEST_RUNS


@dataclass
class SplitConfig:
    source_test_runs: set[int] = field(default_factory=lambda: set(DEFAULT_SOURCE_TEST_RUNS))


@dataclass
class DataConfig:
    data_root: str = "data"
    output_root: str = "outputs"
    max_len_cap: int = 200
    time_col: str = "timeStamp"
    case_col: str = "productIDStr"
    event_col: str = "event"
    vehicle_col: str = "vehicleType"
    unique_id_col: str = "uniqueID"
    decay_col: str = "currentDecayLevel"


@dataclass
class TrainConfig:
    model_name: str = "lstm"
    seed: int = 7
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    transformer_heads: int = 4
    transformer_ff_dim: int = 128
    transformer_layers: int = 2
    dropout: float = 0.1
    validation_fraction: float = 0.10
    use_early_stopping: bool = True
    patience: int = 3
    cpu_workers: int = 4


@dataclass
class BenchmarkConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "BenchmarkConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = BenchmarkConfig()
        for section in ("data", "split", "train"):
            values = payload.get(section, {})
            target = getattr(cfg, section)
            for k, v in values.items():
                setattr(target, k, v)
        cfg.split.source_test_runs = set(cfg.split.source_test_runs)
        return cfg
