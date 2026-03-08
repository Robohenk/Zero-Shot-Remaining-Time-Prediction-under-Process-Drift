from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..features.encoding import Encoding


class BaseRemainingTimeModel(ABC):
    def __init__(self, encoding: Encoding, **kwargs: Any) -> None:
        self.encoding = encoding
        self.kwargs = kwargs
        self.model = None

    @abstractmethod
    def build(self):
        raise NotImplementedError

    def fit(self, train_ds, val_ds=None, **fit_kwargs):
        if self.model is None:
            self.model = self.build()
        return self.model.fit(train_ds, validation_data=val_ds, **fit_kwargs)

    def predict(self, x, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not built")
        return self.model.predict(x, **kwargs)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not built")
        self.model.save(path)
