from __future__ import annotations

from dataclasses import dataclass

SOURCE_EXPS = set(range(10, 19))
TARGET_EXPS = set(range(1, 10))
ALT_EXPS = set(range(19, 28))
DEFAULT_SOURCE_TEST_RUNS = {19, 20}
START_EVENT_NAME = "arrivalAtSource"


@dataclass(frozen=True)
class RunId:
    exp: int
    run: int

    @property
    def key(self) -> str:
        return f"Exp{self.exp}Run{self.run}"


def label_group(exp: int, run: int, source_test_runs: set[int] | None = None) -> str:
    source_test_runs = source_test_runs or DEFAULT_SOURCE_TEST_RUNS
    if exp in TARGET_EXPS:
        return "target"
    if exp in SOURCE_EXPS:
        return "source-test" if run in source_test_runs else "source-train"
    if exp in ALT_EXPS:
        return "alt"
    return "other"
