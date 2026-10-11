from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

_timings: dict[str, list[float]] = defaultdict(list)


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        _timings[name].append(time.perf_counter() - start)


def record(name: str, seconds: float) -> None:
    _timings[name].append(seconds)


def summary() -> dict[str, dict[str, float]]:
    return {
        name: {"count": len(v), "total_s": sum(v), "mean_s": sum(v) / len(v)}
        for name, v in _timings.items()
    }


def reset() -> None:
    _timings.clear()
