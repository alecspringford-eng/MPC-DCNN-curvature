from __future__ import annotations

import time

from dcnn_curvature.profiling import record, reset, summary, timer


def test_timer_and_record_accumulate_into_summary():
    reset()
    with timer("a"):
        time.sleep(0.001)
    record("a", 0.5)
    record("b", 1.0)

    result = summary()

    assert result["a"]["count"] == 2
    assert result["a"]["total_s"] >= 0.5
    assert result["b"] == {"count": 1, "total_s": 1.0, "mean_s": 1.0}


def test_reset_clears_all_timings():
    record("x", 1.0)
    reset()

    assert summary() == {}
