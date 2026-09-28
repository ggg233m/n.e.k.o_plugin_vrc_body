# -*- coding: utf-8 -*-
"""在线链路的阶段耗时统计。

只做观测，不改变任何行为。存在的理由很具体：在线建图期实测帧间隔常超过
0.25 秒、有时超过 0.8 秒，而单帧 SLAM 只要约 28 毫秒 —— 差额必须先看到
耗时分布才能定优化点，凭直觉调参数只会引入新问题。
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import statistics
import threading
import time
from typing import Any, Iterator


class StageProfiler:
    """按阶段记录最近若干次耗时，并可输出中位/p90/峰值。"""

    def __init__(self, capacity: int = 120) -> None:
        self._capacity = max(8, int(capacity))
        self._samples: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}
        self._totals: dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            # 抛异常也要记录：慢和失败往往同时出现，丢掉样本会让分布偏乐观。
            self.record(name, (time.perf_counter() - started) * 1000.0)

    def record(self, name: str, elapsed_ms: float) -> None:
        if not name:
            return
        try:
            value = float(elapsed_ms)
        except (TypeError, ValueError):
            return
        with self._lock:
            bucket = self._samples.get(name)
            if bucket is None:
                bucket = deque(maxlen=self._capacity)
                self._samples[name] = bucket
            bucket.append(value)
            self._counts[name] = self._counts.get(name, 0) + 1
            self._totals[name] = self._totals.get(name, 0.0) + value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            for name, bucket in self._samples.items():
                values = sorted(bucket)
                if not values:
                    continue
                index = min(len(values) - 1, int(len(values) * 0.9))
                result[name] = {
                    "count": int(self._counts.get(name, 0)),
                    "last_ms": round(float(bucket[-1]), 3),
                    "median_ms": round(float(statistics.median(values)), 3),
                    "p90_ms": round(float(values[index]), 3),
                    "max_ms": round(float(values[-1]), 3),
                    "total_ms": round(float(self._totals.get(name, 0.0)), 3),
                }
            return result

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._counts.clear()
            self._totals.clear()


__all__ = ["StageProfiler"]
