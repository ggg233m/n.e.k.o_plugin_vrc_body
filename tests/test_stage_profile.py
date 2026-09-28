# -*- coding: utf-8 -*-
import unittest

from backend.stage_profile import StageProfiler


class StageProfilerTests(unittest.TestCase):
    def test_records_and_reports_percentiles(self) -> None:
        profiler = StageProfiler(capacity=16)
        for value in (10.0, 20.0, 30.0, 40.0, 300.0):
            profiler.record("stage", value)
        stats = profiler.snapshot()["stage"]
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["last_ms"], 300.0)
        self.assertEqual(stats["median_ms"], 30.0)
        self.assertEqual(stats["max_ms"], 300.0)
        self.assertEqual(stats["total_ms"], 400.0)

    def test_measure_records_even_when_the_block_raises(self) -> None:
        """抛异常也要记：慢和失败常常同时出现，丢样本会让分布偏乐观。"""
        profiler = StageProfiler()
        with self.assertRaises(ValueError):
            with profiler.measure("stage"):
                raise ValueError("boom")
        self.assertEqual(profiler.snapshot()["stage"]["count"], 1)

    def test_capacity_bounds_samples_but_count_stays_total(self) -> None:
        profiler = StageProfiler(capacity=8)
        for index in range(20):
            profiler.record("stage", float(index))
        stats = profiler.snapshot()["stage"]
        self.assertEqual(stats["count"], 20)
        self.assertEqual(stats["max_ms"], 19.0)

    def test_reset_clears_everything(self) -> None:
        profiler = StageProfiler()
        profiler.record("stage", 1.0)
        profiler.reset()
        self.assertEqual(profiler.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
