from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.frame_budget import FrameBudget


class FrameBudgetTests(unittest.TestCase):
    def test_calls_under_the_limit_are_allowed(self) -> None:
        budget = FrameBudget(3)
        for index in range(3):
            verdict = budget.consume(1000.0 + index)
            self.assertTrue(verdict["allowed"], index)
            self.assertEqual(verdict["used"], index + 1)

    def test_the_call_past_the_limit_is_refused_with_a_wait(self) -> None:
        budget = FrameBudget(2)
        budget.consume(1000.0)
        budget.consume(1010.0)
        verdict = budget.consume(1020.0)
        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "frame_rate_limited")
        # 最早那次在 1000.0，窗口 60 s，所以 1060.0 才轮得到 → 还要 40 s。
        self.assertEqual(verdict["retry_after_ms"], 40000)

    def test_the_window_slides_instead_of_resetting(self) -> None:
        """固定窗口会在整点放行一整批；滑动窗口任何一分钟内都只放行 limit 次。"""
        budget = FrameBudget(2)
        budget.consume(1000.0)
        budget.consume(1030.0)
        self.assertFalse(budget.consume(1050.0)["allowed"])
        # 1000.0 那次在 1060.0 滑出，于是恰好空出一格——不是两格。
        self.assertTrue(budget.consume(1061.0)["allowed"])
        self.assertFalse(budget.consume(1062.0)["allowed"])

    def test_boundary_call_at_exactly_one_window_is_allowed(self) -> None:
        budget = FrameBudget(1)
        budget.consume(1000.0)
        self.assertFalse(budget.consume(1059.999)["allowed"])
        self.assertTrue(budget.consume(1060.0)["allowed"])

    def test_refused_calls_are_not_counted(self) -> None:
        """否则不断重试会把窗口无限续期，限流变成永久锁死。"""
        budget = FrameBudget(1)
        budget.consume(1000.0)
        for offset in range(1, 50):
            budget.consume(1000.0 + offset)
        # 记账里始终只有最初那一次，所以 1060.0 就能恢复。
        self.assertTrue(budget.consume(1060.0)["allowed"])

    def test_zero_limit_disables_pulling_rather_than_unlocking_it(self) -> None:
        budget = FrameBudget(0)
        verdict = budget.consume(1000.0)
        self.assertFalse(verdict["allowed"])
        self.assertEqual(verdict["reason"], "frame_pull_disabled")
        self.assertIsNone(verdict["retry_after_ms"])

    def test_check_does_not_consume(self) -> None:
        budget = FrameBudget(1)
        self.assertTrue(budget.check(1000.0)["allowed"])
        self.assertTrue(budget.check(1000.0)["allowed"])
        self.assertTrue(budget.consume(1000.0)["allowed"])
        self.assertFalse(budget.check(1000.0)["allowed"])

    def test_malformed_limits_fall_back_instead_of_raising(self) -> None:
        for bogus in (None, "ten", float("nan"), object()):
            self.assertEqual(FrameBudget(bogus).limit, 10, bogus)
        self.assertEqual(FrameBudget(-5).limit, 0)

    def test_malformed_window_falls_back_to_one_minute(self) -> None:
        self.assertEqual(FrameBudget(2, window_s="soon").window_s, 60.0)
        # 窗口下限 1 s：0 或负数会让每次调用都立刻滑出，等于没有限流。
        self.assertEqual(FrameBudget(2, window_s=0).window_s, 1.0)

    def test_status_reports_usage_without_consuming(self) -> None:
        budget = FrameBudget(4)
        budget.consume(1000.0)
        budget.consume(1005.0)
        status = budget.status(1010.0)
        self.assertEqual(status["limit"], 4)
        self.assertEqual(status["used_last_window"], 2)
        self.assertEqual(status["window_s"], 60.0)
        # 过了窗口就该归零。
        self.assertEqual(budget.status(1100.0)["used_last_window"], 0)


if __name__ == "__main__":
    unittest.main()
