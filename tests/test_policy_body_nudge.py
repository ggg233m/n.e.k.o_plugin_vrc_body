# -*- coding: utf-8 -*-
"""policy_body.body_feedback：静止卡死检测 + 冷启动引导。

两版教训：只看 OSC 静默不够。VRChat 对静止角色每帧回传 velocity=0，
available=True、值=0，引导永远进不去。正确判据是"回传约 0 且航点还远"。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

from policy_body import body_feedback, NUDGE_SPEED  # noqa: E402


class BodyFeedbackTests(unittest.TestCase):
    def test_moving_feedback_wins(self) -> None:
        self.assertEqual(body_feedback((0.3, 0.9), (0.3, 0.8)), (0.3, 0.9))

    def test_stall_with_far_waypoint_gets_nudge(self) -> None:
        vx, vz = body_feedback((0.0, 0.0), (0.3, 0.8))
        dist = (0.3 ** 2 + 0.8 ** 2) ** 0.5
        self.assertAlmostEqual(vz, NUDGE_SPEED * 0.8 / dist, places=3)
        self.assertAlmostEqual(vx, NUDGE_SPEED * 0.3 / dist, places=3)

    def test_stall_with_no_waypoint_stays_zero(self) -> None:
        self.assertEqual(body_feedback((0.0, 0.0), None), (0.0, 0.0))

    def test_stall_with_near_waypoint_stays_zero(self) -> None:
        self.assertEqual(body_feedback((0.0, 0.0), (0.05, 0.05)), (0.0, 0.0))

    def test_silent_with_waypoint_gets_nudge(self) -> None:
        vx, vz = body_feedback(None, (0.3, 0.8))
        dist = (0.3 ** 2 + 0.8 ** 2) ** 0.5
        self.assertAlmostEqual(vz, NUDGE_SPEED * 0.8 / dist, places=3)

    def test_disabled_returns_raw(self) -> None:
        self.assertEqual(body_feedback((0.0, 0.0), (0.3, 0.8), nudged=False), (0.0, 0.0))
        self.assertEqual(body_feedback(None, (0.3, 0.8), nudged=False), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()