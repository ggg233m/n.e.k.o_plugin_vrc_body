# -*- coding: utf-8 -*-
"""节拍回归护栏 —— 真跑感知循环，读**真实**的 gap。

为什么必须有这条：2026-10-24 之前 pose_hz=20 的循环实测只有 16.1 Hz（gap p50 62 ms）。
根因是 Windows 上 threading.Event.wait 被量化到 15.625 ms 系统 tick，而 time.monotonic
本身是 GetTickCount64()（读数台阶 16 ms）。修法是 backend/pacing.py 的 Pacer + Gate。

现有测试看不见这个回归：clock 相关测试全都注入假时钟，ThreadsTest 只断言 status 里
键存在。把 Pacer(hz=c.pose_hz) 改成除以 2，一条测试都不会红。这条补的就是这个缺口。
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

import backend.nav_online as nav_online
from tests.test_nav_online import Harness, ground_and_wall


class PosePacingRegressionTest(unittest.TestCase):
    """位姿/双目节拍必须贴近配置值（守护 2026-10-24 之前的 62 ms 回归）。"""

    def _run_for(self, seconds: float) -> dict:
        h = Harness(armed=False, vz=0.0)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)), \
                mock.patch.object(nav_online, "stereo_disparity",
                                  return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(seconds)
                rates = dict(h.nav.status()["rates"])
            finally:
                h.nav.stop()
        return rates

    def test_pose_gap_matches_target(self) -> None:
        rates = self._run_for(2.0)
        gap = rates.get("pose_gap_p50_s")
        self.assertIsNotNone(gap, "pose_gap_p50_s 缺失")
        # 修好前 0.062（62 ms），修好后 0.0500。56 ms 容得下 CI 抖动，
        # 又能把 62 ms 的老毛病挡在门外。
        self.assertLess(gap, 0.056, f"位姿节拍回退：gap p50 {gap}s")

    def test_stereo_gap_matches_target(self) -> None:
        rates = self._run_for(3.0)
        gap = rates.get("stereo_gap_p50_s")
        self.assertIsNotNone(gap, "stereo_gap_p50_s 缺失")
        # 允许 1.15 倍：stereo 落整数个位姿 tick，偶尔在 2 tick 侧。
        self.assertLessEqual(gap, 0.1 * 1.15, f"双目节拍回退：gap p50 {gap}s")

    def test_pacer_telemetry_present(self) -> None:
        """节拍器必须把实测抖动画像外显到 status（声明 vs 实测并排）。"""
        rates = self._run_for(2.0)
        pacer = rates.get("pacer") or {}
        self.assertIn("pose", pacer, "status.rates.pacer.pose 缺失")
        pose = pacer["pose"]
        self.assertIn("late_p99_ms", pose)
        self.assertIn("spin_frac", pose)
        self.assertLess(pose["spin_frac"] or 0.0, 0.10, "自旋占比过高（该是 2-3%）")
