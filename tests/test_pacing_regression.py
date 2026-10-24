# -*- coding: utf-8 -*-
"""节拍回归护栏 —— 真跑感知循环，读**真实**的 gap。

为什么必须有这条：2026-10-24 之前 pose_hz=20 的循环实测只有 16.1 Hz（gap p50 62 ms）。
根因是 Windows 上 threading.Event.wait 被量化到 15.625 ms 系统 tick，而 time.monotonic
本身是 GetTickCount64()（读数台阶 16 ms）。修法是 backend/pacing.py 的 Pacer + Gate。

⚠️ 必须从 **tests.test_nav_online** 取 nav_online，不能直接 import backend.nav_online：
conftest 把 backend 挂到了 stub 包 neko_anyadance_body 下，同一次运行里存在**两个**
nav_online 模块对象。直接 import 的那个 patch 不上 Harness 真正跑的那份，
mock 会静默失效、真实 stereo_disparity 在 4x4 零数组上抛异常被吞掉，
测试照样绿 —— 这正是本文件第一版踩过的坑（2026-10-24 审查发现）。
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

# 同一模块对象：Harness 用的就是它。
from tests.test_nav_online import Harness, ground_and_wall, nav_online


class PosePacingRegressionTest(unittest.TestCase):
    """位姿/双目节拍必须贴近配置值（守护 2026-10-24 之前的 62 ms 回归）。"""

    def _run_for(self, seconds: float, stereo_period_s: float | None = None,
                 decoupled: bool | None = None) -> dict:
        h = Harness(armed=False, vz=0.3)
        if stereo_period_s is not None:
            h.nav.cfg.stereo_period_s = stereo_period_s
        if decoupled is not None:
            h.nav.cfg.decoupled = decoupled
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)), \
                mock.patch.object(nav_online, "stereo_disparity",
                                  return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(seconds)
                st = h.nav.status()
                rates = dict(st["rates"])
                rates["_errors"] = list(st.get("errors") or [])
                # mapper 是 trail 的**真值**：门触发 ≠ 轨迹真的挂上了。
                _tr = getattr(h.nav.mapper, "_trail", {})
                rates["_trail_keys"] = len(_tr)
                rates["_trail_pts"] = sum(len(v) for v in _tr.values())
            finally:
                h.nav.stop()
        return rates

    def test_mocks_are_effective(self) -> None:
        """先证伪：如果 patch 打偏，真实 SGBM 会在 4x4 零数组上崩，errors 非空。

        这条是 D3 的护栏：它保证上面几条测的是被 mock 过的循环，而不是一个
        每帧都抛异常、靠吞异常“看起来正常”的循环。
        """
        rates = self._run_for(1.0)
        self.assertEqual(rates["_errors"], [], f"循环有异常（说明 mock 没生效）: {rates['_errors'][:2]}")

    def test_pose_gap_matches_target(self) -> None:
        rates = self._run_for(2.0)
        gap = rates.get("pose_gap_p50_s")
        target = 1.0 / float(rates["pose_hz_target"])
        self.assertIsNotNone(gap, "pose_gap_p50_s 缺失")
        # 相对界而不是绝对界：改 OnlineNavConfig.pose_hz 时阈值跟着走。
        # 修好前是 62 ms（老实现 = 1.24×target），修好后 0.0500（1.00×）。
        self.assertLess(gap, 1.12 * target, f"位姿节拍回退：gap p50 {gap}s（target {target}s）")

    def test_stereo_gap_matches_target(self) -> None:
        # 显式给 0.1：Harness 默认是 0.05，用默认值会让下面那条上界变得毫无意义。
        rates = self._run_for(3.0, stereo_period_s=0.1)
        gap = rates.get("stereo_gap_p50_s")
        target = float(rates["stereo_hz_target"]) and 1.0 / float(rates["stereo_hz_target"])
        self.assertIsNotNone(gap, "stereo_gap_p50_s 缺失")
        # 双目落整数个位姿 tick，允许 1.15 倍。
        self.assertLessEqual(gap, 1.15 * target, f"双目节拍回退：gap p50 {gap}s（target {target}s）")
        # 而且必须真的在发：样本数不能明显少于位姿样本×比值
        samples = rates.get("samples") or {}
        self.assertGreater(samples.get("stereo", 0), 20, "双目样本太少")

    def test_pacer_telemetry_present(self) -> None:
        """节拍器必须把实测抖动画像外显到 status（声明 vs 实测并排）。"""
        rates = self._run_for(2.0)
        pacer = rates.get("pacer") or {}
        self.assertIn("pose", pacer, "status.rates.pacer.pose 缺失")
        pose = pacer["pose"]
        self.assertIn("late_p99_ms", pose)
        self.assertIn("spin_frac", pose)
        self.assertLess(pose["spin_frac"] or 0.0, 0.10, "自旋占比过高（该是 2-3%）")

    def _trail_stats(self, rates) -> tuple[int, int, int]:
        return (rates["_trail_keys"], rates["_trail_pts"],
                ((rates.get("pacer") or {}).get("trail_gate") or {}).get("fires", 0))

    def test_trail_actually_lands_in_mapper(self) -> None:
        """trail 必须**真的挂进 mapper**，而不只是门被求值。

        2026-10-24 两轮审查踩坑记录（这条测试就是为它们写的）：
          ① 门被 `kf_id >= 0` 守着，而 kf_id 在解耦分支永不递增 ⇒ 一条不写；
          ② 改成 `_kf_next_id - 1` ⇒ 挂的是**正在提交**那帧，mapper 里还没有，
             `add_trail` 的 `if k not in self._pose: return` 静默丢弃；
             且 inline 路径不递增 _kf_next_id ⇒ inline 下一条不写。
        只断言 `trail_gate.fires > 0` 抓不到以上任何一种 —— 它只证明门被求值过。
        """
        for decoupled in (True, False):
            with self.subTest(decoupled=decoupled):
                keys, pts, fires = self._trail_stats(self._run_for(6.0, decoupled=decoupled))
                self.assertGreater(fires, 0, f"decoupled={decoupled} 门没触发")
                self.assertGreater(keys, 0, f"decoupled={decoupled} trail 没挂进任何关键帧")
                self.assertGreater(pts, 0, f"decoupled={decoupled} trail 点数为 0")
