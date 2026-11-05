# -*- coding: utf-8 -*-
"""30 Hz ω 标签源的回归测试（2026-11-05）。

锁死四条：
  1. **因果性**：扰动 t1 之后的 AngularY，窗口 [t0,t1) 的 ω 必须逐位不变；
  2. **30 Hz vs 单点**：等间隔合成信号上两者的差异符合预期（区间法保留短脉冲）；
  3. **转向事件召回**：30 Hz 子网格比单点 ZOH 保留更多短转向事件；
  4. **缺字段回退**：没有 AngularY 时返回 None，调用方按契约写 0.0。

不依赖真实录制 / GPU / VRChat。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

import build_policy_dataset as B  # noqa: E402


def make_streams(t, v):
    """只填 ω 相关字段；其余字段本测试不用。"""
    return dict(ay_t=np.asarray(t, dtype=np.float64),
                ay_v=np.asarray(v, dtype=np.float64))


class CausalOmegaIntervalTests(unittest.TestCase):
    """因果性：窗口之后的数据不能影响窗口内的 ω。"""

    def test_future_perturbation_does_not_change(self):
        # 0..1s 每 10ms 一个采样，前半静止，后半匀速转
        t = np.arange(0.0, 1.0, 0.01)
        v = np.where(t < 0.5, 0.0, 100.0)
        s0 = make_streams(t, v)
        base = B.omega_rad_s_interval(s0, 0.2, 0.3, src_hz=30)
        self.assertIsNotNone(base)

        # 把 t>=0.3 的采样全部改掉（窗口 [0.2,0.3) 之后）
        v2 = v.copy()
        mask = t >= 0.3
        v2[mask] = -999.0
        s1 = make_streams(t, v2)
        got = B.omega_rad_s_interval(s1, 0.2, 0.3, src_hz=30)
        self.assertAlmostEqual(base, got, places=12,
                               msg="窗口之后的数据影响了窗口内 ω，违反因果性")

    def test_pre_window_data_is_allowed(self):
        # 窗口用的 ZOH 会回溯到窗口起点之前（这是合法的「过去」）
        t = np.array([0.0, 0.15, 0.35])
        v = np.array([0.0, 500.0, 0.0])
        s = make_streams(t, v)
        got = B.omega_rad_s_interval(s, 0.2, 0.3, src_hz=30)
        # 窗口内全是 0.15 时刻那条 500 -> 应为非零
        self.assertIsNotNone(got)
        self.assertGreater(abs(got), 0.0)


class AggregationBehaviorTests(unittest.TestCase):
    """30 Hz 区间聚合 vs 单点 ZOH 的差异方向。"""

    def _pulse(self, lo, hi):
        """在 [lo,hi) 内注入一段强转向脉冲，其余为 0。"""
        t = np.arange(0.0, 1.0, 0.002)
        v = np.where((t >= lo) & (t < hi), 1000.0, 0.0)
        return make_streams(t, v)

    def test_short_pulse_kept_by_interval_not_by_point(self):
        # 脉冲落在 [0.25,0.28)，而 0.1s 步的起点 q=0.2 处恰好为 0
        s = self._pulse(0.25, 0.28)
        point = B.omega_rad_s_at(s, 0.2)
        interval = B.omega_rad_s_interval(s, 0.2, 0.3, src_hz=30)
        self.assertAlmostEqual(point, 0.0, places=9,
                               msg="单点 ZOH 在 q=0.2 应读不到 0.25 的脉冲")
        self.assertIsNotNone(interval)
        self.assertGreater(abs(interval), 0.0,
                           msg="区间聚合应捕捉到 0.25 的短脉冲")

    def test_direction_matches_point_zoh(self):
        """方向必须与单点 ZOH 口径**同号**。

        这是本轮抓到的真 bug 的回归锁（2026-11-05）：
        第一版实现写 `sign(sacc) * mag`，漏乘 ANGULARY_RAD_PER_UNIT 自带的负号，
        导致区间法方向整体反转，与单点口径相关系数实测 -0.67。
        改为 `sgn * unit_sign * mag` 后恢复 +0.67。
        """
        t = np.arange(0.0, 1.0, 0.002)
        for raw, label in ((1000.0, "AngularY 正"), (-1000.0, "AngularY 负")):
            v = np.where((t >= 0.2) & (t < 0.3), raw, 0.0)
            s = make_streams(t, v)
            point = B.omega_rad_s_at(s, 0.25)      # 窗口正中的单点 ZOH
            interval = B.omega_rad_s_interval(s, 0.2, 0.3, src_hz=30)
            self.assertIsNotNone(point)
            self.assertIsNotNone(interval)
            self.assertGreater(abs(point), 0.0, "%s: 单点应为非零" % label)
            self.assertGreater(abs(interval), 0.0, "%s: 区间应为非零" % label)
            self.assertEqual(np.sign(point), np.sign(interval),
                             "%s: 区间法与单点法符号不一致（点=%.4f 区间=%.4f）"
                             % (label, point, interval))

    def test_sign_matches_point_on_negative_raw(self):
        """专项：AngularY 为负时，区间法结果也必须与单点同号（防双重取负）。"""
        t = np.arange(0.0, 1.0, 0.002)
        v = np.where((t >= 0.2) & (t < 0.3), -256.0, 0.0)
        s = make_streams(t, v)
        point = B.omega_rad_s_at(s, 0.25)
        interval = B.omega_rad_s_interval(s, 0.2, 0.3, src_hz=30)
        self.assertGreater(point, 0.0,
                           "ANGULARY_RAD_PER_UNIT 为负，raw 为负时应得正的 rad/s")
        self.assertGreater(interval, 0.0,
                           "区间法在这里变负 = 符号被取了两次")

    def test_symmetric_opposite_turns_keep_magnitude(self):
        """窗口内正反等量时，**幅值应当保留**（那 0.1 s 确实在转），不是抵消成 0。

        设计说明：实现取「绝对值均值 × 带符号均值的符号」。
        正反等量时带符号均值≈0，方向不可定，但幅值必须非零——
        若把幅值也抵消成 0，就等于告诉模型「这段没转」，
        而实际转了 180 度再转回来。
        （第一版测试误写成「应抵消成 0」，是断言错不是代码错。）
        """
        t = np.arange(0.0, 1.0, 0.002)
        v = np.where((t >= 0.2) & (t < 0.25), 1000.0,
                     np.where((t >= 0.25) & (t < 0.3), -1000.0, 0.0))
        s = make_streams(t, v)
        got = B.omega_rad_s_interval(s, 0.2, 0.3, src_hz=30)
        self.assertIsNotNone(got)
        self.assertGreater(abs(got), 1.0,
                           "正反转窗口的幅值必须保留，不能被抵消成 0")
        one_side = 1000.0 * abs(B.ANGULARY_RAD_PER_UNIT)
        self.assertAlmostEqual(abs(got), one_side, delta=one_side * 0.3,
                               msg="幅值应接近单侧满幅，实测 %r" % got)

    def test_half_window_turn_not_amplified(self):
        """只有半个窗口在转时，聚合值应约为满幅的一半以下（不会被放大）。"""
        t = np.arange(0.0, 1.0, 0.002)
        v = np.where((t >= 0.25) & (t < 0.3), 1000.0, 0.0)   # 只转后半窗
        s = make_streams(t, v)
        got = B.omega_rad_s_interval(s, 0.2, 0.3, src_hz=30)
        self.assertIsNotNone(got)
        full = 1000.0 * abs(B.ANGULARY_RAD_PER_UNIT)
        self.assertLessEqual(abs(got), full * 1.01,
                             "聚合值不应超过单侧满幅")
        self.assertGreater(abs(got), 0.0, "后半窗在转，应被捕捉到")


class MissingFieldFallbackTests(unittest.TestCase):
    def test_no_angular_y_returns_none(self):
        s = dict(ay_t=np.zeros(0), ay_v=np.zeros(0))
        self.assertIsNone(B.omega_rad_s_interval(s, 0.0, 0.1, src_hz=30))

    def test_missing_keys_returns_none(self):
        self.assertIsNone(B.omega_rad_s_interval({}, 0.0, 0.1, src_hz=30))

    def test_all_stale_returns_none(self):
        t = np.array([0.0])
        v = np.array([100.0])
        s = make_streams(t, v)
        # 窗口离最后一条很远，超过 stale_s=5.0
        self.assertIsNone(B.omega_rad_s_interval(s, 100.0, 100.1, src_hz=30))


class ConstantTests(unittest.TestCase):
    def test_source_hz_is_30(self):
        self.assertEqual(B.OMEGA_SOURCE_HZ, 30)

    def test_freq_layers_are_distinct(self):
        # 视觉观测 5 Hz、动作标签 10 Hz、ω 源 30 Hz —— 三件事不能混
        self.assertAlmostEqual(B.GRID_S, 0.2, places=9)        # 5 Hz
        self.assertAlmostEqual(B.STEP_S, 0.1, places=9)        # 10 Hz
        self.assertEqual(B.K_STEPS, 12)
        self.assertEqual(B.OMEGA_SOURCE_HZ, 30)                # 30 Hz


if __name__ == "__main__":
    unittest.main()
