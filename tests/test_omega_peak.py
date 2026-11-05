# -*- coding: utf-8 -*-
"""peak 聚合的回归测试（2026-11-05）。

锁死四条：
  1. **符号**：peak 保留峰值采样自身的符号（不是永远取正/取 max）
  2. **短脉冲**：窗口内单个尖峰能被捕捉，而 mean 会被稀释
  3. **因果性**：扰动 t1 之后的数据，peak 结果不变
  4. **不抵消**：窗口内先急左再急右时，peak 不把两者相加抵消
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

import build_policy_dataset as B  # noqa: E402


def mk(t, v):
    return dict(ay_t=np.asarray(t, dtype=np.float64),
                ay_v=np.asarray(v, dtype=np.float64))


def ramp(lo, hi, val, period=0.002):
    """0..1 s，[lo,hi) 内取 val，其余 0。"""
    t = np.arange(0.0, 1.0, period)
    v = np.where((t >= lo) & (t < hi), val, 0.0)
    return mk(t, v)


class PeakSignTests(unittest.TestCase):
    def test_negative_raw_gives_positive_rad(self):
        s = ramp(0.2, 0.3, -256.0)
        got = B.omega_rad_s_interval(s, 0.2, 0.3, agg="peak")
        self.assertGreater(got, 0.0,
                           "ANGULARY_RAD_PER_UNIT<0，raw 负应得正 rad/s")

    def test_positive_raw_gives_negative_rad(self):
        s = ramp(0.2, 0.3, 256.0)
        got = B.omega_rad_s_interval(s, 0.2, 0.3, agg="peak")
        self.assertLess(got, 0.0)

    def test_peak_sign_matches_point_zoh(self):
        for raw in (-500.0, 500.0):
            s = ramp(0.2, 0.3, raw)
            pt = B.omega_rad_s_at(s, 0.25)
            pk = B.omega_rad_s_interval(s, 0.2, 0.3, agg="peak")
            self.assertEqual(np.sign(pt), np.sign(pk),
                             "raw=%r: peak 与单点符号不一致" % raw)


class PeakVsMeanTests(unittest.TestCase):
    def test_peak_keeps_short_spike_mean_dilutes(self):
        # 0.1s 窗口里只有 0.01s 的强脉冲，其余为 0
        s = ramp(0.245, 0.255, 1000.0)
        pk = B.omega_rad_s_interval(s, 0.2, 0.3, agg="peak")
        mn = B.omega_rad_s_interval(s, 0.2, 0.3, agg="mean")
        self.assertIsNotNone(pk)
        self.assertIsNotNone(mn)
        self.assertGreater(abs(pk), abs(mn) * 1.5,
                           "peak(%.4f) 应显著大于 mean(%.4f)" % (pk, mn))

    def test_peak_equals_max_abs_sample(self):
        t = np.arange(0.0, 1.0, 0.002)
        v = np.where((t >= 0.2) & (t < 0.25), 100.0,
                     np.where((t >= 0.25) & (t < 0.3), -900.0, 0.0))
        s = mk(t, v)
        pk = B.omega_rad_s_interval(s, 0.2, 0.3, agg="peak")
        expected = -900.0 * B.ANGULARY_RAD_PER_UNIT
        self.assertAlmostEqual(pk, expected, places=9,
                               msg="peak 应取 |·| 最大的 -900 并保留其符号")

    def test_peak_does_not_cancel_opposite_turns(self):
        t = np.arange(0.0, 1.0, 0.002)
        v = np.where((t >= 0.2) & (t < 0.25), 800.0,
                     np.where((t >= 0.25) & (t < 0.3), -800.0, 0.0))
        s = mk(t, v)
        pk = B.omega_rad_s_interval(s, 0.2, 0.3, agg="peak")
        self.assertGreater(abs(pk), 1.0,
                           "正反两次急转不应互相抵消成 0")


class PeakCausalityTests(unittest.TestCase):
    def test_future_perturbation_does_not_change_peak(self):
        s0 = ramp(0.2, 0.3, 500.0)
        base = B.omega_rad_s_interval(s0, 0.2, 0.3, agg="peak")
        t = s0["ay_t"]
        v = s0["ay_v"].copy()
        v[t >= 0.3] = -9999.0
        s1 = mk(t, v)
        got = B.omega_rad_s_interval(s1, 0.2, 0.3, agg="peak")
        self.assertAlmostEqual(base, got, places=12,
                               msg="窗口之后的数据影响了 peak，违反因果性")

    def test_missing_returns_none(self):
        self.assertIsNone(B.omega_rad_s_interval(dict(ay_t=np.zeros(0), ay_v=np.zeros(0)),
                                                  0.0, 0.1, agg="peak"))


class DefaultCompatTests(unittest.TestCase):
    def test_default_is_mean(self):
        self.assertEqual(B.OMEGA_AGG, "mean")
        self.assertEqual(B.OMEGA_AGG_MEAN, "mean")
        self.assertEqual(B.OMEGA_AGG_PEAK, "peak")

    def test_default_call_matches_explicit_mean(self):
        s = ramp(0.2, 0.3, 300.0)
        a = B.omega_rad_s_interval(s, 0.2, 0.3)
        b = B.omega_rad_s_interval(s, 0.2, 0.3, agg="mean")
        self.assertAlmostEqual(a, b, places=12)

class ManifestBuildCmdTests(unittest.TestCase):
    """manifest 必须记录生成命令（2026-11-05 审计要求）。

    动机：本轮两次出现「参数漏传导致静默错口径」（omega_src_hz、omega_agg），
    而 manifest 读的是 CLI 层对象所以「看起来正确」。把 argv 存进 manifest
    是唯一能事后自证「这份数据是用什么命令生成的」的证据。
    """

    def test_manifest_records_build_cmd(self):
        """构建脚本的 manifest 字典必须包含 build_cmd / build_argv。"""
        src = (ROOT / "research" / "tools" / "build_policy_dataset.py").read_text(encoding="utf-8")
        self.assertIn("build_cmd=build_cmd", src,
                      "manifest 未写 build_cmd，无法事后自证生成命令")
        self.assertIn("build_argv=list(_sys.argv)", src,
                      "manifest 未写 build_argv")

    def test_build_cmd_uses_real_argv(self):
        """build_cmd 必须来自 sys.argv，不能是硬编码字符串。"""
        src = (ROOT / "research" / "tools" / "build_policy_dataset.py").read_text(encoding="utf-8")
        self.assertIn('build_cmd = " ".join(_sys.argv)', src,
                      "build_cmd 不是从 sys.argv 生成，可能是硬编码")


if __name__ == "__main__":
    unittest.main()